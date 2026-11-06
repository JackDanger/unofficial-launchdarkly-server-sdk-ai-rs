mod common;

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use common::{test_context, FakeLdClient, FakeRunner, TEST_RUN_ID};
use serde_json::{json, Value};
use tokio::sync::Notify;
use unofficial_launchdarkly_server_sdk_ai::judge::build_evaluation_input;
use unofficial_launchdarkly_server_sdk_ai::{
    BoxError, Evaluator, Judge, LdAiAgentConfig, LdAiConfigTracker, LdAiConfigTrackerParams,
    LdAiJudgeConfig, LdAiMetrics, LdModelConfig, LdProviderConfig, ManagedAgent, Runner,
    RunnerResult, TrackerFactory,
};

const DEADLINE: Duration = Duration::from_secs(5);

fn tracker_factory(client: &Arc<FakeLdClient>, config_key: &str) -> TrackerFactory {
    let ld_client = client.dyn_client();
    let config_key = config_key.to_string();
    TrackerFactory::new(move || {
        LdAiConfigTracker::new(
            Arc::clone(&ld_client),
            test_context(),
            LdAiConfigTrackerParams::new(TEST_RUN_ID, config_key.clone(), "v1", 1),
        )
    })
}

fn agent_config(client: &Arc<FakeLdClient>, evaluator: Evaluator) -> LdAiAgentConfig {
    LdAiAgentConfig {
        key: "test-agent".to_string(),
        enabled: true,
        model: Some(LdModelConfig::named("gpt-4")),
        provider: Some(LdProviderConfig::named("openai")),
        instructions: Some("You are a helpful agent.".to_string()),
        judge_configuration: None,
        tools: None,
        tracker_factory: tracker_factory(client, "test-agent"),
        evaluator,
    }
}

fn judge_evaluator(client: &Arc<FakeLdClient>, runner: Arc<dyn Runner>) -> Evaluator {
    let config = LdAiJudgeConfig {
        key: "judge-1".to_string(),
        enabled: true,
        model: None,
        provider: None,
        messages: None,
        evaluation_metric_key: Some("quality".to_string()),
        tracker_factory: tracker_factory(client, "judge-1"),
    };
    Evaluator::new(vec![Arc::new(Judge::new(config, runner, 1.0, None))])
}

fn scoring_runner(score: f64) -> Arc<FakeRunner> {
    FakeRunner::parsed(json!({ "score": score, "reasoning": "ok" }), true)
}

fn agent_runner() -> Arc<FakeRunner> {
    FakeRunner::returning("Agent response", LdAiMetrics::new(true))
}

struct GatedRunner {
    gate: Arc<Notify>,
    inner: Arc<FakeRunner>,
}

#[async_trait]
impl Runner for GatedRunner {
    async fn run(
        &self,
        input: &str,
        output_type: Option<&Value>,
    ) -> Result<RunnerResult, BoxError> {
        self.gate.notified().await;
        self.inner.run(input, output_type).await
    }
}

mod managed_agent {
    use super::*;

    #[tokio::test]
    async fn returns_a_managed_result_with_content_and_metrics() {
        let client = FakeLdClient::new();
        let config = agent_config(&client, Evaluator::noop());
        let expected_token = config.create_tracker().resumption_token();
        let agent = ManagedAgent::new(config, agent_runner(), None);

        let result = agent.run("Hello agent").await.unwrap();

        assert_eq!(result.content, "Agent response");
        assert_eq!(result.metrics.success, Some(true));
        assert_eq!(result.metrics.resumption_token, Some(expected_token));
    }

    #[tokio::test]
    async fn passes_the_prompt_directly_to_the_runner() {
        let client = FakeLdClient::new();
        let runner = agent_runner();
        let agent = ManagedAgent::new(
            agent_config(&client, Evaluator::noop()),
            runner.clone(),
            None,
        );

        agent.run("My question").await.unwrap();

        assert_eq!(
            *runner.calls.lock().unwrap(),
            vec![("My question".to_string(), None)]
        );
    }

    #[tokio::test]
    async fn resolves_to_empty_evaluations_with_noop_evaluator() {
        let client = FakeLdClient::new();
        let agent = ManagedAgent::new(
            agent_config(&client, Evaluator::noop()),
            agent_runner(),
            None,
        );

        let result = agent.run("Hello").await.unwrap();

        assert_eq!(
            tokio::time::timeout(DEADLINE, result.evaluations)
                .await
                .unwrap(),
            vec![]
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn awaiting_evaluations_calls_tracker_track_judge_result() {
        let client = FakeLdClient::new();
        let agent = ManagedAgent::new(
            agent_config(&client, judge_evaluator(&client, scoring_runner(0.85))),
            agent_runner(),
            None,
        );

        let result = agent.run("Hello").await.unwrap();
        tokio::time::timeout(DEADLINE, result.evaluations)
            .await
            .unwrap();

        let judge_tracks = client.tracks_for("quality");
        assert_eq!(judge_tracks.len(), 1);
        assert_eq!(judge_tracks[0].metric_value, Some(0.85));
        let data = judge_tracks[0].data.as_ref().unwrap();
        assert_eq!(data["configKey"], "test-agent");
        assert_eq!(data["judgeConfigKey"], "judge-1");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn passes_the_prompt_to_evaluator_evaluate_as_input() {
        let client = FakeLdClient::new();
        let judge_runner = scoring_runner(0.5);
        let agent = ManagedAgent::new(
            agent_config(&client, judge_evaluator(&client, judge_runner.clone())),
            agent_runner(),
            None,
        );

        let result = agent.run("user prompt").await.unwrap();
        tokio::time::timeout(DEADLINE, result.evaluations)
            .await
            .unwrap();

        assert_eq!(
            judge_runner.inputs(),
            vec![build_evaluation_input("user prompt", "Agent response")]
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn returns_before_evaluations_resolve() {
        let client = FakeLdClient::new();
        let gate = Arc::new(Notify::new());
        let judge_runner = Arc::new(GatedRunner {
            gate: Arc::clone(&gate),
            inner: scoring_runner(0.9),
        });
        let agent = ManagedAgent::new(
            agent_config(&client, judge_evaluator(&client, judge_runner)),
            agent_runner(),
            None,
        );

        let mut result = agent.run("Hello").await.unwrap();
        assert_eq!(result.content, "Agent response");

        let pending =
            tokio::time::timeout(Duration::from_millis(50), &mut result.evaluations).await;
        assert!(pending.is_err());
        assert!(client.tracks_for("quality").is_empty());

        gate.notify_one();
        let evaluations = tokio::time::timeout(DEADLINE, result.evaluations)
            .await
            .expect("evaluations resolve once the judge is released");
        assert_eq!(evaluations.len(), 1);
        assert_eq!(evaluations[0].score, Some(0.9));
        assert_eq!(client.tracks_for("quality").len(), 1);
    }

    #[tokio::test]
    async fn exposes_the_agent_config_via_get_config() {
        let client = FakeLdClient::new();
        let config = agent_config(&client, Evaluator::noop());
        let agent = ManagedAgent::new(config.clone(), agent_runner(), None);

        let exposed = agent.get_config();
        assert_eq!(exposed.key, config.key);
        assert_eq!(exposed.enabled, config.enabled);
        assert_eq!(exposed.model, config.model);
        assert_eq!(exposed.provider, config.provider);
        assert_eq!(exposed.instructions, config.instructions);
    }
}
