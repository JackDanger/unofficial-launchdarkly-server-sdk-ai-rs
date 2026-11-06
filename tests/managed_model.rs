mod common;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use common::{test_context, FakeLdClient, FakeRunner, TEST_RUN_ID};
use serde_json::{json, Value};
use tokio::sync::Notify;
use unofficial_launchdarkly_server_sdk_ai::judge::build_evaluation_input;
use unofficial_launchdarkly_server_sdk_ai::{
    BoxError, Clock, Evaluator, Judge, LdAiCompletionConfig, LdAiConfigTracker,
    LdAiConfigTrackerParams, LdAiJudgeConfig, LdAiMetricSummary, LdAiMetrics, LdMessage,
    LdModelConfig, LdProviderConfig, LdTokenUsage, ManagedModel, Runner, RunnerResult,
    TrackerFactory,
};

const DEADLINE: Duration = Duration::from_secs(5);

fn stepping_clock(step_ms: f64) -> Clock {
    let ticks = AtomicU64::new(0);
    Arc::new(move || ticks.fetch_add(1, Ordering::SeqCst) as f64 * step_ms)
}

fn tracker_factory(client: &Arc<FakeLdClient>, config_key: &str) -> TrackerFactory {
    let ld_client = client.dyn_client();
    let config_key = config_key.to_string();
    TrackerFactory::new(move || {
        let mut params = LdAiConfigTrackerParams::new(TEST_RUN_ID, config_key.clone(), "v1", 1);
        params.model_name = "gpt-4".to_string();
        params.provider_name = "openai".to_string();
        LdAiConfigTracker::with_clock(
            Arc::clone(&ld_client),
            test_context(),
            params,
            stepping_clock(42.0),
        )
    })
}

fn completion_config(client: &Arc<FakeLdClient>, evaluator: Evaluator) -> LdAiCompletionConfig {
    LdAiCompletionConfig {
        key: "test-config".to_string(),
        enabled: true,
        model: Some(LdModelConfig::named("gpt-4")),
        provider: Some(LdProviderConfig::named("openai")),
        messages: Some(vec![LdMessage::system("You are a helpful assistant.")]),
        judge_configuration: None,
        tools: None,
        tracker_factory: tracker_factory(client, "test-config"),
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

fn content_runner(content: &str, metrics: LdAiMetrics) -> Arc<FakeRunner> {
    FakeRunner::returning(content, metrics)
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

mod managed_model {
    use super::*;

    #[tokio::test]
    async fn passes_the_prompt_directly_to_the_runner_without_prepending_config_messages() {
        let client = FakeLdClient::new();
        let runner = content_runner(
            "Response from model",
            LdAiMetrics {
                tokens: Some(LdTokenUsage {
                    total: 10,
                    input: 4,
                    output: 6,
                }),
                ..LdAiMetrics::new(true)
            },
        );
        let model = ManagedModel::new(
            completion_config(&client, Evaluator::noop()),
            runner.clone(),
            None,
        );

        model.run("Hello").await.unwrap();

        assert_eq!(
            *runner.calls.lock().unwrap(),
            vec![("Hello".to_string(), None)]
        );
    }

    #[tokio::test]
    async fn returns_a_managed_result_with_content_metrics_and_an_evaluations_promise() {
        let client = FakeLdClient::new();
        let runner = FakeRunner::new(|_, _| {
            Ok(RunnerResult {
                content: "Hi there".to_string(),
                metrics: LdAiMetrics {
                    success: true,
                    tokens: Some(LdTokenUsage {
                        total: 12,
                        input: 5,
                        output: 7,
                    }),
                    tool_calls: Some(vec!["tool-1".to_string()]),
                    duration_ms: Some(42.0),
                },
                raw: Some(json!({ "providerSpecific": true })),
                parsed: None,
            })
        });
        let config = completion_config(&client, Evaluator::noop());
        let expected_token = config.create_tracker().resumption_token();
        let model = ManagedModel::new(config, runner, None);

        let result = model.run("say hi").await.unwrap();

        assert_eq!(result.content, "Hi there");
        assert_eq!(
            result.metrics,
            LdAiMetricSummary {
                success: Some(true),
                tokens: Some(LdTokenUsage {
                    total: 12,
                    input: 5,
                    output: 7,
                }),
                tool_calls: Some(vec!["tool-1".to_string()]),
                duration_ms: Some(42.0),
                resumption_token: Some(expected_token),
                ..LdAiMetricSummary::default()
            }
        );
        assert_eq!(result.raw, Some(json!({ "providerSpecific": true })));
        assert_eq!(result.evaluations.await, vec![]);
    }

    #[tokio::test]
    async fn forwards_the_runner_result_through_tracker_track_metrics_of() {
        let client = FakeLdClient::new();
        let runner = content_runner(
            "tracked",
            LdAiMetrics {
                tokens: Some(LdTokenUsage {
                    total: 1,
                    input: 1,
                    output: 0,
                }),
                ..LdAiMetrics::new(true)
            },
        );
        let model = ManagedModel::new(completion_config(&client, Evaluator::noop()), runner, None);

        model.run("prompt").await.unwrap();

        let tracks: Vec<(String, Option<f64>)> = client
            .tracks()
            .into_iter()
            .map(|t| (t.key, t.metric_value))
            .collect();
        assert_eq!(
            tracks,
            vec![
                ("$ld:ai:duration:total".to_string(), Some(42.0)),
                ("$ld:ai:generation:success".to_string(), Some(1.0)),
                ("$ld:ai:tokens:total".to_string(), Some(1.0)),
                ("$ld:ai:tokens:input".to_string(), Some(1.0)),
            ]
        );
    }

    #[tokio::test]
    async fn does_not_retain_conversation_state_across_runs() {
        let client = FakeLdClient::new();
        let runner = content_runner("ok", LdAiMetrics::new(true));
        let model = ManagedModel::new(
            completion_config(&client, Evaluator::noop()),
            runner.clone(),
            None,
        );

        model.run("first").await.unwrap();
        model.run("second").await.unwrap();

        assert_eq!(runner.inputs(), vec!["first", "second"]);
    }

    #[tokio::test]
    async fn exposes_the_ai_config_via_get_config() {
        let client = FakeLdClient::new();
        let config = completion_config(&client, Evaluator::noop());
        let model = ManagedModel::new(
            config.clone(),
            content_runner("ok", LdAiMetrics::new(true)),
            None,
        );

        let exposed = model.get_config();
        assert_eq!(exposed.key, config.key);
        assert_eq!(exposed.enabled, config.enabled);
        assert_eq!(exposed.model, config.model);
        assert_eq!(exposed.provider, config.provider);
        assert_eq!(exposed.messages, config.messages);
    }
}

mod managed_model_run_evaluations {
    use super::*;

    fn model_runner() -> Arc<FakeRunner> {
        content_runner("AI response content", LdAiMetrics::new(true))
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn returns_before_evaluations_resolve() {
        let client = FakeLdClient::new();
        let gate = Arc::new(Notify::new());
        let judge_runner = Arc::new(GatedRunner {
            gate: Arc::clone(&gate),
            inner: scoring_runner(0.9),
        });
        let model = ManagedModel::new(
            completion_config(&client, judge_evaluator(&client, judge_runner)),
            model_runner(),
            None,
        );

        let mut result = model.run("Hello").await.unwrap();
        assert_eq!(result.content, "AI response content");

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

    #[tokio::test(flavor = "multi_thread")]
    async fn awaiting_evaluations_guarantees_tracking_is_complete() {
        let client = FakeLdClient::new();
        let model = ManagedModel::new(
            completion_config(&client, judge_evaluator(&client, scoring_runner(0.8))),
            model_runner(),
            None,
        );

        let result = model.run("Hello").await.unwrap();
        tokio::time::timeout(DEADLINE, result.evaluations)
            .await
            .unwrap();

        let judge_tracks = client.tracks_for("quality");
        assert_eq!(judge_tracks.len(), 1);
        assert_eq!(judge_tracks[0].metric_value, Some(0.8));
        let data = judge_tracks[0].data.as_ref().unwrap();
        assert_eq!(data["configKey"], "test-config");
        assert_eq!(data["judgeConfigKey"], "judge-1");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn builds_managed_result_with_correct_content_and_metrics() {
        let client = FakeLdClient::new();
        let config = completion_config(&client, Evaluator::noop());
        let expected_token = config.create_tracker().resumption_token();
        let model = ManagedModel::new(config, model_runner(), None);

        let result = model.run("test prompt").await.unwrap();

        assert_eq!(result.content, "AI response content");
        assert_eq!(result.metrics.success, Some(true));
        assert_eq!(result.metrics.resumption_token, Some(expected_token));
        assert_eq!(
            tokio::time::timeout(DEADLINE, result.evaluations)
                .await
                .unwrap(),
            vec![]
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn resolves_to_empty_evaluations_when_evaluator_is_noop() {
        let client = FakeLdClient::new();
        let model = ManagedModel::new(
            completion_config(&client, Evaluator::noop()),
            model_runner(),
            None,
        );

        let result = model.run("Hello").await.unwrap();

        assert_eq!(
            tokio::time::timeout(DEADLINE, result.evaluations)
                .await
                .unwrap(),
            vec![]
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn passes_the_prompt_to_evaluator_evaluate_as_input() {
        let client = FakeLdClient::new();
        let judge_runner = scoring_runner(1.0);
        let model = ManagedModel::new(
            completion_config(&client, judge_evaluator(&client, judge_runner.clone())),
            model_runner(),
            None,
        );

        let result = model.run("user prompt here").await.unwrap();
        tokio::time::timeout(DEADLINE, result.evaluations)
            .await
            .unwrap();

        assert_eq!(
            judge_runner.inputs(),
            vec![build_evaluation_input(
                "user prompt here",
                "AI response content"
            )]
        );
    }
}
