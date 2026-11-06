mod common;

use std::sync::Arc;

use common::{test_context, FakeLdClient, FakeRunner, TEST_RUN_ID};
use serde_json::json;
use unofficial_launchdarkly_server_sdk_ai::judge::build_evaluation_input;
use unofficial_launchdarkly_server_sdk_ai::{
    Evaluator, Judge, LdAiConfigTracker, LdAiConfigTrackerParams, LdAiJudgeConfig, LdJudgeResult,
    LdMessage, Runner, TrackerFactory,
};

const QUALITY: &str = "$ld:ai:judge:quality";

fn make_judge_config(client: &Arc<FakeLdClient>, key: &str) -> LdAiJudgeConfig {
    let ld_client = client.dyn_client();
    let tracker_key = key.to_string();
    LdAiJudgeConfig {
        key: key.to_string(),
        enabled: true,
        model: None,
        provider: None,
        messages: Some(vec![LdMessage::system("You are a judge.")]),
        evaluation_metric_key: Some(QUALITY.to_string()),
        tracker_factory: TrackerFactory::new(move || {
            LdAiConfigTracker::new(
                Arc::clone(&ld_client),
                test_context(),
                LdAiConfigTrackerParams::new(TEST_RUN_ID, tracker_key.clone(), "v1", 1),
            )
        }),
    }
}

fn make_judge(client: &Arc<FakeLdClient>, key: &str, runner: &Arc<FakeRunner>) -> Arc<Judge> {
    Arc::new(
        Judge::new(
            make_judge_config(client, key),
            Arc::clone(runner) as Arc<dyn Runner>,
            1.0,
            None,
        )
        .with_random_source(Arc::new(|| 0.5)),
    )
}

fn scoring_runner(score: f64, reasoning: &str) -> Arc<FakeRunner> {
    FakeRunner::parsed(json!({ "score": score, "reasoning": reasoning }), true)
}

mod noop {
    use super::*;

    #[tokio::test]
    async fn returns_an_empty_result_array() {
        let evaluator = Evaluator::noop();
        assert_eq!(evaluator.evaluate("input", "output").await, vec![]);
    }
}

mod evaluate {
    use super::*;

    #[tokio::test]
    async fn calls_each_configured_judge_and_returns_results() {
        let client = FakeLdClient::new();
        let runner = scoring_runner(0.9, "Good response");
        let evaluator = Evaluator::new(vec![make_judge(&client, "judge-1", &runner)]);

        let results = evaluator.evaluate("user input", "ai output").await;

        assert_eq!(
            results,
            vec![LdJudgeResult {
                success: true,
                sampled: true,
                score: Some(0.9),
                reasoning: Some("Good response".to_string()),
                metric_key: Some(QUALITY.to_string()),
                judge_config_key: Some("judge-1".to_string()),
                error_message: None,
            }]
        );
        assert_eq!(
            runner.inputs(),
            vec![build_evaluation_input("user input", "ai output")]
        );
    }

    #[tokio::test]
    async fn does_not_call_tracker_track_judge_result() {
        let client = FakeLdClient::new();
        let runner = scoring_runner(0.8, "ok");
        let evaluator = Evaluator::new(vec![make_judge(&client, "judge-1", &runner)]);

        let results = evaluator.evaluate("input", "output").await;

        assert_eq!(results[0].score, Some(0.8));
        assert!(client.tracks_for(QUALITY).is_empty());
        assert_eq!(
            client.track_keys(),
            vec!["$ld:ai:duration:total", "$ld:ai:generation:success"]
        );
    }

    #[tokio::test]
    async fn runs_multiple_judges_in_parallel_and_returns_all_results() {
        let client = FakeLdClient::new();
        let evaluator = Evaluator::new(vec![
            make_judge(&client, "judge-a", &scoring_runner(0.5, "ok")),
            make_judge(&client, "judge-b", &scoring_runner(0.9, "ok")),
        ]);

        let results = evaluator.evaluate("input", "output").await;

        assert_eq!(results.len(), 2);
        let mut scores: Vec<f64> = results.iter().filter_map(|r| r.score).collect();
        scores.sort_by(f64::total_cmp);
        assert_eq!(scores, vec![0.5, 0.9]);
    }
}
