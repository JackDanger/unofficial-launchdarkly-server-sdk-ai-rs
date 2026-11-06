mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use base64::Engine;
use common::{test_context, FakeLdClient, TEST_RUN_ID};
use serde_json::{json, Value};
use unofficial_launchdarkly_server_sdk_ai::{
    Clock, LdAiConfigTracker, LdAiConfigTrackerParams, LdAiMetrics, LdFeedbackKind, LdJudgeResult,
    LdTokenUsage,
};

const CONFIG_KEY: &str = "test-config";
const VARIATION_KEY: &str = "v1";
const VERSION: u64 = 1;
const MODEL_NAME: &str = "test-model";
const PROVIDER_NAME: &str = "test-provider";

fn expected_track_data() -> Value {
    json!({
        "configKey": CONFIG_KEY,
        "variationKey": VARIATION_KEY,
        "version": VERSION,
        "modelName": MODEL_NAME,
        "providerName": PROVIDER_NAME,
        "modelVersion": 1,
        "runId": TEST_RUN_ID,
    })
}

fn expected_track_data_with(extra: Value) -> Value {
    let mut data = expected_track_data();
    if let (Value::Object(map), Value::Object(extra)) = (&mut data, extra) {
        map.extend(extra);
    }
    data
}

struct TrackerSpec<'a> {
    variation_key: &'a str,
    graph_key: Option<&'a str>,
    model_key: Option<&'a str>,
    model_version: u64,
}

impl Default for TrackerSpec<'_> {
    fn default() -> Self {
        Self {
            variation_key: VARIATION_KEY,
            graph_key: None,
            model_key: None,
            model_version: 1,
        }
    }
}

fn params(spec: TrackerSpec<'_>) -> LdAiConfigTrackerParams {
    let mut params =
        LdAiConfigTrackerParams::new(TEST_RUN_ID, CONFIG_KEY, spec.variation_key, VERSION);
    params.model_name = MODEL_NAME.to_string();
    params.provider_name = PROVIDER_NAME.to_string();
    params.graph_key = spec.graph_key.map(str::to_string);
    params.model_key = spec.model_key.map(str::to_string);
    params.model_version = spec.model_version;
    params
}

fn tracker_with(client: &Arc<FakeLdClient>, spec: TrackerSpec<'_>) -> LdAiConfigTracker {
    LdAiConfigTracker::new(client.dyn_client(), test_context(), params(spec))
}

fn tracker(client: &Arc<FakeLdClient>) -> LdAiConfigTracker {
    tracker_with(client, TrackerSpec::default())
}

fn graph_tracker(client: &Arc<FakeLdClient>) -> LdAiConfigTracker {
    tracker_with(
        client,
        TrackerSpec {
            graph_key: Some("my-graph"),
            ..TrackerSpec::default()
        },
    )
}

fn scripted_clock(values: &'static [f64]) -> Clock {
    let next = AtomicUsize::new(0);
    Arc::new(move || values[next.fetch_add(1, Ordering::SeqCst).min(values.len() - 1)])
}

fn clocked_tracker(client: &Arc<FakeLdClient>, values: &'static [f64]) -> LdAiConfigTracker {
    LdAiConfigTracker::with_clock(
        client.dyn_client(),
        test_context(),
        params(TrackerSpec::default()),
        scripted_clock(values),
    )
}

fn assert_tracked(client: &FakeLdClient, key: &str, data: Value, metric_value: f64) {
    let calls = client.tracks();
    let found = calls.iter().any(|call| {
        call.key == key
            && call.context == test_context()
            && call.data.as_ref() == Some(&data)
            && call.metric_value == Some(metric_value)
    });
    assert!(
        found,
        "expected track({key}, test-user, {data}, {metric_value}); got {calls:#?}"
    );
}

fn assert_not_tracked(client: &FakeLdClient, key: &str) {
    assert!(
        client.tracks_for(key).is_empty(),
        "expected no {key} event; got {:#?}",
        client.tracks()
    );
}

fn decode(token: &str) -> Value {
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(token)
        .expect("token is URL-safe unpadded base64");
    serde_json::from_slice(&bytes).expect("token is JSON")
}

fn track_data_json(tracker: &LdAiConfigTracker) -> Value {
    serde_json::to_value(tracker.get_track_data()).unwrap()
}

fn tokens(total: u64, input: u64, output: u64) -> LdTokenUsage {
    LdTokenUsage {
        total,
        input,
        output,
    }
}

#[test]
fn tracks_duration() {
    let client = FakeLdClient::new();
    tracker(&client).track_duration(1000.0);

    assert_tracked(
        &client,
        "$ld:ai:duration:total",
        expected_track_data(),
        1000.0,
    );
}

#[tokio::test]
async fn tracks_duration_of_async_function() {
    let client = FakeLdClient::new();
    let tracker = clocked_tracker(&client, &[1000.0, 2000.0]);

    let result = tracker.track_duration_of(|| async { "test-result" }).await;

    assert_eq!(result, "test-result");
    assert_tracked(
        &client,
        "$ld:ai:duration:total",
        expected_track_data(),
        1000.0,
    );
}

#[test]
fn tracks_time_to_first_token() {
    let client = FakeLdClient::new();
    tracker(&client).track_time_to_first_token(1000.0);

    assert_tracked(&client, "$ld:ai:tokens:ttf", expected_track_data(), 1000.0);
}

#[test]
fn tracks_positive_feedback() {
    let client = FakeLdClient::new();
    tracker(&client).track_feedback(LdFeedbackKind::Positive);

    assert_tracked(
        &client,
        "$ld:ai:feedback:user:positive",
        expected_track_data(),
        1.0,
    );
}

#[test]
fn tracks_negative_feedback() {
    let client = FakeLdClient::new();
    tracker(&client).track_feedback(LdFeedbackKind::Negative);

    assert_tracked(
        &client,
        "$ld:ai:feedback:user:negative",
        expected_track_data(),
        1.0,
    );
}

#[test]
fn tracks_success() {
    let client = FakeLdClient::new();
    tracker(&client).track_success();

    assert_tracked(
        &client,
        "$ld:ai:generation:success",
        expected_track_data(),
        1.0,
    );
}

#[test]
fn tracks_tokens() {
    let client = FakeLdClient::new();
    tracker(&client).track_tokens(tokens(100, 49, 51));

    assert_tracked(&client, "$ld:ai:tokens:total", expected_track_data(), 100.0);
    assert_tracked(&client, "$ld:ai:tokens:input", expected_track_data(), 49.0);
    assert_tracked(&client, "$ld:ai:tokens:output", expected_track_data(), 51.0);
}

#[test]
fn only_tracks_non_zero_token_counts() {
    let client = FakeLdClient::new();
    tracker(&client).track_tokens(tokens(0, 50, 0));

    assert_not_tracked(&client, "$ld:ai:tokens:total");
    assert_tracked(&client, "$ld:ai:tokens:input", expected_track_data(), 50.0);
    assert_not_tracked(&client, "$ld:ai:tokens:output");
}

#[test]
fn returns_summary_with_resumption_token_immediately_after_construction() {
    let client = FakeLdClient::new();
    let tracker = tracker(&client);

    let summary = tracker.get_summary();

    assert_eq!(summary.resumption_token, Some(tracker.resumption_token()));
    assert!(summary.success.is_none());
    assert!(summary.tokens.is_none());
    assert!(summary.duration_ms.is_none());
}

#[test]
fn summarizes_tracked_metrics() {
    let client = FakeLdClient::new();
    let tracker = tracker(&client);

    tracker.track_duration(1000.0);
    tracker.track_tokens(tokens(100, 40, 60));
    tracker.track_feedback(LdFeedbackKind::Positive);
    tracker.track_success();

    assert_eq!(
        serde_json::to_value(tracker.get_summary()).unwrap(),
        json!({
            "resumptionToken": tracker.resumption_token(),
            "durationMs": 1000.0,
            "tokens": { "total": 100, "input": 40, "output": 60 },
            "feedback": { "kind": "positive" },
            "success": true,
        })
    );
}

#[test]
fn accumulates_tool_calls_in_get_summary_after_track_tool_call() {
    let client = FakeLdClient::new();
    let tracker = tracker(&client);

    tracker.track_tool_call("tool-a");
    tracker.track_tool_call("tool-b");

    assert_eq!(
        tracker.get_summary().tool_calls,
        Some(vec!["tool-a".to_string(), "tool-b".to_string()])
    );
}

#[tokio::test]
async fn tracks_duration_when_async_function_throws() {
    let client = FakeLdClient::new();
    let tracker = clocked_tracker(&client, &[1000.0, 2000.0]);

    let result: Result<(), String> = tracker
        .track_duration_of(|| async { Err("test error".to_string()) })
        .await;

    assert_eq!(result, Err("test error".to_string()));
    assert_tracked(
        &client,
        "$ld:ai:duration:total",
        expected_track_data(),
        1000.0,
    );
}

#[test]
fn tracks_error() {
    let client = FakeLdClient::new();
    tracker(&client).track_error();

    assert_tracked(
        &client,
        "$ld:ai:generation:error",
        expected_track_data(),
        1.0,
    );
}

mod track_metrics_of {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct MockResult {
        response: &'static str,
    }

    const MOCK_RESULT: MockResult = MockResult { response: "test" };

    #[tokio::test]
    async fn tracks_success_and_token_usage_from_metrics() {
        let client = FakeLdClient::new();
        let tracker = tracker(&client);
        let extracted_from = std::sync::Mutex::new(None);
        let operation_called = AtomicBool::new(false);

        let result = tracker
            .track_metrics_of(
                |result: &MockResult| {
                    *extracted_from.lock().unwrap() = Some(result.clone());
                    LdAiMetrics {
                        tokens: Some(tokens(100, 50, 50)),
                        ..LdAiMetrics::new(true)
                    }
                },
                || async {
                    operation_called.store(true, Ordering::SeqCst);
                    Ok::<_, String>(MOCK_RESULT)
                },
            )
            .await;

        assert_eq!(result, Ok(MOCK_RESULT));
        assert_eq!(*extracted_from.lock().unwrap(), Some(MOCK_RESULT));
        assert!(operation_called.load(Ordering::SeqCst));
        assert_tracked(
            &client,
            "$ld:ai:generation:success",
            expected_track_data(),
            1.0,
        );
        assert_tracked(&client, "$ld:ai:tokens:total", expected_track_data(), 100.0);
        assert_tracked(&client, "$ld:ai:tokens:input", expected_track_data(), 50.0);
        assert_tracked(&client, "$ld:ai:tokens:output", expected_track_data(), 50.0);
    }

    #[tokio::test]
    async fn tracks_failure_when_metrics_indicate_failure() {
        let client = FakeLdClient::new();
        let tracker = tracker(&client);

        let _ = tracker
            .track_metrics_of(
                |_: &MockResult| LdAiMetrics::new(false),
                || async { Ok::<_, String>(MOCK_RESULT) },
            )
            .await;

        assert_tracked(
            &client,
            "$ld:ai:generation:error",
            expected_track_data(),
            1.0,
        );
    }

    #[tokio::test]
    async fn tracks_failure_when_operation_throws() {
        let client = FakeLdClient::new();
        let tracker = tracker(&client);
        let extractor_called = AtomicBool::new(false);

        let result = tracker
            .track_metrics_of(
                |_: &MockResult| {
                    extractor_called.store(true, Ordering::SeqCst);
                    LdAiMetrics::new(true)
                },
                || async { Err::<MockResult, _>("Operation failed".to_string()) },
            )
            .await;

        assert_eq!(result, Err("Operation failed".to_string()));
        assert_tracked(
            &client,
            "$ld:ai:generation:error",
            expected_track_data(),
            1.0,
        );
        assert!(!extractor_called.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn tracks_metrics_without_token_usage() {
        let client = FakeLdClient::new();
        let tracker = tracker(&client);

        let _ = tracker
            .track_metrics_of(
                |_: &MockResult| LdAiMetrics::new(true),
                || async { Ok::<_, String>(MOCK_RESULT) },
            )
            .await;

        assert_tracked(
            &client,
            "$ld:ai:generation:success",
            expected_track_data(),
            1.0,
        );
        assert_not_tracked(&client, "$ld:ai:tokens:total");
    }
}

mod track_judge_result {
    use super::*;

    #[test]
    fn tracks_metric_key_with_score() {
        let client = FakeLdClient::new();
        tracker(&client).track_judge_result(&LdJudgeResult {
            judge_config_key: Some("test-judge".to_string()),
            success: true,
            sampled: true,
            score: Some(0.8),
            reasoning: Some("The response is relevant".to_string()),
            metric_key: Some("relevance".to_string()),
            ..LdJudgeResult::default()
        });

        assert_tracked(
            &client,
            "relevance",
            expected_track_data_with(json!({ "judgeConfigKey": "test-judge" })),
            0.8,
        );
    }

    #[test]
    fn does_not_track_when_sampled_is_false() {
        let client = FakeLdClient::new();
        tracker(&client).track_judge_result(&LdJudgeResult {
            judge_config_key: Some("test-judge".to_string()),
            success: false,
            sampled: false,
            score: Some(0.8),
            metric_key: Some("relevance".to_string()),
            ..LdJudgeResult::default()
        });

        assert!(client.tracks().is_empty());
    }

    #[test]
    fn does_not_track_when_success_is_false() {
        let client = FakeLdClient::new();
        tracker(&client).track_judge_result(&LdJudgeResult {
            judge_config_key: Some("test-judge".to_string()),
            success: false,
            sampled: true,
            score: Some(0.8),
            metric_key: Some("relevance".to_string()),
            ..LdJudgeResult::default()
        });

        assert!(client.tracks().is_empty());
    }
}

mod track_tool_call {
    use super::*;

    #[test]
    fn tracks_a_single_tool_call() {
        let client = FakeLdClient::new();
        tracker(&client).track_tool_call("my-tool");

        assert_tracked(
            &client,
            "$ld:ai:tool_call",
            expected_track_data_with(json!({ "toolKey": "my-tool" })),
            1.0,
        );
    }

    #[test]
    fn includes_graph_key_when_set_on_constructor() {
        let client = FakeLdClient::new();
        graph_tracker(&client).track_tool_call("my-tool");

        assert_tracked(
            &client,
            "$ld:ai:tool_call",
            expected_track_data_with(json!({ "graphKey": "my-graph", "toolKey": "my-tool" })),
            1.0,
        );
    }
}

mod track_tool_calls {
    use super::*;

    #[test]
    fn tracks_multiple_tool_calls() {
        let client = FakeLdClient::new();
        tracker(&client).track_tool_calls(&["tool-a", "tool-b", "tool-c"]);

        assert_eq!(client.tracks().len(), 3);
        for tool in ["tool-a", "tool-b", "tool-c"] {
            assert_tracked(
                &client,
                "$ld:ai:tool_call",
                expected_track_data_with(json!({ "toolKey": tool })),
                1.0,
            );
        }
    }
}

mod track_stream_metrics_of {
    use super::*;

    #[tokio::test]
    async fn tracks_tool_calls_from_streaming_metrics_extractor() {
        let client = FakeLdClient::new();
        let tracker = tracker(&client);

        let stream = tracker.track_stream_metrics_of(
            || Ok::<_, String>("mock-stream"),
            |_stream| async {
                Ok::<_, String>(LdAiMetrics {
                    tool_calls: Some(vec!["tool-a".to_string(), "tool-b".to_string()]),
                    ..LdAiMetrics::new(true)
                })
            },
        );
        assert_eq!(stream, Ok("mock-stream"));

        tokio::time::timeout(Duration::from_secs(5), async {
            while tracker.get_summary().duration_ms.is_none() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("background stream metrics finish");

        assert_eq!(
            tracker.get_summary().tool_calls,
            Some(vec!["tool-a".to_string(), "tool-b".to_string()])
        );
        assert_tracked(
            &client,
            "$ld:ai:tool_call",
            expected_track_data_with(json!({ "toolKey": "tool-a" })),
            1.0,
        );
        assert_tracked(
            &client,
            "$ld:ai:tool_call",
            expected_track_data_with(json!({ "toolKey": "tool-b" })),
            1.0,
        );
    }
}

mod graph_key_constructor_support {
    use super::*;

    #[test]
    fn includes_graph_key_in_track_duration_event_when_set_on_constructor() {
        let client = FakeLdClient::new();
        graph_tracker(&client).track_duration(1000.0);

        assert_tracked(
            &client,
            "$ld:ai:duration:total",
            expected_track_data_with(json!({ "graphKey": "my-graph" })),
            1000.0,
        );
    }

    #[test]
    fn includes_graph_key_in_track_success_event_when_set_on_constructor() {
        let client = FakeLdClient::new();
        graph_tracker(&client).track_success();

        assert_tracked(
            &client,
            "$ld:ai:generation:success",
            expected_track_data_with(json!({ "graphKey": "my-graph" })),
            1.0,
        );
    }

    #[test]
    fn does_not_include_graph_key_when_not_set_on_constructor() {
        let client = FakeLdClient::new();
        tracker(&client).track_success();

        assert_tracked(
            &client,
            "$ld:ai:generation:success",
            expected_track_data(),
            1.0,
        );
    }

    #[test]
    fn includes_graph_key_in_get_track_data_when_set_on_constructor() {
        let client = FakeLdClient::new();

        assert_eq!(
            track_data_json(&graph_tracker(&client)),
            expected_track_data_with(json!({ "graphKey": "my-graph" }))
        );
    }

    #[test]
    fn does_not_include_graph_key_in_get_track_data_when_not_set() {
        let client = FakeLdClient::new();
        let data = track_data_json(&tracker(&client));

        assert_eq!(data, expected_track_data());
        assert!(data.get("graphKey").is_none());
    }
}

mod at_most_once_semantics {
    use super::*;

    fn assert_one_track_one_warning(client: &FakeLdClient, tracks: usize) {
        assert_eq!(client.tracks().len(), tracks);
        assert_eq!(client.logger.warnings().len(), 1);
    }

    #[test]
    fn drops_duplicate_track_duration_call_with_warning() {
        let client = FakeLdClient::new();
        let tracker = tracker(&client);
        tracker.track_duration(1000.0);
        tracker.track_duration(2000.0);

        assert_one_track_one_warning(&client, 1);
        assert!(client.logger.warnings()[0].contains("trackDuration"));
    }

    #[test]
    fn drops_duplicate_track_success_call_with_warning() {
        let client = FakeLdClient::new();
        let tracker = tracker(&client);
        tracker.track_success();
        tracker.track_success();

        assert_one_track_one_warning(&client, 1);
    }

    #[test]
    fn drops_track_error_call_after_track_success_with_warning() {
        let client = FakeLdClient::new();
        let tracker = tracker(&client);
        tracker.track_success();
        tracker.track_error();

        assert_one_track_one_warning(&client, 1);
    }

    #[test]
    fn drops_duplicate_track_feedback_call_with_warning() {
        let client = FakeLdClient::new();
        let tracker = tracker(&client);
        tracker.track_feedback(LdFeedbackKind::Positive);
        tracker.track_feedback(LdFeedbackKind::Negative);

        assert_one_track_one_warning(&client, 1);
    }

    #[test]
    fn drops_duplicate_track_tokens_call_with_warning() {
        let client = FakeLdClient::new();
        let tracker = tracker(&client);
        tracker.track_tokens(tokens(100, 50, 50));
        tracker.track_tokens(tokens(200, 100, 100));

        assert_one_track_one_warning(&client, 3);
    }

    #[test]
    fn drops_duplicate_track_time_to_first_token_call_with_warning() {
        let client = FakeLdClient::new();
        let tracker = tracker(&client);
        tracker.track_time_to_first_token(100.0);
        tracker.track_time_to_first_token(200.0);

        assert_one_track_one_warning(&client, 1);
    }
}

mod resumption_token {
    use super::*;

    #[test]
    fn encodes_run_id_config_key_variation_key_and_version_as_url_safe_base64_json() {
        let client = FakeLdClient::new();
        let decoded = decode(&tracker(&client).resumption_token());

        assert_eq!(
            decoded,
            json!({
                "runId": TEST_RUN_ID,
                "configKey": CONFIG_KEY,
                "variationKey": VARIATION_KEY,
                "version": VERSION,
            })
        );
    }

    #[test]
    fn includes_empty_variation_key_explicitly_when_not_set() {
        let client = FakeLdClient::new();
        let tracker = tracker_with(
            &client,
            TrackerSpec {
                variation_key: "",
                ..TrackerSpec::default()
            },
        );
        let decoded = decode(&tracker.resumption_token());

        assert_eq!(
            decoded,
            json!({
                "runId": TEST_RUN_ID,
                "configKey": CONFIG_KEY,
                "variationKey": "",
                "version": VERSION,
            })
        );
        assert!(decoded.get("variationKey").is_some());
    }

    #[test]
    fn uses_url_safe_base64_encoding_no_plus_slash_or_equals_characters() {
        let client = FakeLdClient::new();
        let token = tracker(&client).resumption_token();

        assert!(!token.contains(['+', '/', '=']), "token {token}");
    }
}

mod from_resumption_token {
    use super::*;

    fn reconstruct(client: &Arc<FakeLdClient>, original: &LdAiConfigTracker) -> LdAiConfigTracker {
        LdAiConfigTracker::from_resumption_token(
            &original.resumption_token(),
            client.dyn_client(),
            test_context(),
        )
        .expect("valid token")
    }

    #[test]
    fn reconstructs_tracker_with_original_run_id() {
        let client = FakeLdClient::new();
        let reconstructed = reconstruct(&client, &tracker(&client));
        let data = reconstructed.get_track_data();

        assert_eq!(data.run_id, TEST_RUN_ID);
        assert_eq!(data.config_key, CONFIG_KEY);
        assert_eq!(data.variation_key, VARIATION_KEY);
        assert_eq!(data.version, VERSION);
    }

    #[test]
    fn reconstructs_tracker_with_empty_variation_key_when_none_was_set() {
        let client = FakeLdClient::new();
        let original = tracker_with(
            &client,
            TrackerSpec {
                variation_key: "",
                ..TrackerSpec::default()
            },
        );

        assert_eq!(
            reconstruct(&client, &original)
                .get_track_data()
                .variation_key,
            ""
        );
    }

    #[test]
    fn reconstructed_tracker_emits_track_events_with_original_run_id() {
        let client = FakeLdClient::new();
        reconstruct(&client, &tracker(&client)).track_success();

        let calls = client.tracks_for("$ld:ai:generation:success");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].context, test_context());
        assert_eq!(calls[0].data.as_ref().unwrap()["runId"], json!(TEST_RUN_ID));
        assert_eq!(calls[0].metric_value, Some(1.0));
    }

    #[test]
    fn includes_graph_key_in_resumption_token_when_set_on_constructor() {
        let client = FakeLdClient::new();
        let decoded = decode(&graph_tracker(&client).resumption_token());

        assert_eq!(
            decoded,
            json!({
                "runId": TEST_RUN_ID,
                "configKey": CONFIG_KEY,
                "variationKey": VARIATION_KEY,
                "version": VERSION,
                "graphKey": "my-graph",
            })
        );
    }

    #[test]
    fn does_not_include_graph_key_in_resumption_token_when_not_set() {
        let client = FakeLdClient::new();
        let decoded = decode(&tracker(&client).resumption_token());

        assert_eq!(
            decoded,
            json!({
                "runId": TEST_RUN_ID,
                "configKey": CONFIG_KEY,
                "variationKey": VARIATION_KEY,
                "version": VERSION,
            })
        );
        assert!(decoded.get("graphKey").is_none());
    }

    #[test]
    fn reconstructs_tracker_with_graph_key_from_resumption_token() {
        let client = FakeLdClient::new();
        let reconstructed = reconstruct(&client, &graph_tracker(&client));

        assert_eq!(
            reconstructed.get_track_data().graph_key.as_deref(),
            Some("my-graph")
        );
    }

    #[test]
    fn reconstructed_tracker_without_graph_key_does_not_include_graph_key_in_track_data() {
        let client = FakeLdClient::new();
        let reconstructed = reconstruct(&client, &tracker(&client));

        assert!(track_data_json(&reconstructed).get("graphKey").is_none());
    }
}

fn model_tracker(
    client: &Arc<FakeLdClient>,
    model_key: &str,
    model_version: u64,
) -> LdAiConfigTracker {
    tracker_with(
        client,
        TrackerSpec {
            model_key: Some(model_key),
            model_version,
            ..TrackerSpec::default()
        },
    )
}

#[test]
fn includes_model_key_and_model_version_in_get_track_data_when_set_on_constructor() {
    let client = FakeLdClient::new();

    assert_eq!(
        track_data_json(&model_tracker(&client, "my-model", 2)),
        expected_track_data_with(json!({ "modelKey": "my-model", "modelVersion": 2 }))
    );
}

#[test]
fn omits_model_key_from_get_track_data_when_unset() {
    let client = FakeLdClient::new();
    let data = track_data_json(&tracker(&client));

    assert!(data.get("modelKey").is_none());
    assert_eq!(data["modelVersion"], json!(1));
}

#[test]
fn omits_model_key_from_get_track_data_when_empty_string() {
    let client = FakeLdClient::new();
    let data = track_data_json(&model_tracker(&client, "", 3));

    assert!(data.get("modelKey").is_none());
    assert_eq!(data["modelVersion"], json!(3));
}

#[test]
fn excludes_model_key_and_model_version_from_resumption_token() {
    let client = FakeLdClient::new();
    let decoded = decode(&model_tracker(&client, "my-model", 2).resumption_token());

    assert!(decoded.get("modelKey").is_none());
    assert!(decoded.get("modelVersion").is_none());
}

#[test]
fn from_resumption_token_defaults_model_version_to_1_and_omits_model_key() {
    let client = FakeLdClient::new();
    let original = model_tracker(&client, "my-model", 2);
    let reconstructed = LdAiConfigTracker::from_resumption_token(
        &original.resumption_token(),
        client.dyn_client(),
        test_context(),
    )
    .expect("valid token");
    let data = track_data_json(&reconstructed);

    assert!(data.get("modelKey").is_none());
    assert_eq!(data["modelVersion"], json!(1));
}
