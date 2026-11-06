mod common;

use std::sync::Arc;

use base64::Engine;
use common::{test_context, FakeLdClient};
use serde_json::{json, Value};
use unofficial_launchdarkly_server_sdk_ai::{LdGraphTracker, LdTokenUsage};

const GRAPH_KEY: &str = "my-agent-graph";
const VARIATION_KEY: &str = "v1";
const VERSION: u64 = 2;

fn make_tracker(client: &Arc<FakeLdClient>, run_id: &str) -> LdGraphTracker {
    LdGraphTracker::new(
        client.dyn_client(),
        run_id,
        GRAPH_KEY,
        Some(VARIATION_KEY.to_string()),
        VERSION,
        test_context(),
    )
}

fn tracker_without_variation(client: &Arc<FakeLdClient>, run_id: &str) -> LdGraphTracker {
    LdGraphTracker::new(
        client.dyn_client(),
        run_id,
        GRAPH_KEY,
        None,
        VERSION,
        test_context(),
    )
}

fn track_data_json(tracker: &LdGraphTracker) -> Value {
    serde_json::to_value(tracker.get_track_data()).unwrap()
}

fn track_data_with(tracker: &LdGraphTracker, extra: Value) -> Value {
    let mut data = track_data_json(tracker);
    if let (Value::Object(map), Value::Object(extra)) = (&mut data, extra) {
        map.extend(extra);
    }
    data
}

fn decoded_token_string(tracker: &LdGraphTracker) -> String {
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(tracker.resumption_token())
        .expect("token is URL-safe unpadded base64");
    String::from_utf8(bytes).expect("token is UTF-8")
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

#[test]
fn returns_correct_track_data_with_variation_key() {
    let client = FakeLdClient::new();
    let tracker = make_tracker(&client, "fixed-run-id");

    assert_eq!(
        track_data_json(&tracker),
        json!({
            "runId": "fixed-run-id",
            "graphKey": GRAPH_KEY,
            "version": VERSION,
            "variationKey": VARIATION_KEY,
        })
    );
}

#[test]
fn omits_variation_key_when_not_provided() {
    let client = FakeLdClient::new();
    let data = tracker_without_variation(&client, "some-run-id").get_track_data();

    assert_eq!(data.variation_key, None);
    assert_eq!(data.graph_key, GRAPH_KEY);
    assert_eq!(data.version, VERSION);
    assert_eq!(data.run_id, "some-run-id");
}

#[test]
fn uses_provided_run_id() {
    let client = FakeLdClient::new();

    assert_eq!(
        make_tracker(&client, "my-custom-run-id")
            .get_track_data()
            .run_id,
        "my-custom-run-id"
    );
}

#[test]
fn encodes_a_resumption_token_with_correct_field_order() {
    let client = FakeLdClient::new();
    let tracker = make_tracker(&client, "550e8400-e29b-41d4-a716-446655440000");

    assert_eq!(
        decoded_token_string(&tracker),
        r#"{"runId":"550e8400-e29b-41d4-a716-446655440000","graphKey":"my-agent-graph","variationKey":"v1","version":2}"#
    );
}

#[test]
fn omits_variation_key_from_token_when_not_set() {
    let client = FakeLdClient::new();
    let tracker = tracker_without_variation(&client, "run-abc");

    assert_eq!(
        decoded_token_string(&tracker),
        r#"{"runId":"run-abc","graphKey":"my-agent-graph","version":2}"#
    );
}

#[test]
fn from_resumption_token_reconstructs_the_tracker_with_original_run_id() {
    let client = FakeLdClient::new();
    let token = make_tracker(&client, "orig-run-id").resumption_token();

    let reconstructed =
        LdGraphTracker::from_resumption_token(&token, client.dyn_client(), test_context())
            .expect("valid token");

    assert_eq!(
        track_data_json(&reconstructed),
        json!({
            "runId": "orig-run-id",
            "graphKey": GRAPH_KEY,
            "version": VERSION,
            "variationKey": VARIATION_KEY,
        })
    );
}

#[test]
fn returns_a_summary_with_only_the_resumption_token_initially() {
    let client = FakeLdClient::new();
    let tracker = make_tracker(&client, "r");

    assert_eq!(
        serde_json::to_value(tracker.get_summary()).unwrap(),
        json!({ "resumptionToken": tracker.resumption_token() })
    );
}

#[test]
fn returns_a_copy_of_the_summary_not_a_reference() {
    let client = FakeLdClient::new();
    let tracker = make_tracker(&client, "r");
    tracker.track_invocation_success();

    let mut summary1 = tracker.get_summary();
    let summary2 = tracker.get_summary();
    assert_eq!(summary1, summary2);

    summary1.success = Some(false);
    assert_eq!(tracker.get_summary(), summary2);
}

#[test]
fn track_invocation_success_sets_success_true_and_emits_event() {
    let client = FakeLdClient::new();
    let tracker = make_tracker(&client, "r");
    tracker.track_invocation_success();

    assert_eq!(tracker.get_summary().success, Some(true));
    assert_tracked(
        &client,
        "$ld:ai:graph:invocation_success",
        track_data_json(&tracker),
        1.0,
    );
}

#[test]
fn track_invocation_failure_sets_success_false_and_emits_event() {
    let client = FakeLdClient::new();
    let tracker = make_tracker(&client, "r");
    tracker.track_invocation_failure();

    assert_eq!(tracker.get_summary().success, Some(false));
    assert_tracked(
        &client,
        "$ld:ai:graph:invocation_failure",
        track_data_json(&tracker),
        1.0,
    );
}

fn assert_warned_invocation_already_recorded(client: &FakeLdClient) {
    assert!(
        client
            .logger
            .warnings()
            .iter()
            .any(|w| w.contains("invocation result already recorded on this graph tracker")),
        "warnings: {:?}",
        client.logger.warnings()
    );
}

#[test]
fn drops_second_track_invocation_success_call_and_warns() {
    let client = FakeLdClient::new();
    let tracker = make_tracker(&client, "r");
    tracker.track_invocation_success();
    tracker.track_invocation_success();

    assert_eq!(client.tracks().len(), 1);
    assert_warned_invocation_already_recorded(&client);
}

#[test]
fn drops_track_invocation_failure_after_track_invocation_success_and_warns() {
    let client = FakeLdClient::new();
    let tracker = make_tracker(&client, "r");
    tracker.track_invocation_success();
    tracker.track_invocation_failure();

    assert_eq!(client.tracks().len(), 1);
    assert_warned_invocation_already_recorded(&client);
}

#[test]
fn track_duration_sets_duration_ms_and_emits_event() {
    let client = FakeLdClient::new();
    let tracker = make_tracker(&client, "r");
    tracker.track_duration(1234.0);

    assert_eq!(tracker.get_summary().duration_ms, Some(1234.0));
    assert_tracked(
        &client,
        "$ld:ai:graph:duration:total",
        track_data_json(&tracker),
        1234.0,
    );
}

#[test]
fn drops_second_track_duration_call_and_warns() {
    let client = FakeLdClient::new();
    let tracker = make_tracker(&client, "r");
    tracker.track_duration(100.0);
    tracker.track_duration(200.0);

    assert_eq!(client.tracks().len(), 1);
    assert_eq!(tracker.get_summary().duration_ms, Some(100.0));
    assert!(!client.logger.warnings().is_empty());
}

#[test]
fn track_total_tokens_sets_tokens_and_emits_event_with_total_as_metric_value() {
    let client = FakeLdClient::new();
    let tracker = make_tracker(&client, "r");
    let tokens = LdTokenUsage {
        total: 500,
        input: 200,
        output: 300,
    };
    tracker.track_total_tokens(tokens);

    assert_eq!(tracker.get_summary().tokens, Some(tokens));
    assert_tracked(
        &client,
        "$ld:ai:graph:total_tokens",
        track_data_json(&tracker),
        500.0,
    );
}

#[test]
fn drops_second_track_total_tokens_call_and_warns() {
    let client = FakeLdClient::new();
    let tracker = make_tracker(&client, "r");
    tracker.track_total_tokens(LdTokenUsage {
        total: 100,
        input: 50,
        output: 50,
    });
    tracker.track_total_tokens(LdTokenUsage {
        total: 200,
        input: 100,
        output: 100,
    });

    assert_eq!(client.tracks().len(), 1);
    assert_eq!(tracker.get_summary().tokens.map(|t| t.total), Some(100));
    assert!(!client.logger.warnings().is_empty());
}

#[test]
fn track_path_sets_path_and_emits_event_with_path_in_data_payload() {
    let client = FakeLdClient::new();
    let tracker = make_tracker(&client, "r");
    let path = ["root-agent", "research-agent", "write-agent"];
    tracker.track_path(&path);

    assert_eq!(
        tracker.get_summary().path,
        Some(path.iter().map(|s| s.to_string()).collect())
    );
    assert_tracked(
        &client,
        "$ld:ai:graph:path",
        track_data_with(&tracker, json!({ "path": path })),
        1.0,
    );
}

#[test]
fn drops_second_track_path_call_and_warns() {
    let client = FakeLdClient::new();
    let tracker = make_tracker(&client, "r");
    tracker.track_path(&["a", "b"]);
    tracker.track_path(&["c", "d"]);

    assert_eq!(client.tracks().len(), 1);
    assert_eq!(
        tracker.get_summary().path,
        Some(vec!["a".to_string(), "b".to_string()])
    );
    assert!(!client.logger.warnings().is_empty());
}

#[test]
fn track_redirect_emits_event_with_source_key_and_redirected_target() {
    let client = FakeLdClient::new();
    let tracker = make_tracker(&client, "r");
    tracker.track_redirect("source-agent", "redirected-agent");

    assert_tracked(
        &client,
        "$ld:ai:graph:redirect",
        track_data_with(
            &tracker,
            json!({ "sourceKey": "source-agent", "redirectedTarget": "redirected-agent" }),
        ),
        1.0,
    );
}

#[test]
fn track_handoff_success_emits_event_with_source_key_and_target_key() {
    let client = FakeLdClient::new();
    let tracker = make_tracker(&client, "r");
    tracker.track_handoff_success("agent-a", "agent-b");

    assert_tracked(
        &client,
        "$ld:ai:graph:handoff_success",
        track_data_with(
            &tracker,
            json!({ "sourceKey": "agent-a", "targetKey": "agent-b" }),
        ),
        1.0,
    );
}

#[test]
fn track_handoff_failure_emits_event_with_source_key_and_target_key() {
    let client = FakeLdClient::new();
    let tracker = make_tracker(&client, "r");
    tracker.track_handoff_failure("agent-a", "agent-b");

    assert_tracked(
        &client,
        "$ld:ai:graph:handoff_failure",
        track_data_with(
            &tracker,
            json!({ "sourceKey": "agent-a", "targetKey": "agent-b" }),
        ),
        1.0,
    );
}

#[test]
fn edge_level_methods_can_fire_multiple_times_without_warning() {
    let client = FakeLdClient::new();
    let tracker = make_tracker(&client, "r");
    tracker.track_handoff_success("a", "b");
    tracker.track_handoff_success("a", "b");
    tracker.track_redirect("a", "c");
    tracker.track_handoff_failure("x", "y");

    assert_eq!(client.tracks().len(), 4);
    assert!(client.logger.warnings().is_empty());
}
