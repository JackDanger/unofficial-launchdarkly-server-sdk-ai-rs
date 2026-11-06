mod common;

use std::sync::Arc;

use base64::Engine;
use common::{test_context, FakeLdClient};
use indexmap::IndexMap;
use serde_json::{json, Value};
use unofficial_launchdarkly_server_sdk_ai::{AgentGraphDefinition, LdAiClient};

fn make_client() -> (Arc<FakeLdClient>, LdAiClient) {
    let fake = FakeLdClient::new();
    let client = LdAiClient::new(fake.dyn_client());
    (fake, client)
}

fn make_graph_flag_value(root: &str, edges: Value) -> Value {
    json!({ "_ldMeta": { "variationKey": "v1", "version": 1 }, "root": root, "edges": edges })
}

fn make_agent_flag_value(key: &str, enabled: bool) -> Value {
    json!({
        "_ldMeta": { "variationKey": format!("{key}-v1"), "enabled": enabled, "version": 1, "mode": "agent" },
        "instructions": format!("Instructions for {key}"),
    })
}

async fn agent_graph(client: &LdAiClient) -> AgentGraphDefinition {
    client.agent_graph("my-graph", &test_context(), None).await
}

fn debug_logged(fake: &FakeLdClient, fragment: &str) -> bool {
    fake.logger.debugs().iter().any(|m| m.contains(fragment))
}

fn sorted_keys<'a>(
    nodes: impl IntoIterator<Item = &'a unofficial_launchdarkly_server_sdk_ai::AgentGraphNode>,
) -> Vec<String> {
    let mut keys: Vec<String> = nodes.into_iter().map(|n| n.get_key().to_string()).collect();
    keys.sort();
    keys
}

#[tokio::test]
async fn returns_a_disabled_graph_when_ld_meta_enabled_is_false() {
    let (fake, client) = make_client();
    fake.set_flag(
        "my-graph",
        json!({ "_ldMeta": { "enabled": false }, "root": "root" }),
    );
    let graph = agent_graph(&client).await;
    assert!(!graph.enabled());
}

#[tokio::test]
async fn logs_debug_when_graph_is_disabled_via_ld_meta_enabled() {
    let (fake, client) = make_client();
    fake.set_flag(
        "my-graph",
        json!({ "_ldMeta": { "enabled": false }, "root": "root" }),
    );
    agent_graph(&client).await;
    assert!(debug_logged(&fake, "disabled"));
}

#[tokio::test]
async fn returns_a_disabled_graph_when_graph_flag_has_no_root() {
    let (fake, client) = make_client();
    fake.set_flag("my-graph", json!({ "root": "" }));
    let graph = agent_graph(&client).await;
    assert!(!graph.enabled());
}

#[tokio::test]
async fn logs_debug_when_graph_has_no_root() {
    let (fake, client) = make_client();
    fake.set_flag("my-graph", json!({ "root": "" }));
    agent_graph(&client).await;
    assert!(debug_logged(&fake, "not fetchable"));
}

#[tokio::test]
async fn returns_a_disabled_graph_when_a_node_is_unconnected_not_reachable_from_root() {
    let (fake, client) = make_client();
    fake.set_flag(
        "my-graph",
        make_graph_flag_value(
            "root",
            json!({ "root": [{ "key": "child" }], "orphan": [{ "key": "other" }] }),
        ),
    );
    let graph = agent_graph(&client).await;
    assert!(!graph.enabled());
    assert!(debug_logged(&fake, "unconnected node"));
}

#[tokio::test]
async fn returns_an_enabled_graph_and_traverses_a_cyclic_graph_each_node_visited_once() {
    let (fake, client) = make_client();
    fake.set_flag(
        "my-graph",
        make_graph_flag_value("a", json!({ "a": [{ "key": "b" }], "b": [{ "key": "a" }] })),
    );
    fake.set_flag("a", make_agent_flag_value("agent", true));
    fake.set_flag("b", make_agent_flag_value("agent", true));

    let graph = agent_graph(&client).await;
    assert!(graph.enabled());

    let mut visited: Vec<String> = Vec::new();
    graph.traverse(
        |node, _| {
            visited.push(node.get_key().to_string());
        },
        IndexMap::<String, ()>::new(),
    );
    visited.sort();
    assert_eq!(visited, ["a", "b"]);
}

#[tokio::test]
async fn returns_a_disabled_graph_when_a_child_agent_config_is_disabled() {
    let (fake, client) = make_client();
    fake.set_flag(
        "my-graph",
        make_graph_flag_value("root", json!({ "root": [{ "key": "child" }] })),
    );
    fake.set_flag("root", make_agent_flag_value("root", true));
    fake.set_flag("child", make_agent_flag_value("child", false));
    let graph = agent_graph(&client).await;
    assert!(!graph.enabled());
    assert!(debug_logged(&fake, "not enabled"));
}

#[tokio::test]
async fn returns_an_enabled_graph_for_a_valid_graph_with_a_single_node() {
    let (fake, client) = make_client();
    fake.set_flag("my-graph", make_graph_flag_value("solo-agent", json!({})));
    fake.set_flag("solo-agent", make_agent_flag_value("solo-agent", true));
    let graph = agent_graph(&client).await;
    assert!(graph.enabled());
    assert_eq!(
        graph.root_node().expect("a root node").get_key(),
        "solo-agent"
    );
}

#[tokio::test]
async fn returns_an_enabled_graph_with_correct_nodes_for_multi_node_graph() {
    let (fake, client) = make_client();
    fake.set_flag(
        "my-graph",
        make_graph_flag_value(
            "root",
            json!({
                "root": [{ "key": "child-a" }, { "key": "child-b" }],
                "child-a": [{ "key": "leaf" }],
            }),
        ),
    );
    for key in ["root", "child-a", "child-b", "leaf"] {
        fake.set_flag(key, make_agent_flag_value("agent", true));
    }

    let graph = agent_graph(&client).await;
    assert!(graph.enabled());
    assert_eq!(graph.root_node().expect("a root node").get_key(), "root");
    assert_eq!(
        sorted_keys(graph.get_child_nodes("root")),
        ["child-a", "child-b"]
    );
    assert_eq!(sorted_keys(graph.terminal_nodes()), ["child-b", "leaf"]);
}

#[tokio::test]
async fn tracks_usage_event_when_agent_graph_is_called() {
    let (fake, client) = make_client();
    fake.set_flag("my-graph", json!({ "root": "" }));
    agent_graph(&client).await;
    let tracks = fake.tracks_for("$ld:ai:usage:agent-graph");
    assert_eq!(tracks.len(), 1);
    assert_eq!(tracks[0].context, test_context());
    assert_eq!(tracks[0].data, Some(json!("my-graph")));
    assert_eq!(tracks[0].metric_value, Some(1.0));
}

#[tokio::test]
async fn create_graph_tracker_reconstructs_a_tracker_from_a_resumption_token() {
    let (_fake, client) = make_client();
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(r#"{"runId":"run-1","graphKey":"g-key","variationKey":"v99","version":7}"#);

    let tracker = client
        .create_graph_tracker(&token, &test_context())
        .unwrap();

    let data = tracker.get_track_data();
    assert_eq!(data.graph_key, "g-key");
    assert_eq!(data.version, 7);
    assert_eq!(data.variation_key.as_deref(), Some("v99"));
    assert_eq!(data.run_id, "run-1");
}
