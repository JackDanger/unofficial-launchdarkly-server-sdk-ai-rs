mod common;

use std::collections::HashMap;

use indexmap::IndexMap;
use serde_json::{json, Map, Value};
use unofficial_launchdarkly_server_sdk_ai::{
    AgentGraphDefinition, AgentGraphNode, Evaluator, ExecutionContext, LdAgentGraphFlagValue,
    LdAiAgentConfig, LdAiConfigTracker, LdAiConfigTrackerParams, LdGraphEdge, LdGraphTracker,
    TrackerFactory,
};

use common::{test_context, FakeLdClient};

fn make_agent_config(key: &str) -> LdAiAgentConfig {
    let client = FakeLdClient::new();
    let tracker_key = key.to_string();
    LdAiAgentConfig {
        key: key.to_string(),
        enabled: true,
        model: None,
        provider: None,
        instructions: Some(format!("You are {key}.")),
        judge_configuration: None,
        tools: None,
        tracker_factory: TrackerFactory::new(move || {
            LdAiConfigTracker::new(
                client.dyn_client(),
                test_context(),
                LdAiConfigTrackerParams::new(
                    uuid::Uuid::new_v4().to_string(),
                    tracker_key.clone(),
                    "",
                    1,
                ),
            )
        }),
        evaluator: Evaluator::noop(),
    }
}

fn configs(keys: &[&str]) -> HashMap<String, LdAiAgentConfig> {
    keys.iter()
        .map(|key| (key.to_string(), make_agent_config(key)))
        .collect()
}

fn make_graph_with_meta(
    root: &str,
    edges: Value,
    variation_key: Option<&str>,
    version: u64,
) -> LdAgentGraphFlagValue {
    let mut meta = Map::new();
    if let Some(variation_key) = variation_key {
        meta.insert("variationKey".to_string(), json!(variation_key));
    }
    meta.insert("version".to_string(), json!(version));
    LdAgentGraphFlagValue::from_value(&json!({
        "_ldMeta": meta,
        "root": root,
        "edges": edges,
    }))
}

fn make_graph(root: &str, edges: Value) -> LdAgentGraphFlagValue {
    make_graph_with_meta(root, edges, None, 1)
}

fn make_definition_with_enabled(
    graph: &LdAgentGraphFlagValue,
    agent_configs: &HashMap<String, LdAiAgentConfig>,
    enabled: bool,
) -> AgentGraphDefinition {
    let nodes = AgentGraphDefinition::build_nodes(graph, agent_configs);
    let client = FakeLdClient::new();
    let graph_key = graph.root.clone();
    let meta = graph.ld_meta.clone().unwrap_or_default();
    AgentGraphDefinition::new(graph.clone(), nodes, enabled, move || {
        LdGraphTracker::new(
            client.dyn_client(),
            uuid::Uuid::new_v4().to_string(),
            graph_key.clone(),
            meta.variation_key.clone(),
            meta.version.unwrap_or(1),
            test_context(),
        )
    })
}

fn make_definition(
    graph: &LdAgentGraphFlagValue,
    agent_configs: &HashMap<String, LdAiAgentConfig>,
) -> AgentGraphDefinition {
    make_definition_with_enabled(graph, agent_configs, true)
}

fn keys_of(nodes: &[&AgentGraphNode]) -> Vec<String> {
    nodes.iter().map(|n| n.get_key().to_string()).collect()
}

fn sorted_keys_of(nodes: &[&AgentGraphNode]) -> Vec<String> {
    let mut keys = keys_of(nodes);
    keys.sort();
    keys
}

fn sorted_ctx_keys<T>(ctx: &ExecutionContext<T>) -> Vec<String> {
    let mut keys: Vec<String> = ctx.keys().cloned().collect();
    keys.sort();
    keys
}

fn sorted(keys: &[&str]) -> Vec<String> {
    let mut keys: Vec<String> = keys.iter().map(|k| k.to_string()).collect();
    keys.sort();
    keys
}

fn no_context() -> ExecutionContext<Value> {
    IndexMap::new()
}

fn position(order: &[String], key: &str) -> usize {
    order
        .iter()
        .position(|k| k == key)
        .unwrap_or_else(|| panic!("{key} was not visited in {order:?}"))
}

#[derive(Clone, Copy)]
enum Direction {
    Forward,
    Reverse,
}

fn collect_order(def: &AgentGraphDefinition, direction: Direction) -> Vec<String> {
    let mut order = Vec::new();
    let visit = |node: &AgentGraphNode, _ctx: ExecutionContext<Value>| {
        order.push(node.get_key().to_string());
        Value::Null
    };
    match direction {
        Direction::Forward => def.traverse(visit, no_context()),
        Direction::Reverse => def.reverse_traverse(visit, no_context()),
    }
    order
}

fn g2_edges(a_edges: &[&str]) -> Value {
    let a: Vec<Value> = a_edges.iter().map(|k| json!({ "key": k })).collect();
    json!({
        "a": a,
        "b": [{ "key": "e" }],
        "c": [{ "key": "d" }],
        "d": [{ "key": "e" }],
    })
}

fn g2_definition(a_edges: &[&str]) -> AgentGraphDefinition {
    make_definition(
        &make_graph("a", g2_edges(a_edges)),
        &configs(&["a", "b", "c", "d", "e"]),
    )
}

fn g2_forward_deps(key: &str) -> Vec<String> {
    sorted(match key {
        "a" => &[],
        "b" | "c" => &["a"],
        "d" => &["a", "c"],
        "e" => &["a", "b", "c", "d"],
        other => panic!("unexpected node {other}"),
    })
}

fn g2_reverse_deps(key: &str) -> Vec<String> {
    sorted(match key {
        "a" => &["b", "c", "d", "e"],
        "b" | "d" => &["e"],
        "c" => &["d", "e"],
        "e" => &[],
        other => panic!("unexpected node {other}"),
    })
}

#[test]
fn build_nodes_creates_a_node_for_every_unique_key_in_the_graph() {
    let graph = make_graph(
        "root",
        json!({
            "root": [{ "key": "child-a" }, { "key": "child-b" }],
            "child-a": [{ "key": "leaf" }],
        }),
    );
    let nodes = AgentGraphDefinition::build_nodes(
        &graph,
        &configs(&["root", "child-a", "child-b", "leaf"]),
    );
    let mut keys: Vec<&String> = nodes.keys().collect();
    keys.sort();
    assert_eq!(keys, ["child-a", "child-b", "leaf", "root"]);
}

#[test]
fn build_nodes_skips_keys_whose_agent_config_is_missing() {
    let graph = make_graph("root", json!({ "root": [{ "key": "orphan" }] }));
    let nodes = AgentGraphDefinition::build_nodes(&graph, &configs(&["root"]));
    assert!(nodes.get("root").is_some());
    assert!(nodes.get("orphan").is_none());
}

#[test]
fn build_nodes_assigns_correct_edges_to_each_node() {
    let graph = make_graph(
        "root",
        json!({ "root": [{ "key": "child", "handoff": { "someOption": true } }] }),
    );
    let nodes = AgentGraphDefinition::build_nodes(&graph, &configs(&["root", "child"]));
    assert_eq!(
        nodes["root"].get_edges(),
        [LdGraphEdge {
            key: "child".to_string(),
            handoff: json!({ "someOption": true }).as_object().cloned(),
        }]
    );
    assert!(nodes["child"].get_edges().is_empty());
}

#[test]
fn collect_all_keys_includes_root_edge_sources_and_edge_targets() {
    let graph = make_graph(
        "root",
        json!({
            "root": [{ "key": "a" }, { "key": "b" }],
            "a": [{ "key": "c" }],
        }),
    );
    let mut keys: Vec<String> = AgentGraphDefinition::collect_all_keys(&graph)
        .into_iter()
        .collect();
    keys.sort();
    assert_eq!(keys, ["a", "b", "c", "root"]);
}

#[test]
fn collect_all_keys_works_for_a_graph_with_no_edges() {
    let graph = make_graph("solo", json!({}));
    let keys: Vec<String> = AgentGraphDefinition::collect_all_keys(&graph)
        .into_iter()
        .collect();
    assert_eq!(keys, ["solo"]);
}

#[test]
fn enabled_reflects_the_value_passed_at_construction() {
    let graph = make_graph("r", json!({}));
    let enabled = make_definition_with_enabled(&graph, &configs(&["r"]), true);
    assert!(enabled.enabled());

    let disabled = make_definition_with_enabled(&graph, &configs(&["r"]), false);
    assert!(!disabled.enabled());
}

#[test]
fn root_node_returns_the_root_node() {
    let graph = make_graph("root", json!({ "root": [{ "key": "leaf" }] }));
    let def = make_definition(&graph, &configs(&["root", "leaf"]));
    assert_eq!(def.root_node().unwrap().get_key(), "root");
}

#[test]
fn get_node_returns_null_for_unknown_key() {
    let graph = make_graph("root", json!({}));
    let def = make_definition(&graph, &configs(&["root"]));
    assert!(def.get_node("nonexistent").is_none());
}

#[test]
fn terminal_nodes_returns_nodes_with_no_outgoing_edges() {
    let graph = make_graph(
        "root",
        json!({
            "root": [{ "key": "mid" }],
            "mid": [{ "key": "leaf-a" }, { "key": "leaf-b" }],
        }),
    );
    let def = make_definition(&graph, &configs(&["root", "mid", "leaf-a", "leaf-b"]));
    assert_eq!(sorted_keys_of(&def.terminal_nodes()), ["leaf-a", "leaf-b"]);
}

#[test]
fn get_child_nodes_returns_direct_children() {
    let graph = make_graph("root", json!({ "root": [{ "key": "a" }, { "key": "b" }] }));
    let def = make_definition(&graph, &configs(&["root", "a", "b"]));
    assert_eq!(sorted_keys_of(&def.get_child_nodes("root")), ["a", "b"]);
}

#[test]
fn get_child_nodes_returns_empty_array_for_terminal_node() {
    let graph = make_graph("root", json!({}));
    let def = make_definition(&graph, &configs(&["root"]));
    assert!(def.get_child_nodes("root").is_empty());
}

#[test]
fn get_child_nodes_returns_empty_array_for_unknown_key() {
    let graph = make_graph("root", json!({}));
    let def = make_definition(&graph, &configs(&["root"]));
    assert!(def.get_child_nodes("unknown").is_empty());
}

#[test]
fn get_parent_nodes_returns_nodes_that_have_direct_edges_to_the_given_key() {
    let graph = make_graph(
        "root",
        json!({
            "root": [{ "key": "child" }],
            "sibling": [{ "key": "child" }],
        }),
    );
    let def = make_definition(&graph, &configs(&["root", "sibling", "child"]));
    assert_eq!(
        sorted_keys_of(&def.get_parent_nodes("child")),
        ["root", "sibling"]
    );
}

#[test]
fn get_parent_nodes_returns_empty_array_for_root_node() {
    let graph = make_graph("root", json!({ "root": [{ "key": "child" }] }));
    let def = make_definition(&graph, &configs(&["root", "child"]));
    assert!(def.get_parent_nodes("root").is_empty());
}

#[test]
fn traverse_visits_every_node_with_predecessors_before_dependents_root_first() {
    let graph = make_graph(
        "root",
        json!({
            "root": [{ "key": "a" }, { "key": "b" }],
            "a": [{ "key": "c" }],
        }),
    );
    let def = make_definition(&graph, &configs(&["root", "a", "b", "c"]));

    let order = collect_order(&def, Direction::Forward);

    assert_eq!(order[0], "root");
    let c_idx = position(&order, "c");
    assert!(position(&order, "a") < c_idx);
    assert!(position(&order, "b") < c_idx);
    assert_eq!(order.len(), 4);
}

#[test]
fn traverse_stores_fn_return_values_in_execution_context() {
    let graph = make_graph("root", json!({ "root": [{ "key": "child" }] }));
    let def = make_definition(&graph, &configs(&["root", "child"]));

    let mut captures: Vec<ExecutionContext<Value>> = Vec::new();
    def.traverse(
        |node, ctx| {
            captures.push(ctx);
            json!(format!("result-of-{}", node.get_key()))
        },
        no_context(),
    );

    assert_eq!(captures[1].get("root"), Some(&json!("result-of-root")));
}

#[test]
fn traverse_accepts_and_uses_initial_execution_context() {
    let graph = make_graph("root", json!({}));
    let def = make_definition(&graph, &configs(&["root"]));

    let mut captured: Vec<ExecutionContext<Value>> = Vec::new();
    def.traverse(
        |_node, ctx| {
            captured.push(ctx);
            Value::Null
        },
        IndexMap::from([("initialKey".to_string(), json!("initialValue"))]),
    );

    assert_eq!(captured[0].get("initialKey"), Some(&json!("initialValue")));
}

#[test]
fn traverse_handles_a_single_node_graph() {
    let graph = make_graph("solo", json!({}));
    let def = make_definition(&graph, &configs(&["solo"]));
    assert_eq!(collect_order(&def, Direction::Forward), ["solo"]);
}

#[test]
fn reverse_traverse_processes_terminal_nodes_before_their_parents_root_last() {
    let graph = make_graph(
        "root",
        json!({
            "root": [{ "key": "a" }, { "key": "b" }],
            "a": [{ "key": "c" }],
        }),
    );
    let def = make_definition(&graph, &configs(&["root", "a", "b", "c"]));

    let order = collect_order(&def, Direction::Reverse);

    assert_eq!(order.last().unwrap(), "root");
    assert!(position(&order, "c") < position(&order, "a"));
    let mut all = order.clone();
    all.sort();
    assert_eq!(all, ["a", "b", "c", "root"]);
}

#[test]
fn reverse_traverse_stores_fn_return_values_in_execution_context() {
    let graph = make_graph("root", json!({ "root": [{ "key": "child" }] }));
    let def = make_definition(&graph, &configs(&["root", "child"]));

    let mut context_when_root_runs: Vec<ExecutionContext<Value>> = Vec::new();
    def.reverse_traverse(
        |node, ctx| {
            if node.get_key() == "root" {
                context_when_root_runs.push(ctx);
            }
            json!(format!("result-of-{}", node.get_key()))
        },
        no_context(),
    );

    assert_eq!(
        context_when_root_runs[0].get("child"),
        Some(&json!("result-of-child"))
    );
}

#[test]
fn reverse_traverse_visits_a_node_with_multiple_parents_only_once() {
    let graph = make_graph(
        "root",
        json!({
            "root": [{ "key": "a" }, { "key": "b" }],
            "a": [{ "key": "d" }],
            "b": [{ "key": "c" }],
            "d": [{ "key": "c" }],
        }),
    );
    let def = make_definition(&graph, &configs(&["root", "a", "b", "c", "d"]));

    let order = collect_order(&def, Direction::Reverse);

    assert_eq!(order[0], "c");
    assert_eq!(order.last().unwrap(), "root");
    let mut all = order.clone();
    all.sort();
    assert_eq!(all, ["a", "b", "c", "d", "root"]);
}

#[test]
fn reverse_traverse_accepts_and_uses_initial_execution_context() {
    let graph = make_graph("root", json!({ "root": [{ "key": "child" }] }));
    let def = make_definition(&graph, &configs(&["root", "child"]));

    let mut captured: Vec<ExecutionContext<Value>> = Vec::new();
    def.reverse_traverse(
        |node, ctx| {
            captured.push(ctx);
            json!(format!("result-of-{}", node.get_key()))
        },
        IndexMap::from([("initialKey".to_string(), json!("initialValue"))]),
    );

    assert_eq!(captured[0].get("initialKey"), Some(&json!("initialValue")));
}

#[test]
fn reverse_traverse_visits_each_node_once_on_a_cyclic_graph_with_root_last() {
    let graph = make_graph(
        "a",
        json!({
            "a": [{ "key": "b" }],
            "b": [{ "key": "a" }],
        }),
    );
    let def = make_definition(&graph, &configs(&["a", "b"]));

    let visited = collect_order(&def, Direction::Reverse);

    assert_eq!(visited.len(), 2);
    assert_eq!(visited.last().unwrap(), "a");
    let mut all = visited.clone();
    all.sort();
    assert_eq!(all, ["a", "b"]);
}

mod traversal_vectors {
    use super::*;

    struct TraversalVector {
        root: &'static str,
        nodes: &'static [&'static str],
        edges: &'static [(&'static str, &'static str)],
        traverse: &'static [&'static str],
        reverse_traverse: &'static [&'static str],
        traverse_context: &'static [(&'static str, &'static [&'static str])],
        reverse_traverse_context: &'static [(&'static str, &'static [&'static str])],
    }

    fn edges_from_pairs(pairs: &[(&str, &str)]) -> Value {
        let mut edges: Map<String, Value> = Map::new();
        for (src, tgt) in pairs {
            edges
                .entry(src.to_string())
                .or_insert_with(|| json!([]))
                .as_array_mut()
                .unwrap()
                .push(json!({ "key": tgt }));
        }
        Value::Object(edges)
    }

    fn make_definition_from_vector(v: &TraversalVector) -> AgentGraphDefinition {
        make_definition(
            &make_graph(v.root, edges_from_pairs(v.edges)),
            &configs(v.nodes),
        )
    }

    fn assert_order_and_scoped_context(v: &TraversalVector) {
        let def = make_definition_from_vector(v);

        let mut fwd_order: Vec<String> = Vec::new();
        let mut fwd_ctx: HashMap<String, Vec<String>> = HashMap::new();
        def.traverse(
            |node, ctx: ExecutionContext<Value>| {
                let key = node.get_key().to_string();
                fwd_order.push(key.clone());
                fwd_ctx.insert(key.clone(), sorted_ctx_keys(&ctx));
                json!(format!("{key}-result"))
            },
            no_context(),
        );
        assert_eq!(fwd_order, v.traverse);
        for (key, expected) in v.traverse_context {
            assert_eq!(fwd_ctx[*key], sorted(expected), "traverse context of {key}");
        }

        let mut rev_order: Vec<String> = Vec::new();
        let mut rev_ctx: HashMap<String, Vec<String>> = HashMap::new();
        def.reverse_traverse(
            |node, ctx: ExecutionContext<Value>| {
                let key = node.get_key().to_string();
                rev_order.push(key.clone());
                rev_ctx.insert(key.clone(), sorted_ctx_keys(&ctx));
                json!(format!("{key}-result"))
            },
            no_context(),
        );
        assert_eq!(rev_order, v.reverse_traverse);
        for (key, expected) in v.reverse_traverse_context {
            assert_eq!(
                rev_ctx[*key],
                sorted(expected),
                "reverseTraverse context of {key}"
            );
        }
    }

    #[test]
    fn g1_asserts_traverse_reverse_order_and_exact_scoped_context() {
        assert_order_and_scoped_context(&TraversalVector {
            root: "a",
            nodes: &["a", "b", "c"],
            edges: &[("a", "b"), ("b", "c")],
            traverse: &["a", "b", "c"],
            reverse_traverse: &["c", "b", "a"],
            traverse_context: &[("a", &[]), ("b", &["a"]), ("c", &["a", "b"])],
            reverse_traverse_context: &[("a", &["b", "c"]), ("b", &["c"]), ("c", &[])],
        });
    }

    #[test]
    fn g2_asserts_traverse_reverse_order_and_exact_scoped_context() {
        assert_order_and_scoped_context(&TraversalVector {
            root: "a",
            nodes: &["a", "b", "c", "d", "e"],
            edges: &[("a", "b"), ("a", "c"), ("c", "d"), ("d", "e"), ("b", "e")],
            traverse: &["a", "b", "c", "d", "e"],
            reverse_traverse: &["e", "b", "d", "c", "a"],
            traverse_context: &[
                ("a", &[]),
                ("b", &["a"]),
                ("c", &["a"]),
                ("d", &["a", "c"]),
                ("e", &["a", "b", "c", "d"]),
            ],
            reverse_traverse_context: &[
                ("a", &["b", "c", "d", "e"]),
                ("b", &["e"]),
                ("c", &["d", "e"]),
                ("d", &["e"]),
                ("e", &[]),
            ],
        });
    }

    #[test]
    fn g2b_asserts_traverse_reverse_order_and_exact_scoped_context() {
        assert_order_and_scoped_context(&TraversalVector {
            root: "a",
            nodes: &["a", "b", "c", "d", "e"],
            edges: &[("a", "c"), ("a", "b"), ("c", "d"), ("d", "e"), ("b", "e")],
            traverse: &["a", "c", "b", "d", "e"],
            reverse_traverse: &["e", "b", "d", "c", "a"],
            traverse_context: &[
                ("a", &[]),
                ("b", &["a"]),
                ("c", &["a"]),
                ("d", &["a", "c"]),
                ("e", &["a", "b", "c", "d"]),
            ],
            reverse_traverse_context: &[
                ("a", &["b", "c", "d", "e"]),
                ("b", &["e"]),
                ("c", &["d", "e"]),
                ("d", &["e"]),
                ("e", &[]),
            ],
        });
    }

    #[test]
    fn g3_asserts_traverse_reverse_order_and_exact_scoped_context() {
        assert_order_and_scoped_context(&TraversalVector {
            root: "a",
            nodes: &["a", "b", "c", "d"],
            edges: &[("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
            traverse: &["a", "b", "c", "d"],
            reverse_traverse: &["d", "b", "c", "a"],
            traverse_context: &[
                ("a", &[]),
                ("b", &["a"]),
                ("c", &["a"]),
                ("d", &["a", "b", "c"]),
            ],
            reverse_traverse_context: &[
                ("a", &["b", "c", "d"]),
                ("b", &["d"]),
                ("c", &["d"]),
                ("d", &[]),
            ],
        });
    }

    #[test]
    fn g4_asserts_traverse_reverse_order_and_exact_scoped_context() {
        assert_order_and_scoped_context(&TraversalVector {
            root: "a",
            nodes: &["a", "n", "m", "t"],
            edges: &[("a", "n"), ("n", "m"), ("n", "t"), ("m", "t")],
            traverse: &["a", "n", "m", "t"],
            reverse_traverse: &["t", "m", "n", "a"],
            traverse_context: &[
                ("a", &[]),
                ("n", &["a"]),
                ("m", &["a", "n"]),
                ("t", &["a", "m", "n"]),
            ],
            reverse_traverse_context: &[
                ("a", &["m", "n", "t"]),
                ("n", &["m", "t"]),
                ("m", &["t"]),
                ("t", &[]),
            ],
        });
    }

    #[test]
    fn g5_asserts_traverse_reverse_order_and_exact_scoped_context() {
        assert_order_and_scoped_context(&TraversalVector {
            root: "a",
            nodes: &["a", "b", "c", "d"],
            edges: &[("a", "b"), ("a", "c"), ("b", "d")],
            traverse: &["a", "b", "c", "d"],
            reverse_traverse: &["c", "d", "b", "a"],
            traverse_context: &[("a", &[]), ("b", &["a"]), ("c", &["a"]), ("d", &["a", "b"])],
            reverse_traverse_context: &[
                ("a", &["b", "c", "d"]),
                ("b", &["d"]),
                ("c", &[]),
                ("d", &[]),
            ],
        });
    }

    #[test]
    fn g6_asserts_traverse_reverse_order_and_exact_scoped_context() {
        assert_order_and_scoped_context(&TraversalVector {
            root: "a",
            nodes: &["a", "b", "c"],
            edges: &[("a", "b"), ("b", "c"), ("c", "b")],
            traverse: &["a", "b", "c"],
            reverse_traverse: &["b", "c", "a"],
            traverse_context: &[("a", &[]), ("b", &["a"]), ("c", &["a", "b"])],
            reverse_traverse_context: &[("a", &["b", "c"]), ("b", &[]), ("c", &["b"])],
        });
    }
}

mod given_g1_linear_graph_a_b_c {
    use super::*;

    fn def() -> AgentGraphDefinition {
        make_definition(
            &make_graph("a", json!({ "a": [{ "key": "b" }], "b": [{ "key": "c" }] })),
            &configs(&["a", "b", "c"]),
        )
    }

    #[test]
    fn traverse_visits_a_b_c() {
        assert_eq!(collect_order(&def(), Direction::Forward), ["a", "b", "c"]);
    }

    #[test]
    fn reverse_traverse_visits_c_b_a() {
        assert_eq!(collect_order(&def(), Direction::Reverse), ["c", "b", "a"]);
    }
}

mod given_g2_skewed_diamond {
    use super::*;

    #[test]
    fn traverse_visits_a_b_c_d_e() {
        assert_eq!(
            collect_order(&g2_definition(&["b", "c"]), Direction::Forward),
            ["a", "b", "c", "d", "e"]
        );
    }

    #[test]
    fn reverse_traverse_visits_e_b_d_c_a() {
        assert_eq!(
            collect_order(&g2_definition(&["b", "c"]), Direction::Reverse),
            ["e", "b", "d", "c", "a"]
        );
    }

    #[test]
    fn traverse_keeps_e_last_when_a_edges_are_declared_c_b() {
        let order = collect_order(&g2_definition(&["c", "b"]), Direction::Forward);
        assert_eq!(order, ["a", "c", "b", "d", "e"]);
        assert!(position(&order, "d") < position(&order, "e"));
    }

    #[test]
    fn reverse_traverse_keeps_e_before_d_when_a_edges_are_declared_c_b() {
        let order = collect_order(&g2_definition(&["c", "b"]), Direction::Reverse);
        assert_eq!(order[0], "e");
        assert!(position(&order, "e") < position(&order, "d"));
        assert!(position(&order, "e") < position(&order, "b"));
        assert_eq!(order.last().unwrap(), "a");
    }
}

mod given_g3_symmetric_diamond {
    use super::*;

    fn def() -> AgentGraphDefinition {
        make_definition(
            &make_graph(
                "a",
                json!({
                    "a": [{ "key": "b" }, { "key": "c" }],
                    "b": [{ "key": "d" }],
                    "c": [{ "key": "d" }],
                }),
            ),
            &configs(&["a", "b", "c", "d"]),
        )
    }

    #[test]
    fn traverse_visits_a_b_c_d() {
        assert_eq!(
            collect_order(&def(), Direction::Forward),
            ["a", "b", "c", "d"]
        );
    }

    #[test]
    fn reverse_traverse_visits_d_b_c_a() {
        assert_eq!(
            collect_order(&def(), Direction::Reverse),
            ["d", "b", "c", "a"]
        );
    }
}

mod given_g4_nested_parent {
    use super::*;

    fn def() -> AgentGraphDefinition {
        make_definition(
            &make_graph(
                "a",
                json!({
                    "a": [{ "key": "n" }],
                    "n": [{ "key": "m" }, { "key": "t" }],
                    "m": [{ "key": "t" }],
                }),
            ),
            &configs(&["a", "n", "m", "t"]),
        )
    }

    #[test]
    fn traverse_visits_a_n_m_t() {
        assert_eq!(
            collect_order(&def(), Direction::Forward),
            ["a", "n", "m", "t"]
        );
    }

    #[test]
    fn reverse_traverse_visits_t_m_n_a() {
        assert_eq!(
            collect_order(&def(), Direction::Reverse),
            ["t", "m", "n", "a"]
        );
    }
}

mod given_g5_multi_terminal {
    use super::*;

    fn def() -> AgentGraphDefinition {
        make_definition(
            &make_graph(
                "a",
                json!({
                    "a": [{ "key": "b" }, { "key": "c" }],
                    "b": [{ "key": "d" }],
                }),
            ),
            &configs(&["a", "b", "c", "d"]),
        )
    }

    #[test]
    fn traverse_visits_a_b_c_d() {
        assert_eq!(
            collect_order(&def(), Direction::Forward),
            ["a", "b", "c", "d"]
        );
    }

    #[test]
    fn reverse_traverse_visits_c_d_b_a() {
        assert_eq!(
            collect_order(&def(), Direction::Reverse),
            ["c", "d", "b", "a"]
        );
    }
}

mod given_g6_cycle {
    use super::*;

    fn def() -> AgentGraphDefinition {
        make_definition(
            &make_graph(
                "a",
                json!({
                    "a": [{ "key": "b" }],
                    "b": [{ "key": "c" }],
                    "c": [{ "key": "b" }],
                }),
            ),
            &configs(&["a", "b", "c"]),
        )
    }

    #[test]
    fn traverse_visits_each_node_once_with_a_first_and_a_deterministic_order() {
        let def = def();
        let first = collect_order(&def, Direction::Forward);
        let second = collect_order(&def, Direction::Forward);
        assert_eq!(first[0], "a");
        assert_eq!(first.len(), 3);
        let mut all = first.clone();
        all.sort();
        assert_eq!(all, ["a", "b", "c"]);
        assert_eq!(first, second);
    }

    #[test]
    fn reverse_traverse_visits_each_node_once_with_a_last_and_a_deterministic_order() {
        let def = def();
        let first = collect_order(&def, Direction::Reverse);
        let second = collect_order(&def, Direction::Reverse);
        assert_eq!(first.last().unwrap(), "a");
        assert_eq!(first.len(), 3);
        let mut all = first.clone();
        all.sort();
        assert_eq!(all, ["a", "b", "c"]);
        assert_eq!(first, second);
    }
}

#[test]
fn traverse_scopes_context_to_exact_predecessors_on_g2_skewed_diamond() {
    let mut visited = 0;
    g2_definition(&["b", "c"]).traverse(
        |node, ctx: ExecutionContext<Value>| {
            let key = node.get_key();
            visited += 1;
            assert_eq!(
                sorted_ctx_keys(&ctx),
                g2_forward_deps(key),
                "context of {key}"
            );
            if key == "b" {
                assert!(!ctx.contains_key("c"));
            }
            if key == "d" {
                assert!(!ctx.contains_key("b"));
            }
            json!(format!("result-of-{key}"))
        },
        no_context(),
    );
    assert_eq!(visited, 5);
}

#[test]
fn reverse_traverse_scopes_context_to_exact_descendants_on_g2_skewed_diamond() {
    let mut visited = 0;
    g2_definition(&["b", "c"]).reverse_traverse(
        |node, ctx: ExecutionContext<Value>| {
            let key = node.get_key();
            visited += 1;
            assert_eq!(
                sorted_ctx_keys(&ctx),
                g2_reverse_deps(key),
                "context of {key}"
            );
            if key == "b" || key == "d" {
                assert!(!ctx.contains_key("c"));
            }
            json!(format!("result-of-{key}"))
        },
        no_context(),
    );
    assert_eq!(visited, 5);
}

#[test]
fn traverse_context_scoping_is_independent_of_a_edge_declaration_order_on_g2() {
    let mut visited = 0;
    g2_definition(&["c", "b"]).traverse(
        |node, ctx: ExecutionContext<Value>| {
            let key = node.get_key();
            visited += 1;
            assert_eq!(
                sorted_ctx_keys(&ctx),
                g2_forward_deps(key),
                "context of {key}"
            );
            json!(format!("result-of-{key}"))
        },
        no_context(),
    );
    assert_eq!(visited, 5);
}

#[test]
fn traverse_includes_initial_context_keys_alongside_scoped_predecessors() {
    let mut visited = 0;
    g2_definition(&["b", "c"]).traverse(
        |node, ctx: ExecutionContext<Value>| {
            let key = node.get_key();
            visited += 1;
            let mut expected = g2_forward_deps(key);
            expected.push("seed".to_string());
            expected.sort();
            assert_eq!(sorted_ctx_keys(&ctx), expected, "context of {key}");
            assert_eq!(ctx.get("seed"), Some(&json!("value")));
            json!(format!("result-of-{key}"))
        },
        IndexMap::from([("seed".to_string(), json!("value"))]),
    );
    assert_eq!(visited, 5);
}

#[test]
fn traverse_visits_each_node_once_on_a_self_loop_without_hanging() {
    let graph = make_graph("a", json!({ "a": [{ "key": "a" }] }));
    let def = make_definition(&graph, &configs(&["a"]));
    assert_eq!(collect_order(&def, Direction::Forward), ["a"]);
}

#[test]
fn traverse_and_reverse_traverse_produce_identical_orders_across_repeated_runs() {
    let def = g2_definition(&["b", "c"]);
    assert_eq!(
        collect_order(&def, Direction::Forward),
        collect_order(&def, Direction::Forward)
    );
    assert_eq!(
        collect_order(&def, Direction::Reverse),
        collect_order(&def, Direction::Reverse)
    );
}

#[test]
fn get_config_returns_the_raw_flag_value() {
    let graph = make_graph_with_meta("root", json!({}), Some("var-key"), 5);
    let def = make_definition(&graph, &configs(&["root"]));
    assert_eq!(def.get_config(), &graph);
}
