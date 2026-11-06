mod common;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use serde_json::json;
use unofficial_launchdarkly_server_sdk_ai::{
    AgentGraphDefinition, AgentGraphRunnerResult, BoxError, Evaluator, LdAgentGraphFlagValue,
    LdAiAgentConfig, LdAiConfigTracker, LdAiConfigTrackerParams, LdAiGraphMetrics,
    LdAiMetricSummary, LdAiMetrics, LdGraphTracker, LdLogger, LdTokenUsage, ManagedAgentGraph,
    TrackerFactory,
};

use common::{test_context, FakeLdClient, TrackCall};

const GRAPH_KEY: &str = "g1";

fn node_tracker_params(key: &str) -> LdAiConfigTrackerParams {
    LdAiConfigTrackerParams::new(format!("run-{key}"), key, "var-1", 1)
}

fn node_resumption_token(client: &Arc<FakeLdClient>, key: &str) -> String {
    LdAiConfigTracker::new(
        client.dyn_client(),
        test_context(),
        node_tracker_params(key),
    )
    .resumption_token()
}

fn agent_config(client: &Arc<FakeLdClient>, key: &str) -> LdAiAgentConfig {
    let client = Arc::clone(client);
    let tracker_key = key.to_string();
    LdAiAgentConfig {
        key: key.to_string(),
        enabled: true,
        model: None,
        provider: None,
        instructions: None,
        judge_configuration: None,
        tools: None,
        tracker_factory: TrackerFactory::new(move || {
            LdAiConfigTracker::new(
                client.dyn_client(),
                test_context(),
                node_tracker_params(&tracker_key),
            )
        }),
        evaluator: Evaluator::noop(),
    }
}

struct Fixture {
    client: Arc<FakeLdClient>,
    trackers_created: Arc<AtomicUsize>,
    definition: AgentGraphDefinition,
}

impl Fixture {
    fn with_nodes(keys: &[&str]) -> Self {
        let client = FakeLdClient::new();
        let edges: serde_json::Map<String, serde_json::Value> = keys
            .windows(2)
            .map(|pair| (pair[0].to_string(), json!([{ "key": pair[1] }])))
            .collect();
        let graph = LdAgentGraphFlagValue::from_value(&json!({
            "_ldMeta": { "variationKey": "graph-var", "version": 1 },
            "root": keys.first().copied().unwrap_or("root"),
            "edges": edges,
        }));
        let configs: HashMap<String, LdAiAgentConfig> = keys
            .iter()
            .map(|key| (key.to_string(), agent_config(&client, key)))
            .collect();
        let nodes = AgentGraphDefinition::build_nodes(&graph, &configs);

        let trackers_created = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&trackers_created);
        let tracker_client = Arc::clone(&client);
        let definition = AgentGraphDefinition::new(graph, nodes, true, move || {
            counter.fetch_add(1, Ordering::SeqCst);
            LdGraphTracker::new(
                tracker_client.dyn_client(),
                uuid::Uuid::new_v4().to_string(),
                GRAPH_KEY,
                Some("graph-var".to_string()),
                1,
                test_context(),
            )
        });

        Self {
            client,
            trackers_created,
            definition,
        }
    }

    fn managed(&self) -> ManagedAgentGraph {
        ManagedAgentGraph::new(
            self.definition.clone(),
            Some(Arc::clone(&self.client.logger) as Arc<dyn LdLogger>),
        )
    }

    fn tracks_for_node(&self, key: &str) -> Vec<TrackCall> {
        self.client
            .tracks()
            .into_iter()
            .filter(|t| {
                t.data
                    .as_ref()
                    .and_then(|d| d.get("configKey"))
                    .and_then(|k| k.as_str())
                    == Some(key)
            })
            .collect()
    }
}

fn tokens(total: u64, input: u64, output: u64) -> LdTokenUsage {
    LdTokenUsage {
        total,
        input,
        output,
    }
}

fn graph_metrics(
    success: bool,
    node_metrics: impl IntoIterator<Item = (&'static str, LdAiMetrics)>,
) -> LdAiGraphMetrics {
    LdAiGraphMetrics {
        success,
        node_metrics: node_metrics
            .into_iter()
            .map(|(k, m)| (k.to_string(), m))
            .collect(),
        ..LdAiGraphMetrics::default()
    }
}

fn runner_result(content: &str, metrics: LdAiGraphMetrics) -> AgentGraphRunnerResult {
    AgentGraphRunnerResult {
        content: content.to_string(),
        metrics,
        raw: None,
    }
}

mod managed_agent_graph {
    use super::*;

    #[tokio::test]
    async fn builds_managed_graph_result_from_runner_result() {
        let fixture = Fixture::with_nodes(&["node-a", "node-b"]);
        let metrics = LdAiGraphMetrics {
            success: true,
            path: vec!["node-a".to_string(), "node-b".to_string()],
            duration_ms: Some(1500.0),
            tokens: Some(tokens(100, 50, 50)),
            node_metrics: IndexMap::from([
                (
                    "node-a".to_string(),
                    LdAiMetrics {
                        tokens: Some(tokens(40, 20, 20)),
                        ..LdAiMetrics::new(true)
                    },
                ),
                (
                    "node-b".to_string(),
                    LdAiMetrics {
                        tokens: Some(tokens(60, 30, 30)),
                        ..LdAiMetrics::new(true)
                    },
                ),
            ]),
        };
        let graph_token = Arc::new(Mutex::new(None));
        let seen_token = Arc::clone(&graph_token);

        let result = fixture
            .managed()
            .run(|_def, tracker| async move {
                *seen_token.lock().unwrap() = Some(tracker.resumption_token());
                Ok::<_, BoxError>(runner_result("Graph output", metrics))
            })
            .await
            .unwrap();

        assert_eq!(result.content, "Graph output");
        assert!(result.metrics.success);
        assert_eq!(result.metrics.path, ["node-a", "node-b"]);
        assert_eq!(result.metrics.duration_ms, Some(1500.0));
        assert_eq!(result.metrics.tokens, Some(tokens(100, 50, 50)));
        assert_eq!(
            result.metrics.resumption_token,
            graph_token.lock().unwrap().clone()
        );
        assert_eq!(
            result.metrics.node_metrics,
            IndexMap::from([
                (
                    "node-a".to_string(),
                    LdAiMetricSummary {
                        success: Some(true),
                        tokens: Some(tokens(40, 20, 20)),
                        resumption_token: Some(node_resumption_token(&fixture.client, "node-a")),
                        ..LdAiMetricSummary::default()
                    },
                ),
                (
                    "node-b".to_string(),
                    LdAiMetricSummary {
                        success: Some(true),
                        tokens: Some(tokens(60, 30, 30)),
                        resumption_token: Some(node_resumption_token(&fixture.client, "node-b")),
                        ..LdAiMetricSummary::default()
                    },
                ),
            ])
        );
    }

    #[tokio::test]
    async fn fires_tracking_events_into_per_node_trackers() {
        let fixture = Fixture::with_nodes(&["n1"]);
        let metrics = LdAiGraphMetrics {
            path: vec!["n1".to_string()],
            ..graph_metrics(
                true,
                [(
                    "n1",
                    LdAiMetrics {
                        success: true,
                        tokens: Some(tokens(10, 5, 5)),
                        duration_ms: Some(200.0),
                        tool_calls: Some(vec!["tool-a".to_string()]),
                    },
                )],
            )
        };

        let result = fixture
            .managed()
            .run(|_, _| async move { Ok::<_, BoxError>(runner_result("out", metrics)) })
            .await
            .unwrap();

        let events: Vec<(String, Option<f64>)> = fixture
            .tracks_for_node("n1")
            .into_iter()
            .map(|t| (t.key, t.metric_value))
            .collect();
        let expect = |key: &str, value: f64| (key.to_string(), Some(value));
        assert_eq!(
            events,
            [
                expect("$ld:ai:tokens:total", 10.0),
                expect("$ld:ai:tokens:input", 5.0),
                expect("$ld:ai:tokens:output", 5.0),
                expect("$ld:ai:duration:total", 200.0),
                expect("$ld:ai:tool_call", 1.0),
                expect("$ld:ai:generation:success", 1.0),
            ]
        );
        let tool_call = &fixture.client.tracks_for("$ld:ai:tool_call")[0];
        assert_eq!(tool_call.data.as_ref().unwrap()["toolKey"], json!("tool-a"));
        assert_eq!(
            result.metrics.node_metrics["n1"],
            LdAiMetricSummary {
                success: Some(true),
                tokens: Some(tokens(10, 5, 5)),
                tool_calls: Some(vec!["tool-a".to_string()]),
                duration_ms: Some(200.0),
                resumption_token: Some(node_resumption_token(&fixture.client, "n1")),
                ..LdAiMetricSummary::default()
            }
        );
    }

    #[tokio::test]
    async fn calls_track_error_for_failed_nodes() {
        let fixture = Fixture::with_nodes(&["n1"]);

        fixture
            .managed()
            .run(|_, _| async move {
                Ok::<_, BoxError>(runner_result(
                    "",
                    graph_metrics(false, [("n1", LdAiMetrics::new(false))]),
                ))
            })
            .await
            .unwrap();

        assert_eq!(
            fixture.client.tracks_for("$ld:ai:generation:error").len(),
            1
        );
        assert!(fixture
            .client
            .tracks_for("$ld:ai:generation:success")
            .is_empty());
    }

    #[tokio::test]
    async fn skips_node_metrics_when_get_node_returns_undefined() {
        let fixture = Fixture::with_nodes(&["n1"]);

        let result = fixture
            .managed()
            .run(|_, _| async move {
                Ok::<_, BoxError>(runner_result(
                    "",
                    graph_metrics(true, [("missing", LdAiMetrics::new(true))]),
                ))
            })
            .await
            .unwrap();

        assert!(result.metrics.node_metrics.is_empty());
        assert!(fixture.client.tracks().is_empty());
        assert_eq!(
            fixture.client.logger.warnings(),
            ["ManagedAgentGraph: no node found for key \"missing\", skipping metrics"]
        );
    }

    #[tokio::test]
    async fn passes_graph_definition_and_graph_tracker_to_runner() {
        let fixture = Fixture::with_nodes(&["n1"]);
        let received: Arc<Mutex<Option<(AgentGraphDefinition, LdGraphTracker)>>> =
            Arc::new(Mutex::new(None));
        let sink = Arc::clone(&received);

        let result = fixture
            .managed()
            .run(|def, tracker| async move {
                *sink.lock().unwrap() = Some((def, tracker));
                Ok::<_, BoxError>(runner_result("output", graph_metrics(true, [])))
            })
            .await
            .unwrap();

        let (def, tracker) = received.lock().unwrap().take().unwrap();
        assert_eq!(def.get_config(), fixture.definition.get_config());
        assert_eq!(format!("{def:?}"), format!("{:?}", fixture.definition));
        assert_eq!(fixture.trackers_created.load(Ordering::SeqCst), 1);
        assert_eq!(
            Some(tracker.resumption_token()),
            result.metrics.resumption_token
        );
    }

    #[tokio::test]
    async fn creates_a_tracker_via_graph_definition_create_tracker() {
        let fixture = Fixture::with_nodes(&["n1"]);

        fixture
            .managed()
            .run(
                |_, _| async move { Ok::<_, BoxError>(runner_result("", graph_metrics(true, []))) },
            )
            .await
            .unwrap();

        assert_eq!(fixture.trackers_created.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn resolves_to_empty_evaluations_by_default() {
        let fixture = Fixture::with_nodes(&["n1"]);

        let result = fixture
            .managed()
            .run(
                |_, _| async move { Ok::<_, BoxError>(runner_result("", graph_metrics(true, []))) },
            )
            .await
            .unwrap();

        assert!(result.evaluations.await.is_empty());
    }

    #[test]
    fn returns_the_graph_definition_via_get_graph_definition() {
        let fixture = Fixture::with_nodes(&["n1"]);
        let managed = fixture.managed();
        let returned = managed.get_graph_definition();
        assert_eq!(returned.get_config(), fixture.definition.get_config());
        assert_eq!(format!("{returned:?}"), format!("{:?}", fixture.definition));
        assert_eq!(fixture.trackers_created.load(Ordering::SeqCst), 0);
    }
}
