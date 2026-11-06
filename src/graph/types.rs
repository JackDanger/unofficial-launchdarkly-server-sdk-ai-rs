use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::judge::LdJudgeResult;
use crate::metrics::{LdAiMetricSummary, LdAiMetrics, LdTokenUsage};
use crate::model::JudgeEvaluations;

/// A directed edge from a source node to a target node (JavaScript: `LDGraphEdge`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LdGraphEdge {
    /// The key of the target agent config node.
    pub key: String,
    /// Optional handoff options customizing how data flows between nodes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handoff: Option<Map<String, Value>>,
}

impl LdGraphEdge {
    /// An edge to `key` with no handoff options.
    pub fn to(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            handoff: None,
        }
    }
}

/// `_ldMeta` of an agent graph flag.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LdAgentGraphMeta {
    /// The flag variation key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variation_key: Option<String>,
    /// The flag variation version.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<u64>,
    /// Whether the graph is enabled; only an explicit `false` disables it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

/// The raw agent graph flag value (JavaScript: `LDAgentGraphFlagValue`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LdAgentGraphFlagValue {
    /// Flag metadata.
    #[serde(rename = "_ldMeta", default, skip_serializing_if = "Option::is_none")]
    pub ld_meta: Option<LdAgentGraphMeta>,
    /// The key of the root agent config.
    #[serde(default)]
    pub root: String,
    /// Source agent config key to outgoing edges.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edges: Option<IndexMap<String, Vec<LdGraphEdge>>>,
}

impl LdAgentGraphFlagValue {
    /// Parses a flag value leniently: malformed fields read as absent, as an untyped
    /// JavaScript object would.
    pub fn from_value(value: &Value) -> Self {
        let empty = Map::new();
        let object = value.as_object().unwrap_or(&empty);
        let ld_meta = object.get("_ldMeta").and_then(Value::as_object).map(|m| LdAgentGraphMeta {
            variation_key: m.get("variationKey").and_then(Value::as_str).map(str::to_string),
            version: crate::config::utils::as_version(m.get("version")),
            enabled: m.get("enabled").and_then(Value::as_bool),
        });
        let edges = object.get("edges").and_then(Value::as_object).map(|edges| {
            edges
                .iter()
                .map(|(source, targets)| {
                    let targets = targets
                        .as_array()
                        .map(|items| {
                            items
                                .iter()
                                .filter_map(|e| serde_json::from_value(e.clone()).ok())
                                .collect()
                        })
                        .unwrap_or_default();
                    (source.clone(), targets)
                })
                .collect()
        });
        Self {
            ld_meta,
            root: object
                .get("root")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            edges,
        }
    }

    /// The outgoing edges of `key`.
    pub fn edges_of(&self, key: &str) -> &[LdGraphEdge] {
        self.edges
            .as_ref()
            .and_then(|edges| edges.get(key))
            .map(Vec::as_slice)
            .unwrap_or_default()
    }
}

/// Graph-level metrics for a completed run (JavaScript: `LDAIGraphMetricSummary`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LdAiGraphMetricSummary {
    /// Whether the graph invocation succeeded.
    pub success: bool,
    /// Execution path as an ordered list of config keys.
    pub path: Vec<String>,
    /// Per-node metric summaries keyed by agent config key.
    pub node_metrics: IndexMap<String, LdAiMetricSummary>,
    /// Total duration in milliseconds, if tracked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<f64>,
    /// Aggregate token usage, if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<LdTokenUsage>,
    /// Resumption token for deferred feedback association.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resumption_token: Option<String>,
}

/// What a graph tracker has recorded so far (JavaScript:
/// `Partial<LDAIGraphMetricSummary>`). Node metrics live on the per-node trackers.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LdAiGraphTrackerSummary {
    /// Whether the invocation succeeded, once recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub success: Option<bool>,
    /// The path taken, once recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<Vec<String>>,
    /// Total duration in milliseconds, once recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<f64>,
    /// Aggregate token usage, once recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<LdTokenUsage>,
    /// Resumption token for this run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resumption_token: Option<String>,
}

/// Graph-level metrics a graph runner reports (JavaScript: `LDAIGraphMetrics`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LdAiGraphMetrics {
    /// Whether the graph invocation succeeded.
    pub success: bool,
    /// Execution path as an ordered list of config keys.
    pub path: Vec<String>,
    /// Total duration in milliseconds, if tracked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<f64>,
    /// Aggregate token usage, if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<LdTokenUsage>,
    /// Per-node metrics keyed by agent config key.
    pub node_metrics: IndexMap<String, LdAiMetrics>,
}

/// The result of a graph runner invocation (JavaScript: `AgentGraphRunnerResult`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentGraphRunnerResult {
    /// The text content of the graph's final response.
    pub content: String,
    /// Graph-level metrics for this invocation.
    pub metrics: LdAiGraphMetrics,
    /// The raw provider response, if available.
    pub raw: Option<Value>,
}

/// The result of [`crate::ManagedAgentGraph::run`] (JavaScript: `ManagedGraphResult`).
#[derive(Debug)]
pub struct ManagedGraphResult {
    /// The text content of the graph's final response.
    pub content: String,
    /// Summarized metrics for this graph invocation.
    pub metrics: LdAiGraphMetricSummary,
    /// The raw provider response, if available.
    pub raw: Option<Value>,
    /// Judge results; JavaScript 2.0.19 always resolves this to an empty list.
    pub evaluations: JudgeEvaluations,
}

/// Graph tracker metadata (JavaScript: `LDGraphTrackData`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LdGraphTrackData {
    /// UUIDv4 identifying this run.
    pub run_id: String,
    /// The graph configuration key.
    pub graph_key: String,
    /// The variation key; absent when a default config was used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variation_key: Option<String>,
    /// The flag variation version.
    pub version: u64,
}

pub(crate) fn empty_evaluations() -> Vec<LdJudgeResult> {
    Vec::new()
}
