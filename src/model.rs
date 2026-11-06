//! Runner and managed results (`api/model` in JavaScript).

use serde_json::{Map, Value};

use crate::judge::LdJudgeResult;
use crate::metrics::{LdAiMetricSummary, LdAiMetrics};
use crate::runtime::Eager;

/// The result of a [`crate::providers::Runner`] invocation (JavaScript: `RunnerResult`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunnerResult {
    /// The text content of the model's response.
    pub content: String,
    /// Metrics for the operation.
    pub metrics: LdAiMetrics,
    /// The raw provider response, if available.
    pub raw: Option<Value>,
    /// Parsed structured output, if the provider returned structured data.
    pub parsed: Option<Map<String, Value>>,
}

/// Judge results computed in the background (JavaScript: `Promise<LDJudgeResult[]>`).
///
/// Evaluation starts as soon as a run returns; awaiting this guarantees both the
/// evaluations and their tracking are complete. On a Tokio runtime it proceeds even
/// if never awaited.
pub type JudgeEvaluations = Eager<Vec<LdJudgeResult>>;

/// The result of a managed model or agent run (JavaScript: `ManagedResult`).
#[derive(Debug)]
pub struct ManagedResult {
    /// The text content of the response.
    pub content: String,
    /// Summary of the metrics tracked for this run.
    pub metrics: LdAiMetricSummary,
    /// The raw provider response, if available.
    pub raw: Option<Value>,
    /// Parsed structured output, if available.
    pub parsed: Option<Map<String, Value>>,
    /// Judge results; awaiting guarantees evaluation and tracking are complete.
    pub evaluations: JudgeEvaluations,
}
