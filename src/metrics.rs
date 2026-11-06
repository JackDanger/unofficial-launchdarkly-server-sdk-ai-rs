//! Metric types (`api/metrics` and `LDAIMetricSummary` in JavaScript).

use serde::{Deserialize, Serialize};

/// Information about token usage (JavaScript: `LDTokenUsage`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LdTokenUsage {
    /// Combined token usage.
    pub total: u64,
    /// Number of tokens in the input.
    pub input: u64,
    /// Number of tokens in the output.
    pub output: u64,
}

/// Feedback about the generated content (JavaScript: `LDFeedbackKind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LdFeedbackKind {
    /// The sentiment was positive.
    Positive,
    /// The sentiment was negative.
    Negative,
}

/// User feedback as recorded by a tracker (JavaScript: `{ kind: LDFeedbackKind }`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LdFeedback {
    /// The sentiment of the feedback.
    pub kind: LdFeedbackKind,
}

impl From<LdFeedbackKind> for LdFeedback {
    fn from(kind: LdFeedbackKind) -> Self {
        Self { kind }
    }
}

/// Metrics for one AI operation (JavaScript: `LDAIMetrics`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LdAiMetrics {
    /// Whether the AI operation was successful.
    pub success: bool,
    /// Token usage for the operation, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<LdTokenUsage>,
    /// Tool call identifiers made during the operation, when any were made.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<String>>,
    /// Duration of the operation in milliseconds, when tracked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<f64>,
}

impl LdAiMetrics {
    /// Metrics carrying only a success flag.
    pub fn new(success: bool) -> Self {
        Self {
            success,
            ..Self::default()
        }
    }
}

/// A flat summary of what a tracker has recorded (JavaScript: `LDAIMetricSummary`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LdAiMetricSummary {
    /// Whether the AI operation was successful, once recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub success: Option<bool>,
    /// Token usage, once recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<LdTokenUsage>,
    /// Tool call identifiers recorded so far.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<String>>,
    /// Duration in milliseconds, once recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<f64>,
    /// Time to first token in milliseconds, once recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_to_first_token_ms: Option<f64>,
    /// User feedback, once recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feedback: Option<LdFeedback>,
    /// Resumption token for associating deferred feedback with this run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resumption_token: Option<String>,
}
