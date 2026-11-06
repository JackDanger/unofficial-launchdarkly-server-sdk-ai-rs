use std::sync::Arc;

use super::{Judge, LdJudgeResult};

/// Runs every judge attached to an AI Config against an input/output pair
/// (JavaScript: `Evaluator`). It does not track results; the managed layer does.
#[derive(Debug, Clone, Default)]
pub struct Evaluator {
    judges: Vec<Arc<Judge>>,
}

impl Evaluator {
    /// An evaluator over `judges`.
    pub fn new(judges: Vec<Arc<Judge>>) -> Self {
        Self { judges }
    }

    /// An evaluator with no judges.
    pub fn noop() -> Self {
        Self::default()
    }

    /// The judges this evaluator runs.
    pub fn judges(&self) -> &[Arc<Judge>] {
        &self.judges
    }

    /// Runs all judges concurrently and returns their results in judge order.
    pub async fn evaluate(&self, input: &str, output: &str) -> Vec<LdJudgeResult> {
        if self.judges.is_empty() {
            return Vec::new();
        }
        futures::future::join_all(self.judges.iter().map(|judge| judge.evaluate(input, output, None)))
            .await
    }
}
