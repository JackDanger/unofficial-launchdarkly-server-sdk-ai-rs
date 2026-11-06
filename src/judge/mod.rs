//! Judges: AI Configs in `judge` mode that score other AI output (`api/judge`).

mod evaluator;

use std::fmt;
use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::client_min::LdLogger;
use crate::config::{LdAiJudgeConfig, LdMessage, LdMessageRole};
use crate::model::RunnerResult;
use crate::providers::Runner;

pub use evaluator::Evaluator;

/// Result of a judge evaluation (JavaScript: `LDJudgeResult`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LdJudgeResult {
    /// The key of the judge configuration that produced this result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_config_key: Option<String>,
    /// Whether the evaluation completed successfully.
    pub success: bool,
    /// Error message if the evaluation failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Whether the evaluation was sampled (actually run); false when skipped by sampling.
    pub sampled: bool,
    /// The metric key this evaluation reports under.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric_key: Option<String>,
    /// Score between 0.0 and 1.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// Reasoning behind the score.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

/// The JSON schema judges request as structured output: `{score: 0..1, reasoning}`.
pub fn evaluation_schema() -> &'static Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        json!({
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Score between 0.0 and 1.0."
                },
                "reasoning": {
                    "type": "string",
                    "description": "Reasoning behind the score."
                }
            },
            "required": ["score", "reasoning"],
            "additionalProperties": false
        })
    })
}

/// Removes legacy judge template messages: every non-system message whose content
/// contains `{{message_history}}` or `{{response_to_evaluate}}`.
pub fn strip_legacy_judge_messages(messages: &[LdMessage]) -> Vec<LdMessage> {
    messages
        .iter()
        .filter(|msg| {
            msg.role == LdMessageRole::System
                || (!msg.content.contains("{{message_history}}")
                    && !msg.content.contains("{{response_to_evaluate}}"))
        })
        .cloned()
        .collect()
}

/// Returns a uniformly distributed number in `[0, 1)`; judges sample with it.
pub type RandomSource = Arc<dyn Fn() -> f64 + Send + Sync>;

/// Evaluates AI output with a judge AI Config and a structured-output [`Runner`]
/// (JavaScript: `Judge`).
#[derive(Clone)]
pub struct Judge {
    ai_config: LdAiJudgeConfig,
    runner: Arc<dyn Runner>,
    sample_rate: f64,
    logger: Option<Arc<dyn LdLogger>>,
    random: RandomSource,
}

impl fmt::Debug for Judge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Judge")
            .field("key", &self.ai_config.key)
            .field("sample_rate", &self.sample_rate)
            .finish()
    }
}

impl Judge {
    /// Creates a judge. `sample_rate` is the default sampling rate (JavaScript default 1.0).
    pub fn new(
        ai_config: LdAiJudgeConfig,
        runner: Arc<dyn Runner>,
        sample_rate: f64,
        logger: Option<Arc<dyn LdLogger>>,
    ) -> Self {
        Self {
            ai_config,
            runner,
            sample_rate,
            logger,
            random: Arc::new(fastrand::f64),
        }
    }

    /// Replaces the random source used for sampling (JavaScript uses `Math.random`).
    pub fn with_random_source(mut self, random: RandomSource) -> Self {
        self.random = random;
        self
    }

    /// The default sampling rate used when no per-call rate is supplied.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// The judge's AI Config.
    pub fn get_ai_config(&self) -> &LdAiJudgeConfig {
        &self.ai_config
    }

    /// The judge's runner.
    pub fn get_runner(&self) -> &Arc<dyn Runner> {
        &self.runner
    }

    fn evaluation_metric_key(&self) -> Option<String> {
        self.ai_config
            .evaluation_metric_key
            .as_deref()
            .map(str::trim)
            .filter(|k| !k.is_empty())
            .map(str::to_string)
    }

    /// Evaluates `output` as a response to `input`.
    ///
    /// `sampling_rate` overrides the judge's default rate; an explicit `Some(0.0)`
    /// skips evaluation, only `None` falls back to the default.
    pub async fn evaluate(&self, input: &str, output: &str, sampling_rate: Option<f64>) -> LdJudgeResult {
        let effective_rate = sampling_rate.unwrap_or(self.sample_rate);
        let mut result = LdJudgeResult {
            judge_config_key: Some(self.ai_config.key.clone()),
            ..LdJudgeResult::default()
        };
        let tracker = self.ai_config.create_tracker();

        let Some(evaluation_metric_key) = self.evaluation_metric_key() else {
            self.warn(&format!(
                "Judge configuration is missing required evaluation metric key {}",
                serde_json::to_string(&tracker.get_track_data()).unwrap_or_default()
            ));
            result.sampled = true;
            result.error_message =
                Some("Judge configuration is missing required evaluation metric key".to_string());
            return result;
        };

        if (self.random)() > effective_rate {
            if let Some(logger) = &self.logger {
                logger.debug(&format!(
                    "Judge evaluation skipped due to sampling rate: {effective_rate}"
                ));
            }
            return result;
        }

        result.sampled = true;
        let evaluation_input = build_evaluation_input(input, output);

        let response = tracker
            .track_metrics_of(
                |r: &RunnerResult| r.metrics.clone(),
                || self.runner.run(&evaluation_input, Some(evaluation_schema())),
            )
            .await;

        let response = match response {
            Ok(response) => response,
            Err(err) => {
                if let Some(logger) = &self.logger {
                    logger.error(&format!("Judge evaluation failed: {err}"));
                }
                result.error_message = Some(err.to_string());
                return result;
            }
        };

        let Some((score, reasoning)) = parse_evaluation_response(response.parsed.as_ref()) else {
            self.warn(&format!(
                "Could not parse evaluation response: {} {}",
                response
                    .parsed
                    .as_ref()
                    .map(|p| Value::Object(p.clone()).to_string())
                    .unwrap_or_else(|| "undefined".to_string()),
                serde_json::to_string(&tracker.get_track_data()).unwrap_or_default()
            ));
            return result;
        };

        LdJudgeResult {
            success: response.metrics.success,
            score: Some(score),
            reasoning: Some(reasoning),
            metric_key: Some(evaluation_metric_key),
            ..result
        }
    }

    /// Evaluates a runner response to a conversation. Each message is rendered as
    /// `<role>: <content>`, joined by newlines.
    pub async fn evaluate_messages(
        &self,
        messages: &[LdMessage],
        response: &RunnerResult,
        sampling_ratio: Option<f64>,
    ) -> LdJudgeResult {
        let input = messages
            .iter()
            .map(|msg| format!("{}: {}", msg.role, msg.content))
            .collect::<Vec<_>>()
            .join("\n");
        self.evaluate(&input, &response.content, sampling_ratio).await
    }

    fn warn(&self, message: &str) {
        if let Some(logger) = &self.logger {
            logger.warn(message);
        }
    }
}

/// The input string a judge's runner receives.
pub fn build_evaluation_input(input: &str, output: &str) -> String {
    format!("MESSAGE HISTORY:\n{input}\n\nRESPONSE TO EVALUATE:\n{output}")
}

/// Parses a structured evaluation `{score, reasoning}`; `None` unless `score` is a
/// number within `[0, 1]` and `reasoning` is a string.
pub fn parse_evaluation_response(data: Option<&Map<String, Value>>) -> Option<(f64, String)> {
    let data = data?;
    let score = data.get("score")?.as_f64().filter(|s| (0.0..=1.0).contains(s))?;
    let reasoning = data.get("reasoning")?.as_str()?.to_string();
    Some((score, reasoning))
}
