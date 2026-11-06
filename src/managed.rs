//! Managed invocations with automatic tracking and judge evaluation
//! (`ManagedModel`, `ManagedAgent` and `ManagedAgentGraph` in JavaScript).

use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use futures::FutureExt;
use indexmap::IndexMap;

use crate::client_min::LdLogger;
use crate::config::{LdAiAgentConfig, LdAiCompletionConfig};
use crate::error::BoxError;
use crate::graph::{
    empty_evaluations, AgentGraphDefinition, AgentGraphRunnerResult, LdAiGraphMetricSummary,
    LdGraphTracker, ManagedGraphResult,
};
use crate::judge::{Evaluator, LdJudgeResult};
use crate::metrics::{LdAiMetricSummary, LdAiMetrics};
use crate::model::{JudgeEvaluations, ManagedResult, RunnerResult};
use crate::providers::Runner;
use crate::runtime::Eager;
use crate::tracker::LdAiConfigTracker;

fn no_results() -> Vec<LdJudgeResult> {
    Vec::new()
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "panic".to_string())
}

async fn run_managed(
    tracker: LdAiConfigTracker,
    runner: &dyn Runner,
    evaluator: &Evaluator,
    logger: Option<Arc<dyn LdLogger>>,
    prompt: &str,
) -> Result<ManagedResult, BoxError> {
    let result = tracker
        .track_metrics_of(
            |r: &RunnerResult| r.metrics.clone(),
            || runner.run(prompt, None),
        )
        .await?;
    let metrics = tracker.get_summary();

    let evaluator = evaluator.clone();
    let input = prompt.to_string();
    let output = result.content.clone();
    let evaluations = Eager::spawn(
        async move {
            let evaluated = AssertUnwindSafe(async {
                let results = evaluator.evaluate(&input, &output).await;
                for judge_result in results.iter().filter(|r| r.sampled) {
                    tracker.track_judge_result(judge_result);
                }
                results
            })
            .catch_unwind()
            .await;
            evaluated.unwrap_or_else(|panic| {
                if let Some(logger) = logger {
                    logger.warn(&format!(
                        "Judge evaluation failed unexpectedly: {}",
                        panic_message(panic.as_ref())
                    ));
                }
                Vec::new()
            })
        },
        no_results,
    );

    Ok(ManagedResult {
        content: result.content,
        metrics,
        raw: result.raw,
        parsed: result.parsed,
        evaluations,
    })
}

/// Chat-completion invocation with automatic tracking and judge evaluation
/// (JavaScript: `ManagedModel`). Stateless: each [`Self::run`] sends the prompt
/// directly to the runner. Obtain one from [`crate::LdAiClient::create_model`].
#[derive(Clone)]
pub struct ManagedModel {
    ai_config: LdAiCompletionConfig,
    runner: Arc<dyn Runner>,
    logger: Option<Arc<dyn LdLogger>>,
}

impl std::fmt::Debug for ManagedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagedModel")
            .field("ai_config", &self.ai_config)
            .finish()
    }
}

impl ManagedModel {
    /// Wraps a completion config and a runner.
    pub fn new(
        ai_config: LdAiCompletionConfig,
        runner: Arc<dyn Runner>,
        logger: Option<Arc<dyn LdLogger>>,
    ) -> Self {
        Self {
            ai_config,
            runner,
            logger,
        }
    }

    /// Invokes the model with `prompt` on a fresh tracker.
    ///
    /// Returns before [`ManagedResult::evaluations`] resolves; awaiting it guarantees
    /// judge evaluation and `track_judge_result` are complete. When the runner fails,
    /// its duration and an error are tracked and the error is returned.
    pub async fn run(&self, prompt: &str) -> Result<ManagedResult, BoxError> {
        run_managed(
            self.ai_config.create_tracker(),
            self.runner.as_ref(),
            &self.ai_config.evaluator,
            self.logger.clone(),
            prompt,
        )
        .await
    }

    /// The completion config this model was created from.
    pub fn get_config(&self) -> &LdAiCompletionConfig {
        &self.ai_config
    }

    /// The runner this model invokes.
    pub fn get_runner(&self) -> &Arc<dyn Runner> {
        &self.runner
    }
}

/// Agent invocation with automatic tracking and judge evaluation
/// (JavaScript: `ManagedAgent`). Obtain one from [`crate::LdAiClient::create_agent`].
#[derive(Clone)]
pub struct ManagedAgent {
    ai_agent_config: LdAiAgentConfig,
    runner: Arc<dyn Runner>,
    logger: Option<Arc<dyn LdLogger>>,
}

impl std::fmt::Debug for ManagedAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagedAgent")
            .field("ai_agent_config", &self.ai_agent_config)
            .finish()
    }
}

impl ManagedAgent {
    /// Wraps an agent config and a runner.
    pub fn new(
        ai_agent_config: LdAiAgentConfig,
        runner: Arc<dyn Runner>,
        logger: Option<Arc<dyn LdLogger>>,
    ) -> Self {
        Self {
            ai_agent_config,
            runner,
            logger,
        }
    }

    /// Invokes the agent with `prompt` on a fresh tracker; see [`ManagedModel::run`].
    pub async fn run(&self, prompt: &str) -> Result<ManagedResult, BoxError> {
        run_managed(
            self.ai_agent_config.create_tracker(),
            self.runner.as_ref(),
            &self.ai_agent_config.evaluator,
            self.logger.clone(),
            prompt,
        )
        .await
    }

    /// The agent config this agent was created from.
    pub fn get_config(&self) -> &LdAiAgentConfig {
        &self.ai_agent_config
    }

    /// The runner this agent invokes.
    pub fn get_runner(&self) -> &Arc<dyn Runner> {
        &self.runner
    }
}

/// Runs an agent graph through a caller-supplied runner function and builds a
/// [`ManagedGraphResult`] with per-node tracking (JavaScript: `ManagedAgentGraph`).
#[derive(Debug, Clone)]
pub struct ManagedAgentGraph {
    graph_definition: AgentGraphDefinition,
    logger: Option<Arc<dyn LdLogger>>,
}

impl ManagedAgentGraph {
    /// Wraps a graph definition.
    pub fn new(graph_definition: AgentGraphDefinition, logger: Option<Arc<dyn LdLogger>>) -> Self {
        Self {
            graph_definition,
            logger,
        }
    }

    /// Runs the graph: `runner` receives the definition and a fresh graph tracker and
    /// returns the graph's result. Each node's metrics are then tracked on a new
    /// tracker for that node's agent config (tokens, duration, tool calls, then
    /// success or error); nodes not in the graph are skipped with a warning. Returns
    /// the runner's error unchanged if it fails.
    pub async fn run<F, Fut, E>(&self, runner: F) -> Result<ManagedGraphResult, E>
    where
        F: FnOnce(AgentGraphDefinition, LdGraphTracker) -> Fut,
        Fut: Future<Output = Result<AgentGraphRunnerResult, E>>,
    {
        let graph_tracker = self.graph_definition.create_tracker();
        let runner_result = runner(self.graph_definition.clone(), graph_tracker.clone()).await?;
        let graph_metrics = runner_result.metrics;

        let metrics = LdAiGraphMetricSummary {
            success: graph_metrics.success,
            path: graph_metrics.path,
            duration_ms: graph_metrics.duration_ms,
            tokens: graph_metrics.tokens,
            node_metrics: self.track_node_metrics(&graph_metrics.node_metrics),
            resumption_token: Some(graph_tracker.resumption_token()),
        };

        let evaluations: JudgeEvaluations = Eager::ready(Vec::new(), empty_evaluations);
        Ok(ManagedGraphResult {
            content: runner_result.content,
            metrics,
            raw: runner_result.raw,
            evaluations,
        })
    }

    fn track_node_metrics(
        &self,
        node_metrics: &IndexMap<String, LdAiMetrics>,
    ) -> IndexMap<String, LdAiMetricSummary> {
        let mut summaries = IndexMap::new();
        for (node_key, metrics) in node_metrics {
            let Some(node) = self.graph_definition.get_node(node_key) else {
                if let Some(logger) = &self.logger {
                    logger.warn(&format!(
                        "ManagedAgentGraph: no node found for key \"{node_key}\", skipping metrics"
                    ));
                }
                continue;
            };
            let tracker = node.get_config().create_tracker();
            if let Some(tokens) = metrics.tokens {
                tracker.track_tokens(tokens);
            }
            if let Some(duration_ms) = metrics.duration_ms {
                tracker.track_duration(duration_ms);
            }
            if let Some(tool_calls) = metrics.tool_calls.as_ref().filter(|c| !c.is_empty()) {
                tracker.track_tool_calls(tool_calls);
            }
            if metrics.success {
                tracker.track_success();
            } else {
                tracker.track_error();
            }
            summaries.insert(node_key.clone(), tracker.get_summary());
        }
        summaries
    }

    /// The underlying graph definition.
    pub fn get_graph_definition(&self) -> &AgentGraphDefinition {
        &self.graph_definition
    }
}
