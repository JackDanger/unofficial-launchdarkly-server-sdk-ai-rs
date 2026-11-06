//! The AI client (`LDAIClient` / `LDAIClientImpl` in JavaScript).

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Arc;

use futures::future::{join_all, BoxFuture};
use futures::FutureExt;
use indexmap::{IndexMap, IndexSet};
use serde_json::{json, Value};

use crate::client_min::{context_to_value, Context, ContextBuilder, LdClientMin, LdLogger};
use crate::config::utils::{create_disabled_config, from_flag_value, to_flag_value, FlagValue};
use crate::config::{
    LdAiAgentConfig, LdAiAgentConfigDefault, LdAiAgentRequestConfig, LdAiCompletionConfig,
    LdAiCompletionConfigDefault, LdAiConfigDefaultKind, LdAiConfigKind, LdAiConfigMode,
    LdAiJudgeConfig, LdAiJudgeConfigDefault, LdJudge, LdMessage, TrackerFactory, Variables,
};
use crate::error::Error;
use crate::graph::{AgentGraphDefinition, LdAgentGraphFlagValue, LdGraphTracker};
use crate::judge::{strip_legacy_judge_messages, Evaluator, Judge};
use crate::managed::{ManagedAgent, ManagedModel};
use crate::mustache;
use crate::providers::{ModelConfigRef, RunnerFactory};
use crate::sdk_info::{AI_SDK_LANGUAGE, AI_SDK_NAME, AI_SDK_VERSION};
use crate::tracker::{LdAiConfigTracker, LdAiConfigTrackerParams};

/// `$ld:ai:sdk:info`, sent once when a client is created.
pub const TRACK_SDK_INFO: &str = "$ld:ai:sdk:info";
/// `$ld:ai:usage:completion-config`
pub const TRACK_USAGE_COMPLETION_CONFIG: &str = "$ld:ai:usage:completion-config";
/// `$ld:ai:usage:completion-config-template`
pub const TRACK_USAGE_COMPLETION_CONFIG_TEMPLATE: &str = "$ld:ai:usage:completion-config-template";
/// `$ld:ai:usage:create-chat`, sent by [`LdAiClient::create_model`].
pub const TRACK_USAGE_CREATE_CHAT: &str = "$ld:ai:usage:create-chat";
/// `$ld:ai:usage:create-agent`
pub const TRACK_USAGE_CREATE_AGENT: &str = "$ld:ai:usage:create-agent";
/// `$ld:ai:usage:judge-config`
pub const TRACK_USAGE_JUDGE_CONFIG: &str = "$ld:ai:usage:judge-config";
/// `$ld:ai:usage:judge-config-template`
pub const TRACK_USAGE_JUDGE_CONFIG_TEMPLATE: &str = "$ld:ai:usage:judge-config-template";
/// `$ld:ai:usage:create-judge`
pub const TRACK_USAGE_CREATE_JUDGE: &str = "$ld:ai:usage:create-judge";
/// `$ld:ai:usage:agent-config`
pub const TRACK_USAGE_AGENT_CONFIG: &str = "$ld:ai:usage:agent-config";
/// `$ld:ai:usage:agent-config-template`
pub const TRACK_USAGE_AGENT_CONFIG_TEMPLATE: &str = "$ld:ai:usage:agent-config-template";
/// `$ld:ai:usage:agent-configs`
pub const TRACK_USAGE_AGENT_CONFIGS: &str = "$ld:ai:usage:agent-configs";
/// `$ld:ai:usage:agent-graph`
pub const TRACK_USAGE_AGENT_GRAPH: &str = "$ld:ai:usage:agent-graph";

/// The context the `$ld:ai:sdk:info` event is sent for:
/// `{kind: "ld_ai", key: "ld-internal-tracking", anonymous: true}`.
pub fn init_track_context() -> Context {
    ContextBuilder::new("ld-internal-tracking")
        .kind("ld_ai")
        .anonymous(true)
        .build()
        .expect("the internal tracking context is valid")
}

struct EvaluateRequest<'a> {
    key: &'a str,
    context: &'a Context,
    default_value: LdAiConfigDefaultKind,
    mode: LdAiConfigMode,
    variables: Option<&'a Variables>,
    graph_key: Option<&'a str>,
    default_ai_provider: Option<&'a str>,
    interpolate: bool,
}

/// Performs AI operations using LaunchDarkly (JavaScript: `LDAIClient`).
///
/// Create one with [`init_ai`] (or [`LdAiClient::new`]) and register providers with
/// [`LdAiClient::with_runner_factory`] if you use `create_model`, `create_agent` or
/// `create_judge`. Cloning is cheap and shares the underlying client.
#[derive(Clone)]
pub struct LdAiClient {
    ld_client: Arc<dyn LdClientMin>,
    logger: Option<Arc<dyn LdLogger>>,
    runner_factory: RunnerFactory,
}

impl fmt::Debug for LdAiClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LdAiClient")
            .field("runner_factory", &self.runner_factory)
            .finish()
    }
}

/// Initializes an AI client over a LaunchDarkly client (JavaScript: `initAi`).
///
/// Sends the `$ld:ai:sdk:info` event.
pub fn init_ai<C: LdClientMin + 'static>(ld_client: Arc<C>) -> LdAiClient {
    LdAiClient::new(ld_client)
}

impl LdAiClient {
    /// Creates a client with no AI providers registered. Sends `$ld:ai:sdk:info`.
    pub fn new(ld_client: Arc<dyn LdClientMin>) -> Self {
        Self::with_runner_factory(ld_client, RunnerFactory::new())
    }

    /// Creates a client that builds runners from `runner_factory`. Sends `$ld:ai:sdk:info`.
    pub fn with_runner_factory(ld_client: Arc<dyn LdClientMin>, runner_factory: RunnerFactory) -> Self {
        let logger = ld_client.logger();
        ld_client.track(
            TRACK_SDK_INFO,
            &init_track_context(),
            Some(json!({
                "aiSdkName": AI_SDK_NAME,
                "aiSdkVersion": AI_SDK_VERSION,
                "aiSdkLanguage": AI_SDK_LANGUAGE,
            })),
            Some(1.0),
        );
        Self {
            ld_client,
            logger,
            runner_factory,
        }
    }

    /// The provider registry used by `create_model`, `create_agent` and `create_judge`.
    pub fn runner_factory(&self) -> &RunnerFactory {
        &self.runner_factory
    }

    /// The logger taken from the LaunchDarkly client, if any.
    pub fn logger(&self) -> Option<&Arc<dyn LdLogger>> {
        self.logger.as_ref()
    }

    fn logger_ref(&self) -> Option<&dyn LdLogger> {
        self.logger.as_deref()
    }

    fn track_usage(&self, event: &str, context: &Context, data: Value, metric_value: f64) {
        self.ld_client.track(event, context, Some(data), Some(metric_value));
    }

    fn interpolate(&self, template: &str, view: &Value) -> String {
        match mustache::render(template, view) {
            Ok(rendered) => rendered,
            Err(err) => {
                if let Some(logger) = &self.logger {
                    logger.warn(&format!(
                        "Failed to interpolate AI Config template; using it uninterpolated: {err}"
                    ));
                }
                template.to_string()
            }
        }
    }

    fn tracker_factory(&self, key: &str, flag: &FlagValue, context: &Context, graph_key: Option<&str>) -> TrackerFactory {
        let ld_client = Arc::clone(&self.ld_client);
        let context = context.clone();
        let meta = flag.meta.clone().unwrap_or_default();
        let template = LdAiConfigTrackerParams {
            run_id: String::new(),
            config_key: key.to_string(),
            variation_key: meta.variation_key.unwrap_or_default(),
            version: meta.version.unwrap_or(1),
            model_name: flag.model.as_ref().map(|m| m.name.clone()).unwrap_or_default(),
            provider_name: flag.provider.as_ref().map(|p| p.name.clone()).unwrap_or_default(),
            graph_key: graph_key.map(str::to_string),
            model_key: meta.model_key,
            model_version: meta.model_version.unwrap_or(1),
        };
        TrackerFactory::new(move || {
            let mut params = template.clone();
            params.run_id = uuid::Uuid::new_v4().to_string();
            LdAiConfigTracker::new(Arc::clone(&ld_client), context.clone(), params)
        })
    }

    fn evaluate<'a>(&'a self, request: EvaluateRequest<'a>) -> BoxFuture<'a, LdAiConfigKind> {
        async move {
            let default_flag = to_flag_value(&request.default_value, request.mode);
            let value = self
                .ld_client
                .variation(request.key, request.context, default_flag)
                .await;
            let flag = FlagValue::parse(&value);
            let tracker_factory = self.tracker_factory(request.key, &flag, request.context, request.graph_key);

            let flag_mode = flag.mode().unwrap_or("completion");
            if flag_mode != request.mode.as_str() {
                if let Some(logger) = &self.logger {
                    logger.warn(&format!(
                        "AI Config mode mismatch for {}: expected {}, got {flag_mode}. Returning disabled config.",
                        request.key, request.mode
                    ));
                }
                return create_disabled_config(request.key, request.mode, tracker_factory, Evaluator::noop());
            }

            let evaluator = if flag_mode == "judge" {
                Evaluator::noop()
            } else {
                let judges = flag
                    .judge_configuration
                    .as_ref()
                    .map(|j| j.judges.as_slice())
                    .unwrap_or_default();
                self.build_evaluator(judges, request.context, request.variables, request.default_ai_provider)
                    .await
            };

            let config = from_flag_value(request.key, &flag, tracker_factory, evaluator);
            if !request.interpolate {
                return config;
            }
            self.apply_interpolation(config, request.context, request.variables)
        }
        .boxed()
    }

    fn apply_interpolation(
        &self,
        config: LdAiConfigKind,
        context: &Context,
        variables: Option<&Variables>,
    ) -> LdAiConfigKind {
        let view = mustache::view_with_context(variables, context_to_value(context));
        let render_messages = |messages: Vec<LdMessage>| -> Vec<LdMessage> {
            messages
                .into_iter()
                .map(|m| LdMessage {
                    content: self.interpolate(&m.content, &view),
                    ..m
                })
                .collect()
        };
        match config {
            LdAiConfigKind::Completion(mut c) => {
                c.messages = c.messages.map(render_messages);
                LdAiConfigKind::Completion(c)
            }
            LdAiConfigKind::Judge(mut c) => {
                c.messages = c.messages.map(render_messages);
                LdAiConfigKind::Judge(c)
            }
            LdAiConfigKind::Agent(mut c) => {
                c.instructions = c
                    .instructions
                    .map(|i| if i.is_empty() { i } else { self.interpolate(&i, &view) });
                LdAiConfigKind::Agent(c)
            }
        }
    }

    async fn build_evaluator(
        &self,
        judges: &[LdJudge],
        context: &Context,
        variables: Option<&Variables>,
        default_ai_provider: Option<&str>,
    ) -> Evaluator {
        if judges.is_empty() {
            return Evaluator::noop();
        }
        let instances = join_all(judges.iter().map(|judge| {
            self.create_judge_instance(
                &judge.key,
                context,
                None,
                variables,
                default_ai_provider,
                judge.sampling_rate,
            )
        }))
        .await;
        Evaluator::new(instances.into_iter().flatten().map(Arc::new).collect())
    }

    async fn completion_config_inner(
        &self,
        key: &str,
        context: &Context,
        default_value: LdAiCompletionConfigDefault,
        variables: Option<&Variables>,
        default_ai_provider: Option<&str>,
    ) -> LdAiCompletionConfig {
        let config = self
            .evaluate(EvaluateRequest {
                key,
                context,
                default_value: LdAiConfigDefaultKind::Completion(default_value),
                mode: LdAiConfigMode::Completion,
                variables,
                graph_key: None,
                default_ai_provider,
                interpolate: true,
            })
            .await;
        expect_completion(config)
    }

    /// Retrieves a completion AI Config with its messages interpolated
    /// (JavaScript: `completionConfig`).
    ///
    /// `variables` fill Mustache placeholders (HTML escaping is disabled); the
    /// evaluation context is always available as `{{ldctx.*}}`. When the flag is
    /// missing, `default_value` is used; `None` means a disabled default.
    pub async fn completion_config(
        &self,
        key: &str,
        context: &Context,
        default_value: Option<LdAiCompletionConfigDefault>,
        variables: Option<&Variables>,
        default_ai_provider: Option<&str>,
    ) -> LdAiCompletionConfig {
        self.track_usage(TRACK_USAGE_COMPLETION_CONFIG, context, json!(key), 1.0);
        self.completion_config_inner(
            key,
            context,
            default_value.unwrap_or_default(),
            variables,
            default_ai_provider,
        )
        .await
    }

    /// Retrieves a completion AI Config with Mustache placeholders left intact
    /// (JavaScript: `completionConfigTemplate`).
    pub async fn completion_config_template(
        &self,
        key: &str,
        context: &Context,
        default_value: Option<LdAiCompletionConfigDefault>,
        default_ai_provider: Option<&str>,
    ) -> LdAiCompletionConfig {
        self.track_usage(TRACK_USAGE_COMPLETION_CONFIG_TEMPLATE, context, json!(key), 1.0);
        let config = self
            .evaluate(EvaluateRequest {
                key,
                context,
                default_value: LdAiConfigDefaultKind::Completion(default_value.unwrap_or_default()),
                mode: LdAiConfigMode::Completion,
                variables: None,
                graph_key: None,
                default_ai_provider,
                interpolate: false,
            })
            .await;
        expect_completion(config)
    }

    async fn agent_config_inner(
        &self,
        key: &str,
        context: &Context,
        default_value: LdAiAgentConfigDefault,
        variables: Option<&Variables>,
        graph_key: Option<&str>,
        default_ai_provider: Option<&str>,
    ) -> LdAiAgentConfig {
        let config = self
            .evaluate(EvaluateRequest {
                key,
                context,
                default_value: LdAiConfigDefaultKind::Agent(default_value),
                mode: LdAiConfigMode::Agent,
                variables,
                graph_key,
                default_ai_provider,
                interpolate: true,
            })
            .await;
        expect_agent(config)
    }

    /// Retrieves an agent AI Config with its instructions interpolated
    /// (JavaScript: `agentConfig`).
    pub async fn agent_config(
        &self,
        key: &str,
        context: &Context,
        default_value: Option<LdAiAgentConfigDefault>,
        variables: Option<&Variables>,
        default_ai_provider: Option<&str>,
    ) -> LdAiAgentConfig {
        self.track_usage(TRACK_USAGE_AGENT_CONFIG, context, json!(key), 1.0);
        self.agent_config_inner(
            key,
            context,
            default_value.unwrap_or_default(),
            variables,
            None,
            default_ai_provider,
        )
        .await
    }

    /// Retrieves an agent AI Config with Mustache placeholders left intact
    /// (JavaScript: `agentConfigTemplate`).
    pub async fn agent_config_template(
        &self,
        key: &str,
        context: &Context,
        default_value: Option<LdAiAgentConfigDefault>,
        default_ai_provider: Option<&str>,
    ) -> LdAiAgentConfig {
        self.track_usage(TRACK_USAGE_AGENT_CONFIG_TEMPLATE, context, json!(key), 1.0);
        let config = self
            .evaluate(EvaluateRequest {
                key,
                context,
                default_value: LdAiConfigDefaultKind::Agent(default_value.unwrap_or_default()),
                mode: LdAiConfigMode::Agent,
                variables: None,
                graph_key: None,
                default_ai_provider,
                interpolate: false,
            })
            .await;
        expect_agent(config)
    }

    async fn judge_config_inner(
        &self,
        key: &str,
        context: &Context,
        default_value: LdAiJudgeConfigDefault,
        variables: Option<&Variables>,
    ) -> LdAiJudgeConfig {
        for reserved in ["message_history", "response_to_evaluate"] {
            if variables.is_some_and(|v| v.contains_key(reserved)) {
                if let Some(logger) = &self.logger {
                    logger.warn(&format!(
                        "The variable '{reserved}' is reserved by the judge and will be ignored."
                    ));
                }
            }
        }
        let mut extended = variables.cloned().unwrap_or_default();
        extended.insert("message_history".to_string(), json!("{{message_history}}"));
        extended.insert("response_to_evaluate".to_string(), json!("{{response_to_evaluate}}"));

        let config = self
            .evaluate(EvaluateRequest {
                key,
                context,
                default_value: LdAiConfigDefaultKind::Judge(default_value),
                mode: LdAiConfigMode::Judge,
                variables: Some(&extended),
                graph_key: None,
                default_ai_provider: None,
                interpolate: true,
            })
            .await;
        let mut config = expect_judge(config);
        config.messages = config.messages.map(|m| strip_legacy_judge_messages(&m));
        config
    }

    /// Retrieves a judge AI Config with its messages interpolated and legacy template
    /// messages stripped (JavaScript: `judgeConfig`). The variables `message_history`
    /// and `response_to_evaluate` are reserved and ignored with a warning.
    pub async fn judge_config(
        &self,
        key: &str,
        context: &Context,
        default_value: Option<LdAiJudgeConfigDefault>,
        variables: Option<&Variables>,
    ) -> LdAiJudgeConfig {
        self.track_usage(TRACK_USAGE_JUDGE_CONFIG, context, json!(key), 1.0);
        self.judge_config_inner(key, context, default_value.unwrap_or_default(), variables)
            .await
    }

    /// Retrieves a judge AI Config with Mustache placeholders left intact
    /// (JavaScript: `judgeConfigTemplate`).
    pub async fn judge_config_template(
        &self,
        key: &str,
        context: &Context,
        default_value: Option<LdAiJudgeConfigDefault>,
    ) -> LdAiJudgeConfig {
        self.track_usage(TRACK_USAGE_JUDGE_CONFIG_TEMPLATE, context, json!(key), 1.0);
        let config = self
            .evaluate(EvaluateRequest {
                key,
                context,
                default_value: LdAiConfigDefaultKind::Judge(default_value.unwrap_or_default()),
                mode: LdAiConfigMode::Judge,
                variables: None,
                graph_key: None,
                default_ai_provider: None,
                interpolate: false,
            })
            .await;
        expect_judge(config)
    }

    /// Retrieves several agent AI Configs concurrently, keyed by agent key in request
    /// order (JavaScript: `agentConfigs`).
    pub async fn agent_configs(
        &self,
        agent_configs: &[LdAiAgentRequestConfig],
        context: &Context,
    ) -> IndexMap<String, LdAiAgentConfig> {
        let count = agent_configs.len();
        self.track_usage(TRACK_USAGE_AGENT_CONFIGS, context, json!(count), count as f64);
        let configs = join_all(agent_configs.iter().map(|request| {
            self.agent_config_inner(
                &request.key,
                context,
                request.default_value.clone().unwrap_or_default(),
                request.variables.as_ref(),
                None,
                None,
            )
        }))
        .await;
        agent_configs
            .iter()
            .map(|r| r.key.clone())
            .zip(configs)
            .collect()
    }

    /// Creates a [`Judge`] for the judge AI Config `key` (JavaScript: `createJudge`).
    ///
    /// Returns `None` when the config is disabled or no registered provider can build
    /// a runner. `sample_rate` is the judge's default sampling rate (JavaScript
    /// default 1.0).
    pub async fn create_judge(
        &self,
        key: &str,
        context: &Context,
        default_value: Option<LdAiJudgeConfigDefault>,
        variables: Option<&Variables>,
        default_ai_provider: Option<&str>,
        sample_rate: f64,
    ) -> Option<Judge> {
        self.track_usage(TRACK_USAGE_CREATE_JUDGE, context, json!(key), 1.0);
        self.create_judge_instance(key, context, default_value, variables, default_ai_provider, sample_rate)
            .await
    }

    fn create_judge_instance<'a>(
        &'a self,
        key: &'a str,
        context: &'a Context,
        default_value: Option<LdAiJudgeConfigDefault>,
        variables: Option<&'a Variables>,
        default_ai_provider: Option<&'a str>,
        sample_rate: f64,
    ) -> BoxFuture<'a, Option<Judge>> {
        async move {
            let judge_config = self
                .judge_config_inner(key, context, default_value.unwrap_or_default(), variables)
                .await;
            if !judge_config.enabled {
                if let Some(logger) = &self.logger {
                    logger.info(&format!("Judge configuration is disabled: {key}"));
                }
                return None;
            }
            let runner = self
                .runner_factory
                .create_model(
                    ModelConfigRef::Judge(&judge_config),
                    self.logger_ref(),
                    default_ai_provider,
                    false,
                )
                .await?;
            Some(Judge::new(judge_config, runner, sample_rate, self.logger.clone()))
        }
        .boxed()
    }

    /// Creates a [`ManagedModel`] for the completion AI Config `key`
    /// (JavaScript: `createModel`). Returns `None` when the config is disabled or no
    /// registered provider can build a runner.
    pub async fn create_model(
        &self,
        key: &str,
        context: &Context,
        default_value: Option<LdAiCompletionConfigDefault>,
        variables: Option<&Variables>,
        default_ai_provider: Option<&str>,
    ) -> Option<ManagedModel> {
        self.track_usage(TRACK_USAGE_CREATE_CHAT, context, json!(key), 1.0);
        let config = self
            .completion_config_inner(
                key,
                context,
                default_value.unwrap_or_default(),
                variables,
                default_ai_provider,
            )
            .await;
        if !config.enabled {
            if let Some(logger) = &self.logger {
                logger.info(&format!("Completion configuration is disabled: {key}"));
            }
            return None;
        }
        let runner = self
            .runner_factory
            .create_model(
                ModelConfigRef::Completion(&config),
                self.logger_ref(),
                default_ai_provider,
                true,
            )
            .await?;
        Some(ManagedModel::new(config, runner, self.logger.clone()))
    }

    /// Creates a [`ManagedAgent`] for the agent AI Config `key`
    /// (JavaScript: `createAgent`). Returns `None` when the config is disabled or no
    /// registered provider can build a runner.
    pub async fn create_agent(
        &self,
        key: &str,
        context: &Context,
        default_value: Option<LdAiAgentConfigDefault>,
        variables: Option<&Variables>,
        default_ai_provider: Option<&str>,
    ) -> Option<ManagedAgent> {
        self.track_usage(TRACK_USAGE_CREATE_AGENT, context, json!(key), 1.0);
        let config = self
            .agent_config_inner(
                key,
                context,
                default_value.unwrap_or_default(),
                variables,
                None,
                default_ai_provider,
            )
            .await;
        if !config.enabled {
            if let Some(logger) = &self.logger {
                logger.info(&format!("Agent configuration is disabled: {key}"));
            }
            return None;
        }
        let runner = self
            .runner_factory
            .create_agent(&config, None, self.logger_ref(), default_ai_provider)
            .await?;
        Some(ManagedAgent::new(config, runner, self.logger.clone()))
    }

    /// Reconstructs a config tracker from a resumption token, preserving its `runId`
    /// (JavaScript: `createTracker`).
    pub fn create_tracker(&self, token: &str, context: &Context) -> Result<LdAiConfigTracker, Error> {
        LdAiConfigTracker::from_resumption_token(token, Arc::clone(&self.ld_client), context.clone())
    }

    /// Fetches an agent graph and its node configs (JavaScript: `agentGraph`).
    ///
    /// The result is disabled (no nodes) when the graph flag is explicitly disabled,
    /// has no root, has a node unreachable from the root, or references an agent
    /// config that is disabled or missing; each case logs at debug level. `variables`
    /// interpolate every node's instructions.
    pub async fn agent_graph(
        &self,
        graph_key: &str,
        context: &Context,
        variables: Option<&Variables>,
    ) -> AgentGraphDefinition {
        self.track_usage(TRACK_USAGE_AGENT_GRAPH, context, json!(graph_key), 1.0);

        let value = self
            .ld_client
            .variation(graph_key, context, json!({ "root": "" }))
            .await;
        let graph = LdAgentGraphFlagValue::from_value(&value);
        let meta = graph.ld_meta.clone().unwrap_or_default();

        let ld_client = Arc::clone(&self.ld_client);
        let tracker_context = context.clone();
        let tracker_graph_key = graph_key.to_string();
        let create_tracker = move || {
            LdGraphTracker::new(
                Arc::clone(&ld_client),
                uuid::Uuid::new_v4().to_string(),
                tracker_graph_key.clone(),
                meta.variation_key.clone(),
                meta.version.unwrap_or(1),
                tracker_context.clone(),
            )
        };
        let disabled = |graph: LdAgentGraphFlagValue| {
            AgentGraphDefinition::new(graph, IndexMap::new(), false, create_tracker.clone())
        };
        let debug = |message: String| {
            if let Some(logger) = &self.logger {
                logger.debug(&message);
            }
        };

        if graph.ld_meta.as_ref().and_then(|m| m.enabled) == Some(false) {
            debug(format!("agentGraph: graph \"{graph_key}\" is disabled."));
            return disabled(graph);
        }
        if graph.root.is_empty() {
            debug(format!(
                "agentGraph: graph \"{graph_key}\" is not fetchable or has no root node."
            ));
            return disabled(graph);
        }

        let all_keys = AgentGraphDefinition::collect_all_keys(&graph);
        let reachable = collect_reachable_keys(&graph);
        if let Some(unreachable) = all_keys.iter().find(|k| !reachable.contains(*k)) {
            debug(format!(
                "agentGraph: graph \"{graph_key}\" has unconnected node \"{unreachable}\" that is not reachable from the root."
            ));
            return disabled(graph);
        }

        let configs = join_all(all_keys.iter().map(|key| {
            self.agent_config_inner(
                key,
                context,
                LdAiAgentConfigDefault::default(),
                variables,
                Some(graph_key),
                None,
            )
        }))
        .await;
        let fetched: Vec<(String, LdAiAgentConfig)> = all_keys.iter().cloned().zip(configs).collect();

        if let Some((key, _)) = fetched.iter().find(|(_, config)| !config.enabled) {
            debug(format!(
                "agentGraph: agent config \"{key}\" in graph \"{graph_key}\" is not enabled or could not be fetched."
            ));
            return disabled(graph);
        }

        let agent_configs: HashMap<String, LdAiAgentConfig> = fetched.into_iter().collect();
        let nodes = AgentGraphDefinition::build_nodes(&graph, &agent_configs);
        AgentGraphDefinition::new(graph, nodes, true, create_tracker)
    }

    /// Reconstructs a graph tracker from a resumption token, preserving its `runId`
    /// (JavaScript: `createGraphTracker`).
    pub fn create_graph_tracker(&self, token: &str, context: &Context) -> Result<LdGraphTracker, Error> {
        LdGraphTracker::from_resumption_token(token, Arc::clone(&self.ld_client), context.clone())
    }
}

fn collect_reachable_keys(graph: &LdAgentGraphFlagValue) -> IndexSet<String> {
    let mut visited = IndexSet::from([graph.root.clone()]);
    let mut queue = VecDeque::from([graph.root.clone()]);
    while let Some(key) = queue.pop_front() {
        for edge in graph.edges_of(&key) {
            if visited.insert(edge.key.clone()) {
                queue.push_back(edge.key.clone());
            }
        }
    }
    visited
}

fn expect_completion(config: LdAiConfigKind) -> LdAiCompletionConfig {
    match config {
        LdAiConfigKind::Completion(c) => c,
        other => unreachable!("completion evaluation produced {:?}", other.mode()),
    }
}

fn expect_agent(config: LdAiConfigKind) -> LdAiAgentConfig {
    match config {
        LdAiConfigKind::Agent(c) => c,
        other => unreachable!("agent evaluation produced {:?}", other.mode()),
    }
}

fn expect_judge(config: LdAiConfigKind) -> LdAiJudgeConfig {
    match config {
        LdAiConfigKind::Judge(c) => c,
        other => unreachable!("judge evaluation produced {:?}", other.mode()),
    }
}
