//! Provider plumbing: runners, provider factories and the registry that picks one
//! (`api/providers` in JavaScript).
//!
//! The JavaScript `RunnerFactory` dynamically imports LaunchDarkly's provider packages
//! (`@launchdarkly/server-sdk-ai-openai`, `-langchain`, `-vercel`). Rust cannot load
//! packages at runtime, so here [`RunnerFactory`] is a registry: you register
//! [`AiProvider`] implementations under provider names, and the same selection rules
//! decide which ones to try.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
use serde_json::Value;

use crate::client_min::LdLogger;
use crate::config::{
    LdAiAgentConfig, LdAiCompletionConfig, LdAiJudgeConfig, LdMessage, LdModelConfig,
    LdProviderConfig,
};
use crate::error::BoxError;
use crate::graph::{AgentGraphDefinition, AgentGraphRunnerResult};
use crate::model::RunnerResult;

/// Invokes a model (JavaScript: `Runner`). One protocol covers completion, agent and
/// judge use; judges pass a JSON schema in `output_type` and read
/// [`RunnerResult::parsed`].
#[async_trait]
pub trait Runner: Send + Sync {
    /// Invokes the model with `input`, optionally requesting structured output that
    /// matches the JSON schema `output_type`.
    async fn run(&self, input: &str, output_type: Option<&Value>) -> Result<RunnerResult, BoxError>;
}

/// Executes a whole agent graph (JavaScript: `AgentGraphRunner`).
#[async_trait]
pub trait AgentGraphRunner: Send + Sync {
    /// Runs the graph on `input`.
    async fn run(&self, input: &str) -> Result<AgentGraphRunnerResult, BoxError>;
}

/// A callable tool the provider invokes when the model requests it.
pub type ToolFn = Arc<dyn Fn(Value) -> BoxFuture<'static, Result<Value, BoxError>> + Send + Sync>;

/// Callable tools keyed by tool name (JavaScript: `ToolRegistry`).
pub type ToolRegistry = HashMap<String, ToolFn>;

/// The config a model runner is built from: completion or judge
/// (JavaScript: `LDAICompletionConfig | LDAIJudgeConfig`).
#[derive(Debug, Clone, Copy)]
pub enum ModelConfigRef<'a> {
    /// A completion config.
    Completion(&'a LdAiCompletionConfig),
    /// A judge config.
    Judge(&'a LdAiJudgeConfig),
}

impl<'a> ModelConfigRef<'a> {
    /// The config key.
    pub fn key(&self) -> &'a str {
        match self {
            Self::Completion(c) => &c.key,
            Self::Judge(c) => &c.key,
        }
    }

    /// The model configuration.
    pub fn model(&self) -> Option<&'a LdModelConfig> {
        match self {
            Self::Completion(c) => c.model.as_ref(),
            Self::Judge(c) => c.model.as_ref(),
        }
    }

    /// The provider configuration.
    pub fn provider(&self) -> Option<&'a LdProviderConfig> {
        match self {
            Self::Completion(c) => c.provider.as_ref(),
            Self::Judge(c) => c.provider.as_ref(),
        }
    }

    /// The configured messages.
    pub fn messages(&self) -> Option<&'a [LdMessage]> {
        match self {
            Self::Completion(c) => c.messages.as_deref(),
            Self::Judge(c) => c.messages.as_deref(),
        }
    }
}

/// A provider factory (JavaScript: abstract class `AIProvider`). Implement the modes
/// your provider supports; the defaults return `None`, meaning "not supported".
#[async_trait]
pub trait AiProvider: Send + Sync {
    /// Creates a runner for a completion or judge config. `multi_turn` asks the runner
    /// to accumulate conversation history across `run` calls (judges pass `false`).
    async fn create_model(
        &self,
        _config: ModelConfigRef<'_>,
        _multi_turn: bool,
    ) -> Option<Arc<dyn Runner>> {
        None
    }

    /// Creates a runner for an agent config.
    async fn create_agent(
        &self,
        _config: &LdAiAgentConfig,
        _tools: Option<&ToolRegistry>,
    ) -> Option<Arc<dyn Runner>> {
        None
    }

    /// Creates a runner for a whole agent graph.
    async fn create_agent_graph(
        &self,
        _graph_def: &AgentGraphDefinition,
        _tools: Option<&ToolRegistry>,
    ) -> Option<Arc<dyn AgentGraphRunner>> {
        None
    }
}

#[derive(Clone)]
struct RegisteredProvider {
    name: String,
    multi_provider: bool,
    provider: Arc<dyn AiProvider>,
}

/// Picks and invokes [`AiProvider`]s (JavaScript: `RunnerFactory`).
///
/// Selection mirrors JavaScript: when a default provider name is given, only that
/// provider is tried; otherwise the provider named in the config (if registered) is
/// tried first, then every multi-provider in registration order. Names compare
/// case-insensitively.
#[derive(Clone, Default)]
pub struct RunnerFactory {
    providers: Vec<RegisteredProvider>,
}

impl fmt::Debug for RunnerFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.providers.iter().map(|p| (&p.name, p.multi_provider)))
            .finish()
    }
}

impl RunnerFactory {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a provider that handles configs whose `provider.name` is `name`
    /// (like JavaScript's `openai` package).
    pub fn with_provider(mut self, name: impl Into<String>, provider: Arc<dyn AiProvider>) -> Self {
        self.register(name, provider, false);
        self
    }

    /// Registers a multi-provider fallback that is tried for any config (like
    /// JavaScript's `langchain` and `vercel` packages).
    pub fn with_multi_provider(
        mut self,
        name: impl Into<String>,
        provider: Arc<dyn AiProvider>,
    ) -> Self {
        self.register(name, provider, true);
        self
    }

    /// Registers `provider` under `name`, replacing any provider with that name.
    pub fn register(&mut self, name: impl Into<String>, provider: Arc<dyn AiProvider>, multi_provider: bool) {
        let name = name.into().to_lowercase();
        self.providers.retain(|p| p.name != name);
        self.providers.push(RegisteredProvider {
            name,
            multi_provider,
            provider,
        });
    }

    /// The registered provider names, in registration order.
    pub fn provider_names(&self) -> Vec<&str> {
        self.providers.iter().map(|p| p.name.as_str()).collect()
    }

    fn lookup(&self, name: &str) -> Option<&RegisteredProvider> {
        let name = name.to_lowercase();
        self.providers.iter().find(|p| p.name == name)
    }

    /// The provider names to try, in order (JavaScript: `_getProvidersToTry`).
    pub fn providers_to_try(&self, default_ai_provider: Option<&str>, provider_name: Option<&str>) -> Vec<String> {
        if let Some(default) = default_ai_provider {
            return vec![default.to_lowercase()];
        }
        let mut names: Vec<String> = Vec::new();
        if let Some(found) = provider_name.and_then(|name| self.lookup(name)) {
            names.push(found.name.clone());
        }
        for provider in self.providers.iter().filter(|p| p.multi_provider) {
            if !names.contains(&provider.name) {
                names.push(provider.name.clone());
            }
        }
        names
    }

    async fn with_fallback<'a, T, F>(
        &'a self,
        names: &[String],
        logger: Option<&dyn LdLogger>,
        create: F,
    ) -> Option<T>
    where
        F: Fn(&'a dyn AiProvider) -> BoxFuture<'a, Option<T>>,
    {
        for name in names {
            if let Some(logger) = logger {
                logger.debug(&format!("Attempting to create runner with provider: {name}"));
            }
            let Some(registered) = self.lookup(name) else {
                if let Some(logger) = logger {
                    logger.warn(&format!(
                        "Unable to load provider package. Check that you have installed the correct package. No AI provider is registered under the name \"{name}\"."
                    ));
                }
                continue;
            };
            if let Some(result) = create(registered.provider.as_ref()).await {
                if let Some(logger) = logger {
                    logger.debug(&format!("Successfully created runner with provider: {name}"));
                }
                return Some(result);
            }
        }
        None
    }

    /// Creates a runner for a completion or judge config.
    pub async fn create_model(
        &self,
        config: ModelConfigRef<'_>,
        logger: Option<&dyn LdLogger>,
        default_ai_provider: Option<&str>,
        multi_turn: bool,
    ) -> Option<Arc<dyn Runner>> {
        let provider_name = config.provider().map(|p| p.name.to_lowercase());
        let names = self.providers_to_try(default_ai_provider, provider_name.as_deref());
        let runner = self
            .with_fallback(&names, logger, |provider| provider.create_model(config, multi_turn))
            .await;
        if runner.is_none() {
            warn_unsupported(logger, config.provider());
        }
        runner
    }

    /// Creates a runner for an agent config.
    pub async fn create_agent(
        &self,
        config: &LdAiAgentConfig,
        tools: Option<&ToolRegistry>,
        logger: Option<&dyn LdLogger>,
        default_ai_provider: Option<&str>,
    ) -> Option<Arc<dyn Runner>> {
        let provider_name = config.provider.as_ref().map(|p| p.name.to_lowercase());
        let names = self.providers_to_try(default_ai_provider, provider_name.as_deref());
        let runner = self
            .with_fallback(&names, logger, |provider| provider.create_agent(config, tools))
            .await;
        if runner.is_none() {
            warn_unsupported(logger, config.provider.as_ref());
        }
        runner
    }

    /// Creates a runner for an agent graph. No per-node provider applies, so only the
    /// default provider or the multi-providers are tried.
    pub async fn create_agent_graph(
        &self,
        graph_def: &AgentGraphDefinition,
        tools: Option<&ToolRegistry>,
        logger: Option<&dyn LdLogger>,
        default_ai_provider: Option<&str>,
    ) -> Option<Arc<dyn AgentGraphRunner>> {
        let names = self.providers_to_try(default_ai_provider, None);
        let runner = self
            .with_fallback(&names, logger, |provider| provider.create_agent_graph(graph_def, tools))
            .await;
        if runner.is_none() {
            if let Some(logger) = logger {
                logger.warn("No provider could create an AgentGraphRunner for the given graph definition.");
            }
        }
        runner
    }
}

fn warn_unsupported(logger: Option<&dyn LdLogger>, provider: Option<&LdProviderConfig>) {
    if let Some(logger) = logger {
        logger.warn(&format!(
            "Provider is not supported or failed to initialize: {}",
            provider.map(|p| p.name.as_str()).unwrap_or("unknown")
        ));
    }
}
