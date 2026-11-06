use std::fmt;
use std::sync::Arc;

use indexmap::IndexMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use crate::judge::Evaluator;
use crate::tracker::LdAiConfigTracker;

/// Template variables for Mustache interpolation (JavaScript: `Record<string, unknown>`).
///
/// `serde_json::json!({...}).as_object()` yields an `Option<&Variables>` directly.
pub type Variables = Map<String, Value>;

/// The role of a message (JavaScript: `'user' | 'assistant' | 'system'`).
///
/// LaunchDarkly only produces the three named roles; [`LdMessageRole::Other`] keeps
/// any other string a flag variation carries, as the untyped JavaScript object would.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LdMessageRole {
    /// `system`
    System,
    /// `user`
    User,
    /// `assistant`
    Assistant,
    /// Any other role string.
    Other(String),
}

impl LdMessageRole {
    /// The role as it appears on the wire.
    pub fn as_str(&self) -> &str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Other(role) => role,
        }
    }
}

impl fmt::Display for LdMessageRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for LdMessageRole {
    fn from(role: &str) -> Self {
        match role {
            "system" => Self::System,
            "user" => Self::User,
            "assistant" => Self::Assistant,
            other => Self::Other(other.to_string()),
        }
    }
}

impl Serialize for LdMessageRole {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for LdMessageRole {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let role = String::deserialize(deserializer)?;
        Ok(Self::from(role.as_str()))
    }
}

/// A prompt message (JavaScript: `LDMessage`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LdMessage {
    /// The role of the message.
    pub role: LdMessageRole,
    /// The message content.
    pub content: String,
}

impl LdMessage {
    /// Creates a message.
    pub fn new(role: impl Into<LdMessageRole>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    /// A `system` message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(LdMessageRole::System, content)
    }

    /// A `user` message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(LdMessageRole::User, content)
    }

    /// An `assistant` message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(LdMessageRole::Assistant, content)
    }
}

/// Model configuration (JavaScript: `LDModelConfig`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LdModelConfig {
    /// The ID of the model.
    #[serde(default)]
    pub name: String,
    /// Model-specific parameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Map<String, Value>>,
    /// Additional user-specified parameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom: Option<Map<String, Value>>,
    /// The region for the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
}

impl LdModelConfig {
    /// A model configuration with only a name.
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }
}

/// Provider configuration (JavaScript: `LDProviderConfig`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LdProviderConfig {
    /// The name of the provider.
    #[serde(default)]
    pub name: String,
}

impl LdProviderConfig {
    /// A provider configuration with the given name.
    pub fn named(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// One entry of the root-level tools map (JavaScript: `LDTool`).
///
/// Distinct from `model.parameters.tools[]`, which is the array passed to the LLM.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LdTool {
    /// The tool name.
    #[serde(default)]
    pub name: String,
    /// A description of the tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The tool type.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,
    /// The tool's parameter schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Map<String, Value>>,
    /// Additional user-specified parameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_parameters: Option<Map<String, Value>>,
}

/// The root-level tools map, keyed by tool name, in flag order.
pub type LdTools = IndexMap<String, LdTool>;

/// A single judge attachment (JavaScript: `LDJudge`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LdJudge {
    /// The key of the judge AI Config.
    pub key: String,
    /// Sampling rate for evaluation (0.0 to 1.0).
    pub sampling_rate: f64,
}

/// Judge attachments for an AI Config (JavaScript: `LDJudgeConfiguration`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LdJudgeConfiguration {
    /// The attached judges.
    #[serde(default)]
    pub judges: Vec<LdJudge>,
}

/// The kind of AI Config (JavaScript: `LDAIConfigMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LdAiConfigMode {
    /// A completion (chat) config with messages.
    Completion,
    /// An agent config with instructions.
    Agent,
    /// A judge config with an evaluation metric key.
    Judge,
}

impl LdAiConfigMode {
    /// The mode as it appears in `_ldMeta.mode`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Completion => "completion",
            Self::Agent => "agent",
            Self::Judge => "judge",
        }
    }
}

impl fmt::Display for LdAiConfigMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Default for a completion AI Config (JavaScript: `LDAICompletionConfigDefault`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LdAiCompletionConfigDefault {
    /// Whether the configuration is enabled; `None` means disabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Model configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<LdModelConfig>,
    /// Provider configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<LdProviderConfig>,
    /// Prompt messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<LdMessage>>,
    /// Judges that evaluate this config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_configuration: Option<LdJudgeConfiguration>,
    /// Root-level tools map.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<LdTools>,
}

/// Default for an agent AI Config (JavaScript: `LDAIAgentConfigDefault`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LdAiAgentConfigDefault {
    /// Whether the configuration is enabled; `None` means disabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Model configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<LdModelConfig>,
    /// Provider configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<LdProviderConfig>,
    /// Instructions for the agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Judges that evaluate this config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_configuration: Option<LdJudgeConfiguration>,
    /// Root-level tools map.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<LdTools>,
}

/// Default for a judge AI Config (JavaScript: `LDAIJudgeConfigDefault`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LdAiJudgeConfigDefault {
    /// Whether the configuration is enabled; `None` means disabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Model configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<LdModelConfig>,
    /// Provider configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<LdProviderConfig>,
    /// Judge prompt messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<LdMessage>>,
    /// The key of the metric this judge evaluates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation_metric_key: Option<String>,
}

/// Any of the three default kinds (JavaScript: `LDAIConfigDefaultKind`).
#[derive(Debug, Clone, PartialEq)]
pub enum LdAiConfigDefaultKind {
    /// A completion default.
    Completion(LdAiCompletionConfigDefault),
    /// An agent default.
    Agent(LdAiAgentConfigDefault),
    /// A judge default.
    Judge(LdAiJudgeConfigDefault),
}

/// Creates a fresh [`LdAiConfigTracker`] for each AI run (JavaScript: `createTracker`).
#[derive(Clone)]
pub struct TrackerFactory(Arc<dyn Fn() -> LdAiConfigTracker + Send + Sync>);

impl TrackerFactory {
    /// Wraps a closure that mints trackers.
    pub fn new(factory: impl Fn() -> LdAiConfigTracker + Send + Sync + 'static) -> Self {
        Self(Arc::new(factory))
    }

    /// Mints a tracker.
    pub fn create(&self) -> LdAiConfigTracker {
        (self.0)()
    }
}

impl fmt::Debug for TrackerFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TrackerFactory")
    }
}

/// An evaluated completion AI Config (JavaScript: `LDAICompletionConfig`).
#[derive(Debug, Clone)]
pub struct LdAiCompletionConfig {
    /// The key of the AI Config.
    pub key: String,
    /// Whether the configuration is enabled.
    pub enabled: bool,
    /// Model configuration.
    pub model: Option<LdModelConfig>,
    /// Provider configuration.
    pub provider: Option<LdProviderConfig>,
    /// Prompt messages, interpolated unless fetched as a template.
    pub messages: Option<Vec<LdMessage>>,
    /// Judges that evaluate this config.
    pub judge_configuration: Option<LdJudgeConfiguration>,
    /// Root-level tools map.
    pub tools: Option<LdTools>,
    /// Mints the tracker for each run; see [`Self::create_tracker`].
    pub tracker_factory: TrackerFactory,
    /// The judges attached to this config, built at evaluation time.
    pub evaluator: Evaluator,
}

impl LdAiCompletionConfig {
    /// Creates a new tracker for a fresh AI run, with a new `runId`.
    ///
    /// Call this once per AI run; metrics from different `runId`s cannot be combined.
    pub fn create_tracker(&self) -> LdAiConfigTracker {
        self.tracker_factory.create()
    }
}

/// An evaluated agent AI Config (JavaScript: `LDAIAgentConfig`).
#[derive(Debug, Clone)]
pub struct LdAiAgentConfig {
    /// The key of the AI Config.
    pub key: String,
    /// Whether the configuration is enabled.
    pub enabled: bool,
    /// Model configuration.
    pub model: Option<LdModelConfig>,
    /// Provider configuration.
    pub provider: Option<LdProviderConfig>,
    /// Agent instructions, interpolated unless fetched as a template.
    pub instructions: Option<String>,
    /// Judges that evaluate this config.
    pub judge_configuration: Option<LdJudgeConfiguration>,
    /// Root-level tools map.
    pub tools: Option<LdTools>,
    /// Mints the tracker for each run; see [`Self::create_tracker`].
    pub tracker_factory: TrackerFactory,
    /// The judges attached to this config, built at evaluation time.
    pub evaluator: Evaluator,
}

impl LdAiAgentConfig {
    /// Creates a new tracker for a fresh AI run, with a new `runId`.
    pub fn create_tracker(&self) -> LdAiConfigTracker {
        self.tracker_factory.create()
    }
}

/// An evaluated judge AI Config (JavaScript: `LDAIJudgeConfig`).
#[derive(Debug, Clone)]
pub struct LdAiJudgeConfig {
    /// The key of the AI Config.
    pub key: String,
    /// Whether the configuration is enabled.
    pub enabled: bool,
    /// Model configuration.
    pub model: Option<LdModelConfig>,
    /// Provider configuration.
    pub provider: Option<LdProviderConfig>,
    /// Judge prompt messages, with legacy template messages stripped.
    pub messages: Option<Vec<LdMessage>>,
    /// The key of the metric this judge evaluates.
    pub evaluation_metric_key: Option<String>,
    /// Mints the tracker for each run; see [`Self::create_tracker`].
    pub tracker_factory: TrackerFactory,
}

impl LdAiJudgeConfig {
    /// Creates a new tracker for a fresh AI run, with a new `runId`.
    pub fn create_tracker(&self) -> LdAiConfigTracker {
        self.tracker_factory.create()
    }
}

/// Any evaluated AI Config (JavaScript: `LDAIConfigKind`).
#[derive(Debug, Clone)]
pub enum LdAiConfigKind {
    /// A completion config.
    Completion(LdAiCompletionConfig),
    /// An agent config.
    Agent(LdAiAgentConfig),
    /// A judge config.
    Judge(LdAiJudgeConfig),
}

impl LdAiConfigKind {
    /// The config key.
    pub fn key(&self) -> &str {
        match self {
            Self::Completion(c) => &c.key,
            Self::Agent(c) => &c.key,
            Self::Judge(c) => &c.key,
        }
    }

    /// Whether the config is enabled.
    pub fn enabled(&self) -> bool {
        match self {
            Self::Completion(c) => c.enabled,
            Self::Agent(c) => c.enabled,
            Self::Judge(c) => c.enabled,
        }
    }

    /// The mode of this config.
    pub fn mode(&self) -> LdAiConfigMode {
        match self {
            Self::Completion(_) => LdAiConfigMode::Completion,
            Self::Agent(_) => LdAiConfigMode::Agent,
            Self::Judge(_) => LdAiConfigMode::Judge,
        }
    }

    /// Creates a new tracker for a fresh AI run.
    pub fn create_tracker(&self) -> LdAiConfigTracker {
        match self {
            Self::Completion(c) => c.create_tracker(),
            Self::Agent(c) => c.create_tracker(),
            Self::Judge(c) => c.create_tracker(),
        }
    }
}

/// One request in [`crate::LdAiClient::agent_configs`] (JavaScript: `LDAIAgentRequestConfig`).
#[derive(Debug, Clone, Default)]
pub struct LdAiAgentRequestConfig {
    /// The agent key to retrieve.
    pub key: String,
    /// Default for the agent; `None` uses a disabled default.
    pub default_value: Option<LdAiAgentConfigDefault>,
    /// Variables for instructions interpolation.
    pub variables: Option<Variables>,
}

impl LdAiAgentRequestConfig {
    /// A request for `key` with a disabled default and no variables.
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            ..Self::default()
        }
    }
}
