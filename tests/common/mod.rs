//! Test doubles shared by the ported JavaScript suites: a fake LaunchDarkly client that
//! records `track` calls and log lines, and scriptable runners and providers.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;
use unofficial_launchdarkly_server_sdk_ai::{
    AgentGraphDefinition, AgentGraphRunner, AiProvider, BoxError, Context, ContextBuilder,
    LdAiAgentConfig, LdAiMetrics, LdClientMin, LdLogger, ModelConfigRef, Runner, RunnerResult,
    ToolRegistry,
};

pub const TEST_RUN_ID: &str = "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11";

/// `{ kind: 'user', key: 'test-user' }`
pub fn test_context() -> Context {
    ContextBuilder::new("test-user").build().unwrap()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Error,
    Warn,
    Info,
    Debug,
}

#[derive(Default)]
pub struct RecordingLogger {
    pub lines: Mutex<Vec<(Level, String)>>,
}

impl RecordingLogger {
    pub fn at(&self, level: Level) -> Vec<String> {
        self.lines
            .lock()
            .unwrap()
            .iter()
            .filter(|(l, _)| *l == level)
            .map(|(_, m)| m.clone())
            .collect()
    }
    pub fn warnings(&self) -> Vec<String> {
        self.at(Level::Warn)
    }
    pub fn debugs(&self) -> Vec<String> {
        self.at(Level::Debug)
    }
    pub fn infos(&self) -> Vec<String> {
        self.at(Level::Info)
    }
    pub fn errors(&self) -> Vec<String> {
        self.at(Level::Error)
    }
    pub fn clear(&self) {
        self.lines.lock().unwrap().clear();
    }
    fn push(&self, level: Level, message: &str) {
        self.lines
            .lock()
            .unwrap()
            .push((level, message.to_string()));
    }
}

impl LdLogger for RecordingLogger {
    fn error(&self, message: &str) {
        self.push(Level::Error, message)
    }
    fn warn(&self, message: &str) {
        self.push(Level::Warn, message)
    }
    fn info(&self, message: &str) {
        self.push(Level::Info, message)
    }
    fn debug(&self, message: &str) {
        self.push(Level::Debug, message)
    }
}

/// One recorded `track(key, context, data, metricValue)` call.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackCall {
    pub key: String,
    pub context: Context,
    pub data: Option<Value>,
    pub metric_value: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VariationCall {
    pub key: String,
    pub context: Context,
    pub default_value: Value,
}

/// A fake `LDClientMin`: flags set with [`FakeLdClient::set_flag`] are returned from
/// `variation`; any other key returns the default, as the real SDK does for an
/// unknown flag.
pub struct FakeLdClient {
    flags: Mutex<HashMap<String, Value>>,
    pub variation_calls: Mutex<Vec<VariationCall>>,
    pub track_calls: Mutex<Vec<TrackCall>>,
    pub logger: Arc<RecordingLogger>,
    has_logger: bool,
}

impl FakeLdClient {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::build(true))
    }

    pub fn without_logger() -> Arc<Self> {
        Arc::new(Self::build(false))
    }

    fn build(has_logger: bool) -> Self {
        Self {
            flags: Mutex::new(HashMap::new()),
            variation_calls: Mutex::new(Vec::new()),
            track_calls: Mutex::new(Vec::new()),
            logger: Arc::new(RecordingLogger::default()),
            has_logger,
        }
    }

    pub fn set_flag(&self, key: &str, value: Value) {
        self.flags.lock().unwrap().insert(key.to_string(), value);
    }

    pub fn tracks(&self) -> Vec<TrackCall> {
        self.track_calls.lock().unwrap().clone()
    }

    pub fn tracks_for(&self, key: &str) -> Vec<TrackCall> {
        self.tracks().into_iter().filter(|t| t.key == key).collect()
    }

    pub fn track_keys(&self) -> Vec<String> {
        self.tracks().into_iter().map(|t| t.key).collect()
    }

    pub fn variations(&self) -> Vec<VariationCall> {
        self.variation_calls.lock().unwrap().clone()
    }

    /// Forgets recorded calls and log lines (like `jest.clearAllMocks()`), keeping flags.
    pub fn clear(&self) {
        self.track_calls.lock().unwrap().clear();
        self.variation_calls.lock().unwrap().clear();
        self.logger.clear();
    }

    /// The client as the trait object the SDK stores.
    pub fn dyn_client(self: &Arc<Self>) -> Arc<dyn LdClientMin> {
        Arc::clone(self) as Arc<dyn LdClientMin>
    }
}

#[async_trait]
impl LdClientMin for FakeLdClient {
    async fn variation(&self, key: &str, context: &Context, default_value: Value) -> Value {
        self.variation_calls.lock().unwrap().push(VariationCall {
            key: key.to_string(),
            context: context.clone(),
            default_value: default_value.clone(),
        });
        self.flags
            .lock()
            .unwrap()
            .get(key)
            .cloned()
            .unwrap_or(default_value)
    }

    fn track(&self, key: &str, context: &Context, data: Option<Value>, metric_value: Option<f64>) {
        self.track_calls.lock().unwrap().push(TrackCall {
            key: key.to_string(),
            context: context.clone(),
            data,
            metric_value,
        });
    }

    fn logger(&self) -> Option<Arc<dyn LdLogger>> {
        self.has_logger
            .then(|| Arc::clone(&self.logger) as Arc<dyn LdLogger>)
    }
}

type RunnerScript = dyn Fn(&str, Option<&Value>) -> Result<RunnerResult, BoxError> + Send + Sync;

/// A runner that records each `(input, output_type)` and answers from a script.
pub struct FakeRunner {
    pub calls: Mutex<Vec<(String, Option<Value>)>>,
    script: Box<RunnerScript>,
}

impl FakeRunner {
    pub fn new(
        script: impl Fn(&str, Option<&Value>) -> Result<RunnerResult, BoxError> + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(Self {
            calls: Mutex::new(Vec::new()),
            script: Box::new(script),
        })
    }

    /// Always succeeds with `content` and `metrics`.
    pub fn returning(content: &str, metrics: LdAiMetrics) -> Arc<Self> {
        let content = content.to_string();
        Self::new(move |_, _| {
            Ok(RunnerResult {
                content: content.clone(),
                metrics: metrics.clone(),
                ..RunnerResult::default()
            })
        })
    }

    /// Always succeeds with structured output `parsed`.
    pub fn parsed(parsed: Value, success: bool) -> Arc<Self> {
        Self::new(move |_, _| {
            Ok(RunnerResult {
                content: parsed.to_string(),
                metrics: LdAiMetrics::new(success),
                raw: Some(parsed.clone()),
                parsed: parsed.as_object().cloned(),
            })
        })
    }

    /// Always fails with `message`.
    pub fn failing(message: &str) -> Arc<Self> {
        let message = message.to_string();
        Self::new(move |_, _| Err(message.clone().into()))
    }

    pub fn inputs(&self) -> Vec<String> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .map(|(i, _)| i.clone())
            .collect()
    }
}

#[async_trait]
impl Runner for FakeRunner {
    async fn run(
        &self,
        input: &str,
        output_type: Option<&Value>,
    ) -> Result<RunnerResult, BoxError> {
        self.calls
            .lock()
            .unwrap()
            .push((input.to_string(), output_type.cloned()));
        (self.script)(input, output_type)
    }
}

/// A provider that hands out a fixed runner (or none) and records what it was asked for.
#[derive(Default)]
pub struct FakeProvider {
    pub model_runner: Option<Arc<dyn Runner>>,
    pub agent_runner: Option<Arc<dyn Runner>>,
    pub graph_runner: Option<Arc<dyn AgentGraphRunner>>,
    pub model_calls: Mutex<Vec<(String, bool)>>,
    pub agent_calls: Mutex<Vec<(String, bool)>>,
    pub graph_calls: Mutex<Vec<bool>>,
}

impl FakeProvider {
    pub fn with_model(runner: Arc<dyn Runner>) -> Arc<Self> {
        Arc::new(Self {
            model_runner: Some(runner),
            ..Self::default()
        })
    }

    pub fn with_agent(runner: Arc<dyn Runner>) -> Arc<Self> {
        Arc::new(Self {
            agent_runner: Some(runner),
            ..Self::default()
        })
    }

    pub fn empty() -> Arc<Self> {
        Arc::new(Self::default())
    }
}

#[async_trait]
impl AiProvider for FakeProvider {
    async fn create_model(
        &self,
        config: ModelConfigRef<'_>,
        multi_turn: bool,
    ) -> Option<Arc<dyn Runner>> {
        self.model_calls
            .lock()
            .unwrap()
            .push((config.key().to_string(), multi_turn));
        self.model_runner.clone()
    }

    async fn create_agent(
        &self,
        config: &LdAiAgentConfig,
        tools: Option<&ToolRegistry>,
    ) -> Option<Arc<dyn Runner>> {
        self.agent_calls
            .lock()
            .unwrap()
            .push((config.key.clone(), tools.is_some()));
        self.agent_runner.clone()
    }

    async fn create_agent_graph(
        &self,
        _graph_def: &AgentGraphDefinition,
        tools: Option<&ToolRegistry>,
    ) -> Option<Arc<dyn AgentGraphRunner>> {
        self.graph_calls.lock().unwrap().push(tools.is_some());
        self.graph_runner.clone()
    }
}
