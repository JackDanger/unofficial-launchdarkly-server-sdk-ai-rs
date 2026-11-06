//! A minimal [`AiProvider`] for any OpenAI-compatible chat-completions endpoint
//! (OpenAI, vLLM, llama.cpp server, Ollama's `/v1`, LiteLLM, …). Enabled by the
//! `openai-compatible` cargo feature.
//!
//! This is not a port of LaunchDarkly's `@launchdarkly/server-sdk-ai-openai`; it is a
//! small provider written for this crate so judges and managed models work against
//! self-hosted models. Structured output (used by judges) is requested with
//! `response_format: {"type": "json_schema", ...}`.
//!
//! ```no_run
//! # use std::sync::Arc;
//! use unofficial_launchdarkly_server_sdk_ai::openai_compatible::OpenAiCompatibleProvider;
//! use unofficial_launchdarkly_server_sdk_ai::RunnerFactory;
//!
//! let provider = OpenAiCompatibleProvider::new("http://localhost:8000/v1")
//!     .with_api_key(std::env::var("OPENAI_API_KEY").ok());
//! let factory = RunnerFactory::new().with_provider("openai", Arc::new(provider));
//! ```

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{json, Map, Value};

use crate::config::{LdAiAgentConfig, LdMessage};
use crate::error::BoxError;
use crate::metrics::{LdAiMetrics, LdTokenUsage};
use crate::model::RunnerResult;
use crate::providers::{AiProvider, ModelConfigRef, Runner, ToolRegistry};

/// The `json_schema.name` sent when a runner is asked for structured output.
pub const STRUCTURED_OUTPUT_NAME: &str = "structured_output";

/// Builds [`OpenAiCompatibleRunner`]s that call `{base_url}/chat/completions`.
#[derive(Debug, Clone)]
pub struct OpenAiCompatibleProvider {
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
}

impl OpenAiCompatibleProvider {
    /// A provider for the API rooted at `base_url` (e.g. `https://api.openai.com/v1`).
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: None,
        }
    }

    /// Sends `Authorization: Bearer <api_key>` when set.
    pub fn with_api_key(mut self, api_key: Option<String>) -> Self {
        self.api_key = api_key;
        self
    }

    /// Uses `http` (for timeouts, proxies, custom TLS) instead of a default client.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    fn runner(
        &self,
        model: Option<&crate::config::LdModelConfig>,
        initial_messages: Vec<LdMessage>,
        multi_turn: bool,
    ) -> Option<Arc<dyn Runner>> {
        let model = model.filter(|m| !m.name.is_empty())?;
        let parameters = model
            .parameters
            .clone()
            .unwrap_or_default()
            .into_iter()
            .filter(|(key, _)| key != "tools")
            .collect();
        Some(Arc::new(OpenAiCompatibleRunner {
            http: self.http.clone(),
            url: format!("{}/chat/completions", self.base_url),
            api_key: self.api_key.clone(),
            model: model.name.clone(),
            parameters,
            initial_messages,
            multi_turn,
            history: Mutex::new(Vec::new()),
        }))
    }
}

#[async_trait]
impl AiProvider for OpenAiCompatibleProvider {
    async fn create_model(
        &self,
        config: ModelConfigRef<'_>,
        multi_turn: bool,
    ) -> Option<Arc<dyn Runner>> {
        let messages = config
            .messages()
            .map(<[LdMessage]>::to_vec)
            .unwrap_or_default();
        self.runner(config.model(), messages, multi_turn)
    }

    /// Tools are not executed: the agent's instructions become the system message
    /// and the registry is ignored.
    async fn create_agent(
        &self,
        config: &LdAiAgentConfig,
        _tools: Option<&ToolRegistry>,
    ) -> Option<Arc<dyn Runner>> {
        let messages = config
            .instructions
            .iter()
            .filter(|i| !i.is_empty())
            .map(LdMessage::system)
            .collect();
        self.runner(config.model.as_ref(), messages, false)
    }
}

/// Calls a chat-completions endpoint for one AI Config.
///
/// Each `run` sends the config's messages, then (when multi-turn) the conversation so
/// far, then `input` as a user message. Model parameters are sent as top-level
/// request fields, except `tools`. Usage maps to [`LdTokenUsage`] and tool-call
/// names to [`LdAiMetrics::tool_calls`]. A transport failure or non-2xx status is
/// returned as an error.
pub struct OpenAiCompatibleRunner {
    http: reqwest::Client,
    url: String,
    api_key: Option<String>,
    model: String,
    parameters: Map<String, Value>,
    initial_messages: Vec<LdMessage>,
    multi_turn: bool,
    history: Mutex<Vec<LdMessage>>,
}

impl std::fmt::Debug for OpenAiCompatibleRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatibleRunner")
            .field("url", &self.url)
            .field("model", &self.model)
            .field("multi_turn", &self.multi_turn)
            .finish()
    }
}

impl OpenAiCompatibleRunner {
    fn history(&self) -> std::sync::MutexGuard<'_, Vec<LdMessage>> {
        self.history.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn request_body(&self, input: &str, output_type: Option<&Value>) -> Value {
        let mut messages = self.initial_messages.clone();
        if self.multi_turn {
            messages.extend(self.history().iter().cloned());
        }
        messages.push(LdMessage::user(input));

        let mut body = self.parameters.clone();
        body.insert("model".to_string(), json!(self.model));
        body.insert("messages".to_string(), json!(messages));
        if let Some(schema) = output_type {
            body.insert(
                "response_format".to_string(),
                json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": STRUCTURED_OUTPUT_NAME,
                        "schema": reasoning_first(schema),
                        "strict": true
                    }
                }),
            );
        }
        Value::Object(body)
    }
}

/// Moves every `reasoning` property ahead of its siblings, recursively.
///
/// Constrained decoders emit properties in schema order, so a judge schema that lists
/// `score` before `reasoning` makes the model commit to a number before it has reasoned,
/// and it then writes reasoning that contradicts its own score. The JavaScript SDK's
/// schema lists `score` first; reordering here keeps that schema while getting the
/// reasoning-then-score generation order on every endpoint.
pub fn reasoning_first(schema: &Value) -> Value {
    match schema {
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (key, value) in map {
                let value = match key.as_str() {
                    "properties" => match value {
                        Value::Object(props) => {
                            let mut ordered = serde_json::Map::new();
                            if let Some(reasoning) = props.get("reasoning") {
                                ordered.insert("reasoning".to_string(), reasoning_first(reasoning));
                            }
                            for (name, prop) in props {
                                if name != "reasoning" {
                                    ordered.insert(name.clone(), reasoning_first(prop));
                                }
                            }
                            Value::Object(ordered)
                        }
                        other => reasoning_first(other),
                    },
                    "required" => match value {
                        Value::Array(names) => {
                            let (first, rest): (Vec<_>, Vec<_>) =
                                names.iter().cloned().partition(|n| n == "reasoning");
                            Value::Array(first.into_iter().chain(rest).collect())
                        }
                        other => other.clone(),
                    },
                    _ => reasoning_first(value),
                };
                out.insert(key.clone(), value);
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(reasoning_first).collect()),
        other => other.clone(),
    }
}

fn usage(response: &Value) -> Option<LdTokenUsage> {
    let usage = response.get("usage")?;
    let count = |key: &str| usage.get(key).and_then(Value::as_u64).unwrap_or(0);
    let input = count("prompt_tokens");
    let output = count("completion_tokens");
    let total = usage
        .get("total_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(input + output);
    Some(LdTokenUsage {
        total,
        input,
        output,
    })
}

fn tool_calls(message: &Value) -> Option<Vec<String>> {
    let names: Vec<String> = message
        .get("tool_calls")?
        .as_array()?
        .iter()
        .filter_map(|call| call.pointer("/function/name").and_then(Value::as_str))
        .map(str::to_string)
        .collect();
    (!names.is_empty()).then_some(names)
}

#[async_trait]
impl Runner for OpenAiCompatibleRunner {
    async fn run(
        &self,
        input: &str,
        output_type: Option<&Value>,
    ) -> Result<RunnerResult, BoxError> {
        let mut request = self
            .http
            .post(&self.url)
            .json(&self.request_body(input, output_type));
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let response = request.send().await?;
        let status = response.status();
        let text = response.text().await?;
        if !status.is_success() {
            return Err(
                format!("chat completion request failed with status {status}: {text}").into(),
            );
        }
        let raw: Value = serde_json::from_str(&text)?;
        let message = raw
            .pointer("/choices/0/message")
            .cloned()
            .unwrap_or(Value::Null);
        let content = message
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();

        if self.multi_turn {
            let mut history = self.history();
            history.push(LdMessage::user(input));
            history.push(LdMessage::assistant(content.clone()));
        }

        let parsed = output_type.and_then(|_| {
            serde_json::from_str::<Value>(&content)
                .ok()
                .and_then(|v| v.as_object().cloned())
        });
        let metrics = LdAiMetrics {
            success: true,
            tokens: usage(&raw),
            tool_calls: tool_calls(&message),
            duration_ms: None,
        };
        Ok(RunnerResult {
            content,
            metrics,
            raw: Some(raw),
            parsed,
        })
    }
}
