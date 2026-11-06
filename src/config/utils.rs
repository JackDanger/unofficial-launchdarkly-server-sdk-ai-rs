use serde::de::DeserializeOwned;
use serde_json::{json, Map, Value};

use super::types::*;
use crate::judge::Evaluator;

#[derive(Debug, Clone, Default)]
pub(crate) struct FlagMeta {
    pub variation_key: Option<String>,
    pub enabled: bool,
    pub version: Option<u64>,
    pub mode: Option<String>,
    pub model_key: Option<String>,
    pub model_version: Option<u64>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct FlagValue {
    pub meta: Option<FlagMeta>,
    pub model: Option<LdModelConfig>,
    pub messages: Option<Vec<LdMessage>>,
    pub provider: Option<LdProviderConfig>,
    pub instructions: Option<String>,
    pub evaluation_metric_key: Option<String>,
    pub evaluation_metric_keys: Option<Vec<Value>>,
    pub judge_configuration: Option<LdJudgeConfiguration>,
    pub tools: Option<LdTools>,
}

pub(crate) fn lenient<T: DeserializeOwned>(object: &Map<String, Value>, key: &str) -> Option<T> {
    object
        .get(key)
        .filter(|v| !v.is_null())
        .and_then(|v| serde_json::from_value(v.clone()).ok())
}

pub(crate) fn as_version(value: Option<&Value>) -> Option<u64> {
    let value = value?;
    value
        .as_u64()
        .or_else(|| value.as_f64().filter(|f| *f >= 0.0).map(|f| f as u64))
}

fn lenient_messages(object: &Map<String, Value>) -> Option<Vec<LdMessage>> {
    let items = object.get("messages")?.as_array()?;
    Some(
        items
            .iter()
            .filter_map(|item| serde_json::from_value(item.clone()).ok())
            .collect(),
    )
}

fn lenient_tools(value: &Value) -> Option<LdTools> {
    let object = value.as_object()?;
    Some(
        object
            .iter()
            .filter_map(|(name, tool)| {
                serde_json::from_value::<LdTool>(tool.clone())
                    .ok()
                    .map(|tool| (name.clone(), tool))
            })
            .collect(),
    )
}

impl FlagValue {
    pub fn parse(value: &Value) -> Self {
        let empty = Map::new();
        let object = value.as_object().unwrap_or(&empty);
        let meta = object
            .get("_ldMeta")
            .and_then(Value::as_object)
            .map(|m| FlagMeta {
                variation_key: m
                    .get("variationKey")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                enabled: m.get("enabled").and_then(Value::as_bool).unwrap_or(false),
                version: as_version(m.get("version")),
                mode: m.get("mode").and_then(Value::as_str).map(str::to_string),
                model_key: m
                    .get("modelKey")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                model_version: as_version(m.get("modelVersion")),
            });
        Self {
            meta,
            model: lenient(object, "model"),
            messages: lenient_messages(object),
            provider: lenient(object, "provider"),
            instructions: lenient(object, "instructions"),
            evaluation_metric_key: lenient(object, "evaluationMetricKey"),
            evaluation_metric_keys: lenient(object, "evaluationMetricKeys"),
            judge_configuration: lenient(object, "judgeConfiguration"),
            tools: object.get("tools").and_then(lenient_tools),
        }
    }

    pub fn mode(&self) -> Option<&str> {
        self.meta.as_ref().and_then(|m| m.mode.as_deref())
    }

    fn enabled(&self) -> bool {
        self.meta.as_ref().is_some_and(|m| m.enabled)
    }

    fn resolve_tools(&self) -> Option<LdTools> {
        if self.tools.is_some() {
            return self.tools.clone();
        }
        let raw = self
            .model
            .as_ref()?
            .parameters
            .as_ref()?
            .get("tools")?
            .as_array()?;
        let tools: LdTools = raw
            .iter()
            .filter_map(|entry| serde_json::from_value::<LdTool>(entry.clone()).ok())
            .filter(|tool| !tool.name.is_empty())
            .map(|tool| (tool.name.clone(), tool))
            .collect();
        (!tools.is_empty()).then_some(tools)
    }

    fn judge_metric_key(&self) -> Option<String> {
        if let Some(key) = self
            .evaluation_metric_key
            .as_deref()
            .map(str::trim)
            .filter(|k| !k.is_empty())
        {
            return Some(key.to_string());
        }
        self.evaluation_metric_keys
            .as_ref()?
            .iter()
            .filter_map(Value::as_str)
            .map(str::trim)
            .find(|k| !k.is_empty())
            .map(str::to_string)
    }
}

pub(crate) fn to_flag_value(config: &LdAiConfigDefaultKind, mode: LdAiConfigMode) -> Value {
    let (enabled, model, provider) = match config {
        LdAiConfigDefaultKind::Completion(c) => (c.enabled, &c.model, &c.provider),
        LdAiConfigDefaultKind::Agent(c) => (c.enabled, &c.model, &c.provider),
        LdAiConfigDefaultKind::Judge(c) => (c.enabled, &c.model, &c.provider),
    };
    let mut flag = Map::new();
    flag.insert(
        "_ldMeta".to_string(),
        json!({ "variationKey": "", "enabled": enabled.unwrap_or(false), "mode": mode.as_str() }),
    );
    let mut put = |key: &str, value: Option<Value>| {
        if let Some(value) = value {
            flag.insert(key.to_string(), value);
        }
    };
    put("model", model.as_ref().map(to_value));
    put("provider", provider.as_ref().map(to_value));
    match config {
        LdAiConfigDefaultKind::Completion(c) => {
            put("messages", c.messages.as_ref().map(to_value));
            put(
                "judgeConfiguration",
                c.judge_configuration.as_ref().map(to_value),
            );
            put("tools", c.tools.as_ref().map(to_value));
        }
        LdAiConfigDefaultKind::Agent(c) => {
            put("instructions", c.instructions.clone().map(Value::String));
            put(
                "judgeConfiguration",
                c.judge_configuration.as_ref().map(to_value),
            );
            put("tools", c.tools.as_ref().map(to_value));
        }
        LdAiConfigDefaultKind::Judge(c) => {
            put("messages", c.messages.as_ref().map(to_value));
            put(
                "evaluationMetricKey",
                c.evaluation_metric_key.clone().map(Value::String),
            );
        }
    }
    Value::Object(flag)
}

fn to_value<T: serde::Serialize>(value: &T) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

pub(crate) fn from_flag_value(
    key: &str,
    flag: &FlagValue,
    tracker_factory: TrackerFactory,
    evaluator: Evaluator,
) -> LdAiConfigKind {
    match flag.mode() {
        Some("agent") => {
            LdAiConfigKind::Agent(to_agent_config(key, flag, tracker_factory, evaluator))
        }
        Some("judge") => LdAiConfigKind::Judge(to_judge_config(key, flag, tracker_factory)),
        _ => {
            LdAiConfigKind::Completion(to_completion_config(key, flag, tracker_factory, evaluator))
        }
    }
}

pub(crate) fn create_disabled_config(
    key: &str,
    mode: LdAiConfigMode,
    tracker_factory: TrackerFactory,
    evaluator: Evaluator,
) -> LdAiConfigKind {
    let disabled = FlagValue::default();
    match mode {
        LdAiConfigMode::Agent => {
            LdAiConfigKind::Agent(to_agent_config(key, &disabled, tracker_factory, evaluator))
        }
        LdAiConfigMode::Judge => {
            LdAiConfigKind::Judge(to_judge_config(key, &disabled, tracker_factory))
        }
        LdAiConfigMode::Completion => LdAiConfigKind::Completion(to_completion_config(
            key,
            &disabled,
            tracker_factory,
            evaluator,
        )),
    }
}

pub(crate) fn to_completion_config(
    key: &str,
    flag: &FlagValue,
    tracker_factory: TrackerFactory,
    evaluator: Evaluator,
) -> LdAiCompletionConfig {
    LdAiCompletionConfig {
        key: key.to_string(),
        enabled: flag.enabled(),
        model: flag.model.clone(),
        provider: flag.provider.clone(),
        messages: flag.messages.clone(),
        judge_configuration: flag.judge_configuration.clone(),
        tools: flag.resolve_tools(),
        tracker_factory,
        evaluator,
    }
}

pub(crate) fn to_agent_config(
    key: &str,
    flag: &FlagValue,
    tracker_factory: TrackerFactory,
    evaluator: Evaluator,
) -> LdAiAgentConfig {
    LdAiAgentConfig {
        key: key.to_string(),
        enabled: flag.enabled(),
        model: flag.model.clone(),
        provider: flag.provider.clone(),
        instructions: flag.instructions.clone(),
        judge_configuration: flag.judge_configuration.clone(),
        tools: flag.resolve_tools(),
        tracker_factory,
        evaluator,
    }
}

pub(crate) fn to_judge_config(
    key: &str,
    flag: &FlagValue,
    tracker_factory: TrackerFactory,
) -> LdAiJudgeConfig {
    LdAiJudgeConfig {
        key: key.to_string(),
        enabled: flag.enabled(),
        model: flag.model.clone(),
        provider: flag.provider.clone(),
        messages: flag.messages.clone(),
        evaluation_metric_key: flag.judge_metric_key(),
        tracker_factory,
    }
}
