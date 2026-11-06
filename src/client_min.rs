//! The minimal LaunchDarkly client surface the AI layer needs (`LDClientMin` in JavaScript).

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

pub use launchdarkly_server_sdk_evaluation::{Context, ContextBuilder, MultiContextBuilder};

/// A logger with the four levels of the JavaScript SDK's `LDLogger`.
pub trait LdLogger: Send + Sync {
    /// Logs at error level.
    fn error(&self, message: &str);
    /// Logs at warn level.
    fn warn(&self, message: &str);
    /// Logs at info level.
    fn info(&self, message: &str);
    /// Logs at debug level.
    fn debug(&self, message: &str);
}

impl std::fmt::Debug for dyn LdLogger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LdLogger")
    }
}

/// An [`LdLogger`] that forwards to the [`log`] crate facade (target
/// `unofficial_launchdarkly_server_sdk_ai`), matching how the Rust core SDK logs.
#[derive(Debug, Default, Clone, Copy)]
pub struct LogCrateLogger;

const LOG_TARGET: &str = "unofficial_launchdarkly_server_sdk_ai";

impl LdLogger for LogCrateLogger {
    fn error(&self, message: &str) {
        log::error!(target: LOG_TARGET, "{message}");
    }
    fn warn(&self, message: &str) {
        log::warn!(target: LOG_TARGET, "{message}");
    }
    fn info(&self, message: &str) {
        log::info!(target: LOG_TARGET, "{message}");
    }
    fn debug(&self, message: &str) {
        log::debug!(target: LOG_TARGET, "{message}");
    }
}

/// The operations the AI SDK needs from a LaunchDarkly server-side client.
///
/// This mirrors the JavaScript SDK's `LDClientMin` interface: a JSON flag
/// evaluation, a custom-event `track` call and an optional logger. It is implemented
/// for [`launchdarkly_server_sdk::Client`] when the `ld-client` feature is enabled
/// (the default); implement it yourself to wrap a different client or a test double.
#[async_trait]
pub trait LdClientMin: Send + Sync {
    /// Evaluates a JSON flag for `context`, returning `default_value` when the flag
    /// cannot be evaluated (JavaScript: `variation(key, context, defaultValue)`).
    async fn variation(&self, key: &str, context: &Context, default_value: Value) -> Value;

    /// Records a custom event (JavaScript: `track(key, context, data, metricValue)`).
    ///
    /// Every event this SDK emits carries a `metric_value`; `data` is `None` only
    /// when the caller of a custom event omits it.
    fn track(&self, key: &str, context: &Context, data: Option<Value>, metric_value: Option<f64>);

    /// The logger the AI SDK writes diagnostics to, if any.
    fn logger(&self) -> Option<Arc<dyn LdLogger>> {
        None
    }
}

#[cfg(feature = "ld-client")]
#[async_trait]
impl LdClientMin for launchdarkly_server_sdk::Client {
    async fn variation(&self, key: &str, context: &Context, default_value: Value) -> Value {
        self.json_variation(context, key, default_value)
    }

    fn track(&self, key: &str, context: &Context, data: Option<Value>, metric_value: Option<f64>) {
        let data = data.unwrap_or(Value::Null);
        match metric_value {
            Some(value) => self.track_metric(context.clone(), key, value, data),
            None => {
                if let Err(err) = self.track_data(context.clone(), key, data) {
                    log::warn!(target: LOG_TARGET, "failed to serialize data for event {key}: {err}");
                }
            }
        }
    }

    fn logger(&self) -> Option<Arc<dyn LdLogger>> {
        Some(Arc::new(LogCrateLogger))
    }
}

#[async_trait]
impl<T: LdClientMin + ?Sized> LdClientMin for Arc<T> {
    async fn variation(&self, key: &str, context: &Context, default_value: Value) -> Value {
        (**self).variation(key, context, default_value).await
    }

    fn track(&self, key: &str, context: &Context, data: Option<Value>, metric_value: Option<f64>) {
        (**self).track(key, context, data, metric_value)
    }

    fn logger(&self) -> Option<Arc<dyn LdLogger>> {
        (**self).logger()
    }
}

pub(crate) fn context_to_value(context: &Context) -> Value {
    serde_json::to_value(context).unwrap_or(Value::Null)
}
