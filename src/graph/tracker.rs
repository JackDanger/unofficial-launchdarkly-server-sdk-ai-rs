use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use base64::Engine;
use serde_json::{json, Value};

use crate::client_min::{Context, LdClientMin};
use crate::error::Error;
use crate::metrics::LdTokenUsage;
use crate::tracker::{decode_token, TOKEN_ENGINE};

use super::types::{LdAiGraphTrackerSummary, LdGraphTrackData};

struct Inner {
    ld_client: Arc<dyn LdClientMin>,
    run_id: String,
    graph_key: String,
    variation_key: Option<String>,
    version: u64,
    context: Context,
    summary: Mutex<LdAiGraphTrackerSummary>,
}

/// Records metrics for a single run of an agent graph (JavaScript: `LDGraphTracker`
/// and `LDGraphTrackerImpl`).
///
/// Graph-level methods are at-most-once per tracker (a repeat call is dropped with a
/// warning); edge-level methods ([`Self::track_redirect`], [`Self::track_handoff_success`],
/// [`Self::track_handoff_failure`]) may fire any number of times. Clones share state.
#[derive(Clone)]
pub struct LdGraphTracker {
    inner: Arc<Inner>,
}

impl fmt::Debug for LdGraphTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LdGraphTracker")
            .field("track_data", &self.get_track_data())
            .field("summary", &*self.summary())
            .finish()
    }
}

impl LdGraphTracker {
    /// Creates a graph tracker.
    pub fn new(
        ld_client: Arc<dyn LdClientMin>,
        run_id: impl Into<String>,
        graph_key: impl Into<String>,
        variation_key: Option<String>,
        version: u64,
        context: Context,
    ) -> Self {
        let tracker = Self {
            inner: Arc::new(Inner {
                ld_client,
                run_id: run_id.into(),
                graph_key: graph_key.into(),
                variation_key,
                version,
                context,
                summary: Mutex::new(LdAiGraphTrackerSummary::default()),
            }),
        };
        let token = tracker.resumption_token();
        tracker.summary().resumption_token = Some(token);
        tracker
    }

    /// Reconstructs a tracker from a [`Self::resumption_token`], preserving its `runId`.
    ///
    /// The token contains the flag variation key and version; keep it server-side.
    pub fn from_resumption_token(
        token: &str,
        ld_client: Arc<dyn LdClientMin>,
        context: Context,
    ) -> Result<Self, Error> {
        let payload = decode_token(token)?;
        let field = |name: &str| payload.get(name).and_then(Value::as_str).map(str::to_string);
        let run_id = field("runId")
            .ok_or_else(|| Error::InvalidResumptionToken("missing runId".to_string()))?;
        let graph_key = field("graphKey")
            .ok_or_else(|| Error::InvalidResumptionToken("missing graphKey".to_string()))?;
        let version = crate::config::utils::as_version(payload.get("version"))
            .ok_or_else(|| Error::InvalidResumptionToken("missing version".to_string()))?;
        Ok(Self::new(
            ld_client,
            run_id,
            graph_key,
            field("variationKey"),
            version,
            context,
        ))
    }

    fn summary(&self) -> MutexGuard<'_, LdAiGraphTrackerSummary> {
        self.inner
            .summary
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn warn(&self, message: &str) {
        if let Some(logger) = self.inner.ld_client.logger() {
            logger.warn(message);
        }
    }

    fn track(&self, key: &str, data: Value, metric_value: f64) {
        self.inner
            .ld_client
            .track(key, &self.inner.context, Some(data), Some(metric_value));
    }

    fn track_data_with(&self, extra: Value) -> Value {
        let mut data = serde_json::to_value(self.get_track_data()).unwrap_or(Value::Null);
        if let (Value::Object(map), Value::Object(extra)) = (&mut data, extra) {
            map.extend(extra);
        }
        data
    }

    /// The metadata attached to every event.
    pub fn get_track_data(&self) -> LdGraphTrackData {
        LdGraphTrackData {
            run_id: self.inner.run_id.clone(),
            graph_key: self.inner.graph_key.clone(),
            variation_key: self.inner.variation_key.clone(),
            version: self.inner.version,
        }
    }

    /// A snapshot of the graph-level metrics recorded so far.
    pub fn get_summary(&self) -> LdAiGraphTrackerSummary {
        self.summary().clone()
    }

    /// A URL-safe, unpadded Base64 token of `{"runId","graphKey","variationKey"?,"version"}`
    /// in that exact key order.
    pub fn resumption_token(&self) -> String {
        let mut json = format!(
            "{{\"runId\":{},\"graphKey\":{}",
            Value::from(self.inner.run_id.as_str()),
            Value::from(self.inner.graph_key.as_str())
        );
        if let Some(variation_key) = &self.inner.variation_key {
            json.push_str(&format!(",\"variationKey\":{}", Value::from(variation_key.as_str())));
        }
        json.push_str(&format!(",\"version\":{}}}", self.inner.version));
        TOKEN_ENGINE.encode(json)
    }

    fn record_outcome(&self, success: bool, method: &str) -> bool {
        let mut summary = self.summary();
        if summary.success.is_some() {
            drop(summary);
            self.warn(&format!("Skipping {method}: invocation result already recorded on this graph tracker. Call createTracker on the agent graph for a new run."));
            return false;
        }
        summary.success = Some(success);
        true
    }

    /// Tracks a successful graph run (`$ld:ai:graph:invocation_success`). At most once,
    /// shared with [`Self::track_invocation_failure`].
    pub fn track_invocation_success(&self) {
        if self.record_outcome(true, "trackInvocationSuccess") {
            self.track("$ld:ai:graph:invocation_success", self.track_data_with(json!({})), 1.0);
        }
    }

    /// Tracks a failed graph run (`$ld:ai:graph:invocation_failure`). At most once,
    /// shared with [`Self::track_invocation_success`].
    pub fn track_invocation_failure(&self) {
        if self.record_outcome(false, "trackInvocationFailure") {
            self.track("$ld:ai:graph:invocation_failure", self.track_data_with(json!({})), 1.0);
        }
    }

    /// Tracks the total graph duration (`$ld:ai:graph:duration:total`). At most once.
    pub fn track_duration(&self, duration_ms: f64) {
        {
            let mut summary = self.summary();
            if summary.duration_ms.is_some() {
                drop(summary);
                self.warn("Skipping trackDuration: duration already recorded on this graph tracker. Call createTracker on the agent graph for a new run.");
                return;
            }
            summary.duration_ms = Some(duration_ms);
        }
        self.track("$ld:ai:graph:duration:total", self.track_data_with(json!({})), duration_ms);
    }

    /// Tracks aggregate token usage (`$ld:ai:graph:total_tokens`, metric value = total).
    /// At most once.
    pub fn track_total_tokens(&self, tokens: LdTokenUsage) {
        {
            let mut summary = self.summary();
            if summary.tokens.is_some() {
                drop(summary);
                self.warn("Skipping trackTotalTokens: tokens already recorded on this graph tracker. Call createTracker on the agent graph for a new run.");
                return;
            }
            summary.tokens = Some(tokens);
        }
        self.track(
            "$ld:ai:graph:total_tokens",
            self.track_data_with(json!({})),
            tokens.total as f64,
        );
    }

    /// Tracks the path taken through the graph (`$ld:ai:graph:path`, with `path` in
    /// the data). At most once.
    pub fn track_path<S: AsRef<str>>(&self, path: &[S]) {
        let path: Vec<String> = path.iter().map(|s| s.as_ref().to_string()).collect();
        {
            let mut summary = self.summary();
            if summary.path.is_some() {
                drop(summary);
                self.warn("Skipping trackPath: path already recorded on this graph tracker. Call createTracker on the agent graph for a new run.");
                return;
            }
            summary.path = Some(path.clone());
        }
        self.track("$ld:ai:graph:path", self.track_data_with(json!({ "path": path })), 1.0);
    }

    /// Tracks a node redirecting to a different target (`$ld:ai:graph:redirect`).
    pub fn track_redirect(&self, source_key: &str, redirected_target: &str) {
        self.track(
            "$ld:ai:graph:redirect",
            self.track_data_with(json!({ "sourceKey": source_key, "redirectedTarget": redirected_target })),
            1.0,
        );
    }

    /// Tracks a successful handoff between nodes (`$ld:ai:graph:handoff_success`).
    pub fn track_handoff_success(&self, source_key: &str, target_key: &str) {
        self.track(
            "$ld:ai:graph:handoff_success",
            self.track_data_with(json!({ "sourceKey": source_key, "targetKey": target_key })),
            1.0,
        );
    }

    /// Tracks a failed handoff between nodes (`$ld:ai:graph:handoff_failure`).
    pub fn track_handoff_failure(&self, source_key: &str, target_key: &str) {
        self.track(
            "$ld:ai:graph:handoff_failure",
            self.track_data_with(json!({ "sourceKey": source_key, "targetKey": target_key })),
            1.0,
        );
    }
}
