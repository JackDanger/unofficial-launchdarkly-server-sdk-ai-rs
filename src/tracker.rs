//! Per-run metric tracking for an AI Config (`LDAIConfigTracker` / `LDAIConfigTrackerImpl`).

use std::fmt;
use std::future::Future;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::client_min::{Context, LdClientMin};
use crate::error::Error;
use crate::judge::LdJudgeResult;
use crate::metrics::{LdAiMetricSummary, LdAiMetrics, LdFeedback, LdFeedbackKind, LdTokenUsage};

/// Returns the current time in milliseconds; trackers measure durations with it.
pub type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;

pub(crate) fn monotonic_clock() -> Clock {
    let origin = Instant::now();
    Arc::new(move || origin.elapsed().as_millis() as f64)
}

pub(crate) const TOKEN_ENGINE: base64::engine::GeneralPurpose = base64::engine::GeneralPurpose::new(
    &base64::alphabet::URL_SAFE,
    base64::engine::GeneralPurposeConfig::new()
        .with_encode_padding(false)
        .with_decode_padding_mode(base64::engine::DecodePaddingMode::Indifferent),
);

pub(crate) fn decode_token(token: &str) -> Result<Value, Error> {
    let bytes = TOKEN_ENGINE
        .decode(token.trim())
        .map_err(|e| Error::InvalidResumptionToken(e.to_string()))?;
    serde_json::from_slice(&bytes).map_err(|e| Error::InvalidResumptionToken(e.to_string()))
}

/// The metadata attached to every event a tracker emits (JavaScript: `getTrackData()`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LdAiTrackData {
    /// UUIDv4 shared by every event of one AI run.
    pub run_id: String,
    /// The AI Config key.
    pub config_key: String,
    /// The flag variation key (empty when a default was used).
    pub variation_key: String,
    /// The flag variation version.
    pub version: u64,
    /// The model name.
    pub model_name: String,
    /// The provider name.
    pub provider_name: String,
    /// The model version from `_ldMeta.modelVersion` (default 1).
    pub model_version: u64,
    /// The model key from `_ldMeta.modelKey`, omitted when absent or empty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_key: Option<String>,
    /// The agent graph key, when the config was fetched as a graph node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_key: Option<String>,
}

/// Construction parameters for [`LdAiConfigTracker::new`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LdAiConfigTrackerParams {
    /// UUIDv4 identifying the run.
    pub run_id: String,
    /// The AI Config key.
    pub config_key: String,
    /// The flag variation key.
    pub variation_key: String,
    /// The flag variation version.
    pub version: u64,
    /// The model name.
    pub model_name: String,
    /// The provider name.
    pub provider_name: String,
    /// The agent graph key, when tracking a graph node.
    pub graph_key: Option<String>,
    /// The model key from `_ldMeta.modelKey`.
    pub model_key: Option<String>,
    /// The model version from `_ldMeta.modelVersion`.
    pub model_version: u64,
}

impl LdAiConfigTrackerParams {
    /// Parameters with the given identity, empty model/provider names and `modelVersion` 1.
    pub fn new(
        run_id: impl Into<String>,
        config_key: impl Into<String>,
        variation_key: impl Into<String>,
        version: u64,
    ) -> Self {
        Self {
            run_id: run_id.into(),
            config_key: config_key.into(),
            variation_key: variation_key.into(),
            version,
            model_name: String::new(),
            provider_name: String::new(),
            graph_key: None,
            model_key: None,
            model_version: 1,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ResumptionPayload<'a> {
    run_id: &'a str,
    config_key: &'a str,
    variation_key: &'a str,
    version: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    graph_key: Option<&'a str>,
}

struct TrackerInner {
    ld_client: Arc<dyn LdClientMin>,
    context: Context,
    params: LdAiConfigTrackerParams,
    summary: Mutex<LdAiMetricSummary>,
    clock: Clock,
}

/// Records metrics for a single AI run (JavaScript: `LDAIConfigTracker`).
///
/// Every event shares the tracker's `runId`. Most metrics are recorded at most once
/// per tracker: a second call is dropped with a warning; call `create_tracker()` on
/// the AI Config to start a new run. Clones share state.
#[derive(Clone)]
pub struct LdAiConfigTracker {
    inner: Arc<TrackerInner>,
}

impl fmt::Debug for LdAiConfigTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LdAiConfigTracker")
            .field("params", &self.inner.params)
            .field("summary", &*self.summary())
            .finish()
    }
}

impl LdAiConfigTracker {
    /// Creates a tracker that sends events through `ld_client` for `context`.
    pub fn new(
        ld_client: Arc<dyn LdClientMin>,
        context: Context,
        params: LdAiConfigTrackerParams,
    ) -> Self {
        Self::with_clock(ld_client, context, params, monotonic_clock())
    }

    /// Like [`Self::new`], but measuring durations with `clock` (milliseconds).
    pub fn with_clock(
        ld_client: Arc<dyn LdClientMin>,
        context: Context,
        params: LdAiConfigTrackerParams,
        clock: Clock,
    ) -> Self {
        let tracker = Self {
            inner: Arc::new(TrackerInner {
                ld_client,
                context,
                params,
                summary: Mutex::new(LdAiMetricSummary::default()),
                clock,
            }),
        };
        let token = tracker.resumption_token();
        tracker.summary().resumption_token = Some(token);
        tracker
    }

    /// Reconstructs a tracker from a [`Self::resumption_token`], preserving its `runId`
    /// (JavaScript: `LDAIConfigTrackerImpl.fromResumptionToken`).
    ///
    /// Model and provider names come back empty and `modelVersion` is 1, as in JavaScript.
    pub fn from_resumption_token(
        token: &str,
        ld_client: Arc<dyn LdClientMin>,
        context: Context,
    ) -> Result<Self, Error> {
        let payload = decode_token(token)?;
        let field = |name: &str| payload.get(name).and_then(Value::as_str).map(str::to_string);
        let run_id = field("runId")
            .ok_or_else(|| Error::InvalidResumptionToken("missing runId".to_string()))?;
        let config_key = field("configKey")
            .ok_or_else(|| Error::InvalidResumptionToken("missing configKey".to_string()))?;
        let version = crate::config::utils::as_version(payload.get("version"))
            .ok_or_else(|| Error::InvalidResumptionToken("missing version".to_string()))?;
        let mut params = LdAiConfigTrackerParams::new(
            run_id,
            config_key,
            field("variationKey").unwrap_or_default(),
            version,
        );
        params.graph_key = field("graphKey");
        Ok(Self::new(ld_client, context, params))
    }

    fn summary(&self) -> MutexGuard<'_, LdAiMetricSummary> {
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

    fn track_data_value(&self) -> Value {
        serde_json::to_value(self.get_track_data()).unwrap_or(Value::Null)
    }

    fn track_data_with(&self, extra: &[(&str, Value)]) -> Value {
        let mut data = self.track_data_value();
        if let Value::Object(map) = &mut data {
            for (key, value) in extra {
                map.insert((*key).to_string(), value.clone());
            }
        }
        data
    }

    /// The metadata attached to every event.
    pub fn get_track_data(&self) -> LdAiTrackData {
        let p = &self.inner.params;
        LdAiTrackData {
            run_id: p.run_id.clone(),
            config_key: p.config_key.clone(),
            variation_key: p.variation_key.clone(),
            version: p.version,
            model_name: p.model_name.clone(),
            provider_name: p.provider_name.clone(),
            model_version: p.model_version,
            model_key: p.model_key.clone().filter(|k| !k.is_empty()),
            graph_key: p.graph_key.clone(),
        }
    }

    /// The context events are recorded for.
    pub fn context(&self) -> &Context {
        &self.inner.context
    }

    /// A URL-safe, unpadded Base64 token encoding `runId`, `configKey`,
    /// `variationKey`, `version` (and `graphKey` when set), in that order.
    ///
    /// Pass it to [`crate::LdAiClient::create_tracker`] to resume this run elsewhere.
    pub fn resumption_token(&self) -> String {
        let p = &self.inner.params;
        let json = serde_json::to_vec(&ResumptionPayload {
            run_id: &p.run_id,
            config_key: &p.config_key,
            variation_key: &p.variation_key,
            version: p.version,
            graph_key: p.graph_key.as_deref(),
        })
        .unwrap_or_default();
        TOKEN_ENGINE.encode(json)
    }

    /// Tracks generation duration (`$ld:ai:duration:total`). At most once per tracker.
    pub fn track_duration(&self, duration_ms: f64) {
        {
            let mut summary = self.summary();
            if summary.duration_ms.is_some() {
                drop(summary);
                self.warn("Skipping trackDuration: duration already recorded on this tracker. Call createTracker on the AI Config for a new run.");
                return;
            }
            summary.duration_ms = Some(duration_ms);
        }
        self.track("$ld:ai:duration:total", self.track_data_value(), duration_ms);
    }

    /// Runs `func`, tracking its duration whatever it returns (JavaScript:
    /// `trackDurationOf`). It does not record an error when `func` returns `Err`.
    pub async fn track_duration_of<T, Fut>(&self, func: impl FnOnce() -> Fut) -> T
    where
        Fut: Future<Output = T>,
    {
        let start = (self.inner.clock)();
        let result = func().await;
        let end = (self.inner.clock)();
        self.track_duration(end - start);
        result
    }

    /// Tracks time to first token (`$ld:ai:tokens:ttf`). At most once per tracker.
    pub fn track_time_to_first_token(&self, time_to_first_token_ms: f64) {
        {
            let mut summary = self.summary();
            if summary.time_to_first_token_ms.is_some() {
                drop(summary);
                self.warn("Skipping trackTimeToFirstToken: time-to-first-token already recorded on this tracker. Call createTracker on the AI Config for a new run.");
                return;
            }
            summary.time_to_first_token_ms = Some(time_to_first_token_ms);
        }
        self.track(
            "$ld:ai:tokens:ttf",
            self.track_data_value(),
            time_to_first_token_ms,
        );
    }

    /// Tracks a judge result under its metric key, with the score as the metric value
    /// and `judgeConfigKey` added to the data. Nothing is sent unless the result was
    /// sampled and successful and carries both a metric key and a score.
    pub fn track_judge_result(&self, result: &LdJudgeResult) {
        if !result.sampled || !result.success {
            return;
        }
        if let (Some(metric_key), Some(score)) = (&result.metric_key, result.score) {
            let data = match result.judge_config_key.as_deref().filter(|k| !k.is_empty()) {
                Some(judge_key) => self.track_data_with(&[("judgeConfigKey", judge_key.into())]),
                None => self.track_data_value(),
            };
            self.track(metric_key, data, score);
        }
    }

    /// Tracks one tool invocation (`$ld:ai:tool_call`, with `toolKey` in the data).
    pub fn track_tool_call(&self, tool_key: &str) {
        self.summary()
            .tool_calls
            .get_or_insert_with(Vec::new)
            .push(tool_key.to_string());
        self.track(
            "$ld:ai:tool_call",
            self.track_data_with(&[("toolKey", tool_key.into())]),
            1.0,
        );
    }

    /// Tracks several tool invocations.
    pub fn track_tool_calls<S: AsRef<str>>(&self, tool_keys: &[S]) {
        for key in tool_keys {
            self.track_tool_call(key.as_ref());
        }
    }

    /// Tracks user feedback (`$ld:ai:feedback:user:positive|negative`). At most once.
    pub fn track_feedback(&self, feedback: impl Into<LdFeedback>) {
        let feedback = feedback.into();
        {
            let mut summary = self.summary();
            if summary.feedback.is_some() {
                drop(summary);
                self.warn("Skipping trackFeedback: feedback already recorded on this tracker. Call createTracker on the AI Config for a new run.");
                return;
            }
            summary.feedback = Some(feedback);
        }
        let key = match feedback.kind {
            LdFeedbackKind::Positive => "$ld:ai:feedback:user:positive",
            LdFeedbackKind::Negative => "$ld:ai:feedback:user:negative",
        };
        self.track(key, self.track_data_value(), 1.0);
    }

    fn record_outcome(&self, success: bool, method: &str) -> bool {
        let mut summary = self.summary();
        if summary.success.is_some() {
            drop(summary);
            self.warn(&format!("Skipping {method}: success/error already recorded on this tracker. Call createTracker on the AI Config for a new run."));
            return false;
        }
        summary.success = Some(success);
        true
    }

    /// Tracks a successful generation (`$ld:ai:generation:success`). Shares its
    /// at-most-once slot with [`Self::track_error`].
    pub fn track_success(&self) {
        if self.record_outcome(true, "trackSuccess") {
            self.track("$ld:ai:generation:success", self.track_data_value(), 1.0);
        }
    }

    /// Tracks a failed generation (`$ld:ai:generation:error`). Shares its
    /// at-most-once slot with [`Self::track_success`].
    pub fn track_error(&self) {
        if self.record_outcome(false, "trackError") {
            self.track("$ld:ai:generation:error", self.track_data_value(), 1.0);
        }
    }

    /// Tracks token usage (`$ld:ai:tokens:total|input|output`, each only when
    /// non-zero). At most once per tracker.
    pub fn track_tokens(&self, tokens: LdTokenUsage) {
        {
            let mut summary = self.summary();
            if summary.tokens.is_some() {
                drop(summary);
                self.warn("Skipping trackTokens: token usage already recorded on this tracker. Call createTracker on the AI Config for a new run.");
                return;
            }
            summary.tokens = Some(tokens);
        }
        let data = self.track_data_value();
        for (key, count) in [
            ("$ld:ai:tokens:total", tokens.total),
            ("$ld:ai:tokens:input", tokens.input),
            ("$ld:ai:tokens:output", tokens.output),
        ] {
            if count > 0 {
                self.track(key, data.clone(), count as f64);
            }
        }
    }

    fn track_metrics(&self, metrics: &LdAiMetrics) {
        if metrics.success {
            self.track_success();
        } else {
            self.track_error();
        }
        if let Some(tokens) = metrics.tokens {
            self.track_tokens(tokens);
        }
        if let Some(tool_calls) = metrics.tool_calls.as_ref().filter(|c| !c.is_empty()) {
            self.track_tool_calls(tool_calls);
        }
    }

    /// Runs an AI operation, tracking its duration, then success or error, token usage
    /// and tool calls from the metrics `extractor` derives from the result
    /// (JavaScript: `trackMetricsOf`). When `func` fails, the duration and an error are
    /// tracked and the error is returned.
    pub async fn track_metrics_of<T, E, Fut>(
        &self,
        extractor: impl FnOnce(&T) -> LdAiMetrics,
        func: impl FnOnce() -> Fut,
    ) -> Result<T, E>
    where
        Fut: Future<Output = Result<T, E>>,
    {
        let result = match self.track_duration_of(func).await {
            Ok(result) => result,
            Err(err) => {
                self.track_error();
                return Err(err);
            }
        };
        self.track_metrics(&extractor(&result));
        Ok(result)
    }

    /// Tracks a streaming AI operation (JavaScript: `trackStreamMetricsOf`).
    ///
    /// `stream_creator` runs synchronously and its stream is returned immediately.
    /// `metrics_extractor` borrows the stream to build a future that resolves once
    /// the stream's metrics are known; that future runs in the background (on the
    /// current Tokio runtime, or a helper thread without one) and records success or
    /// error, tokens, tool calls and finally the duration. When the extractor's future
    /// fails, an error is tracked. When `stream_creator` fails, the duration and an
    /// error are tracked and the error is returned.
    pub fn track_stream_metrics_of<S, E, ME, Fut>(
        &self,
        stream_creator: impl FnOnce() -> Result<S, E>,
        metrics_extractor: impl FnOnce(&S) -> Fut,
    ) -> Result<S, E>
    where
        Fut: Future<Output = Result<LdAiMetrics, ME>> + Send + 'static,
    {
        let start = (self.inner.clock)();
        let stream = match stream_creator() {
            Ok(stream) => stream,
            Err(err) => {
                self.track_duration((self.inner.clock)() - start);
                self.track_error();
                return Err(err);
            }
        };
        let metrics = metrics_extractor(&stream);
        let tracker = self.clone();
        crate::runtime::spawn_detached(async move {
            match metrics.await {
                Ok(metrics) => tracker.track_metrics(&metrics),
                Err(_) => tracker.track_error(),
            }
            tracker.track_duration((tracker.inner.clock)() - start);
        });
        Ok(stream)
    }

    /// A snapshot of everything recorded so far, including the resumption token.
    pub fn get_summary(&self) -> LdAiMetricSummary {
        self.summary().clone()
    }
}
