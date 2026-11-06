//! # unofficial-launchdarkly-server-sdk-ai
//!
//! **Unofficial community port. Not affiliated with, endorsed by or supported by
//! LaunchDarkly. Use at your own risk.** For LaunchDarkly's official SDKs, see
//! <https://docs.launchdarkly.com/sdk>.
//!
//! A Rust port of the server-side JavaScript AI SDK, `@launchdarkly/server-sdk-ai`
//! version 2.0.19 (Apache-2.0). It reads AI Configs from LaunchDarkly flags, renders
//! their prompts with Mustache, and reports the same `$ld:ai:*` events the JavaScript
//! SDK sends, so LaunchDarkly's AI Config monitoring and judges work unchanged.
//!
//! With the default `ld-client` feature, `ld_client` is an
//! `Arc<launchdarkly_server_sdk::Client>`.
//!
//! ```no_run
//! # async fn demo(ld_client: std::sync::Arc<impl unofficial_launchdarkly_server_sdk_ai::LdClientMin + 'static>) {
//! use serde_json::json;
//! use unofficial_launchdarkly_server_sdk_ai::{init_ai, ContextBuilder, LdAiMetrics};
//!
//! let ai_client = init_ai(ld_client);
//! let context = ContextBuilder::new("user-key").build().unwrap();
//!
//! let config = ai_client
//!     .completion_config("my-ai-config", &context, None, json!({"name": "Sandy"}).as_object(), None)
//!     .await;
//! if config.enabled {
//!     let tracker = config.create_tracker();
//!     let response: Result<String, std::io::Error> = tracker
//!         .track_metrics_of(|_| LdAiMetrics::new(true), || async {
//!             Ok("call your model with config.messages here".to_string())
//!         })
//!         .await;
//! }
//! # }
//! ```
//!
//! Method names are the JavaScript ones in `snake_case`; optional JavaScript
//! parameters are `Option`s in the same positions. See the README for the
//! differences from the JavaScript SDK.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod client;
mod client_min;
pub mod config;
mod error;
pub mod graph;
pub mod judge;
mod managed;
pub mod metrics;
pub mod model;
mod mustache;
#[cfg(feature = "openai-compatible")]
pub mod openai_compatible;
pub mod providers;
mod runtime;
mod sdk_info;
mod tracker;

pub use client::*;
pub use client_min::{
    Context, ContextBuilder, LdClientMin, LdLogger, LogCrateLogger, MultiContextBuilder,
};
pub use config::*;
pub use error::{BoxError, Error};
pub use graph::*;
pub use judge::{Evaluator, Judge, LdJudgeResult};
pub use managed::{ManagedAgent, ManagedAgentGraph, ManagedModel};
pub use metrics::*;
pub use model::{JudgeEvaluations, ManagedResult, RunnerResult};
pub use providers::{
    AgentGraphRunner, AiProvider, ModelConfigRef, Runner, RunnerFactory, ToolFn, ToolRegistry,
};
pub use runtime::Eager;
pub use sdk_info::{AI_SDK_LANGUAGE, AI_SDK_NAME, AI_SDK_VERSION, PORTED_JS_SDK_VERSION};
pub use tracker::{Clock, LdAiConfigTracker, LdAiConfigTrackerParams, LdAiTrackData};

/// Renders a Mustache template the way the JavaScript AI SDK does (mustache.js with
/// HTML escaping disabled). Missing variables render empty.
pub fn interpolate_template(
    template: &str,
    variables: &serde_json::Value,
) -> Result<String, mustache::TemplateError> {
    mustache::render(template, variables)
}

pub use mustache::TemplateError;
