//! Identity this crate reports in the `$ld:ai:sdk:info` event.

/// The name reported as `aiSdkName` in the `$ld:ai:sdk:info` event.
///
/// The JavaScript SDK reports `@launchdarkly/server-sdk-ai`; this port reports its own
/// crate name so LaunchDarkly never mistakes it for an official SDK.
pub const AI_SDK_NAME: &str = env!("CARGO_PKG_NAME");

/// The version reported as `aiSdkVersion` in the `$ld:ai:sdk:info` event.
pub const AI_SDK_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The language reported as `aiSdkLanguage` in the `$ld:ai:sdk:info` event.
pub const AI_SDK_LANGUAGE: &str = "rust";

/// The version of `@launchdarkly/server-sdk-ai` whose behaviour this crate ports.
pub const PORTED_JS_SDK_VERSION: &str = "2.0.19";
