//! Error types.

/// A boxed, thread-safe error, used wherever the JavaScript SDK lets an arbitrary
/// exception propagate (runner invocations, metrics extractors).
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Errors returned by this crate's own operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A resumption token could not be decoded (bad Base64 or bad JSON payload).
    #[error("invalid resumption token: {0}")]
    InvalidResumptionToken(String),
}
