//! Cross-module error utilities.
//!
//! If a variant in an error type (e.g. [`crate::llm::ProviderError`], or `tool::ToolError`
//! in `defect-agent`) needs to propagate an arbitrary `std::error::Error`,
//! **always use [`BoxError`]** instead of a bare
//! `Box<dyn std::error::Error + Send + Sync>`.
//!
//! Using a newtype (rather than a type alias) has these advantages:
//! - Shorter, more readable type signatures
//! - Distinguishes from "any dyn Error" at the type level, making caller intent clearer
//! - Future implementation changes (e.g. switching to `anyhow::Error`, adding backtrace
//!   support) require only one change

use std::error::Error as StdError;
use std::fmt;

/// A type-erased error value. Carries an error from any source in a public API without
/// exposing the concrete type.
///
/// Construction:
/// - [`BoxError::new`]: wraps any `E: Error + Send + Sync + 'static`
/// - `From<Box<dyn Error + Send + Sync>>`: migrates from an already-boxed form
///
/// **No** `From<E>` for arbitrary `E: Error`: under Rust's coherence rules, this would
/// overlap with the blanket `From<T> for T` impl (since `BoxError` itself implements
/// `Error`). Callers should use [`BoxError::new`] to wrap explicitly.
#[derive(Debug)]
pub struct BoxError(Box<dyn StdError + Send + Sync>);

impl BoxError {
    /// Wraps any `std::error::Error`.
    pub fn new<E>(err: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self(Box::new(err))
    }
}

impl fmt::Display for BoxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl StdError for BoxError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.0.source()
    }
}

impl From<Box<dyn StdError + Send + Sync>> for BoxError {
    fn from(value: Box<dyn StdError + Send + Sync>) -> Self {
        Self(value)
    }
}
