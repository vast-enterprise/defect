//! [`LlmProvider`] trait — the primary LLM vendor integration interface.

use std::pin::Pin;

use futures::{Stream, future::BoxFuture};
use tokio_util::sync::CancellationToken;

use super::capability::{Capabilities, HostedCapabilities};
use super::chunk::ProviderChunk;
use super::error::ProviderError;
use super::model::{ModelInfo, ProviderInfo};
use super::request::CompletionRequest;

/// A type-erased stream of events produced by a provider during streaming generation,
/// enabling direct use with `dyn LlmProvider`.
pub type ProviderStream = Pin<Box<dyn Stream<Item = Result<ProviderChunk, ProviderError>> + Send>>;

/// LLM provider abstraction.
///
/// Cancellation semantics: [`LlmProvider::complete`] receives a [`CancellationToken`];
/// the caller may call `cancel()` at any point to abort the call and the downstream
/// stream. Dropping the returned stream also counts as cancellation.
pub trait LlmProvider: Send + Sync {
    /// Provider metadata (vendor name, API style, tracing labels, etc.).
    fn info(&self) -> ProviderInfo;

    /// Provider-level capability matrix. Model-level differences are expressed via
    /// [`super::ModelCapabilityOverrides`] and merged on demand by the main loop.
    fn capabilities(&self) -> Capabilities;

    /// The set of hosted capabilities that this provider adapter advertises.
    ///
    /// Unlike [`Self::capabilities`] (which describes model-level properties), this
    /// reflects the current adapter implementation state: whether it can expose hosted
    /// `web_search`, `fetch`, etc. to the model over the wire. During session startup,
    /// this value is read together with `capabilities.web_search.mode` to determine the
    /// source of hosted web search capabilities.
    ///
    /// The default implementation returns all `false`; new providers do not need to
    /// override it. Adapters that truly support hosted capabilities (Anthropic / OpenAI
    /// Responses) should explicitly override this method.
    fn hosted_capabilities(&self) -> HostedCapabilities {
        HostedCapabilities::default()
    }

    /// Lists the models currently available from this provider.
    ///
    /// The implementation may make network calls (e.g., OpenAI `/v1/models`); results
    /// should be cached inside the provider for synchronous lookup by
    /// [`Self::model_info`].
    ///
    /// # Errors
    ///
    /// Network errors, authentication errors, server errors, etc., are all mapped to
    /// [`ProviderError`].
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>>;

    /// Synchronously query metadata for a given model.
    ///
    /// This is a fast path for context trimming in the main loop; **must not trigger a
    /// network call**.
    /// Returns `None` if the provider's cache does not contain the model. The caller may
    /// then decide to call [`Self::list_models`] and retry, or treat it as an unknown
    /// model.
    fn model_info(&self, model_id: &str) -> Option<ModelInfo>;

    /// Start a streaming generation.
    ///
    /// # Errors
    ///
    /// Authentication failures, invalid parameters, transport errors, server errors, etc.
    /// are all mapped to [`ProviderError`]. Errors produced within the stream are
    /// delivered via the stream's `Err` variant, not through this return value.
    fn complete(
        &self,
        req: CompletionRequest,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>>;
}
