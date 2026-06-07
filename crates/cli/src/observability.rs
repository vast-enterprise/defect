//! Translates a typed Langfuse config from `defect-config` into an observer for
//! `defect-obs`.
//!
//! Translation and validation happen at CLI assembly time: `defect-obs` does not depend
//! on `defect-config` (preserving a one-way dependency), and policy checks like "enabled
//! but missing key" naturally belong in the assembly layer.
//!
//! Observability setup — tracing and Langfuse integration.

use std::time::Duration;

use defect_config::LangfuseConfig;
use defect_obs::LangfuseObserver;
use defect_obs::langfuse::{
    DEFAULT_FLUSH_INTERVAL, DEFAULT_HOST, DEFAULT_MAX_BATCH, LangfuseSetup, build_observer,
};

/// Builds an observer from a typed `LangfuseConfig`.
///
/// Returns `Ok(None)` when the observer is not enabled (config missing or `enabled =
/// false`) — the caller should not attach an observer. When `enabled = true` but
/// `public_key` / `secret_key` are missing, a warning is emitted and the observer is
/// **disabled** (returns `Ok(None)`). This does not error or silently succeed, matching
/// the explicit validation contract.
///
/// # Errors
///
/// Returns an error if the HTTP stack fails to build (e.g., TLS roots or proxy URL
/// parsing).
pub fn build_langfuse_observer(
    config: Option<&LangfuseConfig>,
    http_stack_config: defect_http::HttpStackConfig,
) -> anyhow::Result<Option<LangfuseObserver>> {
    let Some(cfg) = config.filter(|c| c.enabled) else {
        return Ok(None);
    };

    let (Some(public_key), Some(secret_key)) = (cfg.public_key.clone(), cfg.secret_key.clone())
    else {
        tracing::warn!(
            "tracing.langfuse.enabled = true but public_key / secret_key missing; \
             langfuse reporting disabled"
        );
        return Ok(None);
    };

    let http = defect_http::build_http_stack(http_stack_config)
        .map_err(|e| anyhow::anyhow!("langfuse http stack init failed: {e}"))?;

    let setup = LangfuseSetup {
        host: cfg.host.clone().unwrap_or_else(|| DEFAULT_HOST.to_string()),
        public_key,
        secret_key,
        flush_interval: cfg
            .flush_interval_ms
            .map_or(DEFAULT_FLUSH_INTERVAL, Duration::from_millis),
        max_batch: cfg.max_batch.unwrap_or(DEFAULT_MAX_BATCH),
    };

    Ok(Some(build_observer(setup, http)))
}
