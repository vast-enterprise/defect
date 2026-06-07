//! Translates the typed HTTP configuration from `defect-config` into a
//! [`defect_http::HttpStackConfig`].
//!
//! `defect-config` avoids a direct dependency on `defect-http` to keep the crate
//! dependency one-way (see the comment on [`defect_config::HttpClientConfig`]).
//! Performing this translation during CLI assembly is the most natural place — the same
//! stack config is shared by three providers, and any proxy URI parsing failures are
//! reported centrally here.

use std::time::Duration;

use defect_config::{HttpClientConfig, HttpProxyMode, HttpProxySettings};

/// Construct a [`defect_http::HttpStackConfig`] from the typed config.
///
/// # Errors
///
/// Returns an error when `proxy.mode = Explicit` and the `http_proxy` / `https_proxy` URI
/// cannot be parsed, to avoid triggering the same error later during provider assembly.
pub fn build_http_stack_config(
    config: &HttpClientConfig,
) -> anyhow::Result<defect_http::HttpStackConfig> {
    let mut stack = defect_http::HttpStackConfig::default();
    if let Some(ms) = config.total_timeout_ms {
        stack.total_timeout = if ms == 0 {
            None
        } else {
            Some(Duration::from_millis(ms))
        };
    }
    if let Some(retries) = config.transport_retries {
        stack.transport_retries = retries;
    }
    if let Some(ms) = config.initial_backoff_ms {
        stack.initial_backoff = Duration::from_millis(ms);
    }
    if let Some(ua) = &config.user_agent {
        stack.user_agent = Some(ua.clone());
    }
    stack.proxy = match config.proxy.mode {
        HttpProxyMode::FromEnv => defect_http::ProxyConfig::FromEnv,
        HttpProxyMode::Disabled => defect_http::ProxyConfig::Disabled,
        HttpProxyMode::Explicit => {
            defect_http::ProxyConfig::Explicit(parse_proxy_settings(&config.proxy.explicit)?)
        }
    };
    Ok(stack)
}

fn parse_proxy_settings(
    settings: &HttpProxySettings,
) -> anyhow::Result<defect_http::ProxySettings> {
    let parse_uri = |raw: &str, field: &str| -> anyhow::Result<http::Uri> {
        raw.parse::<http::Uri>()
            .map_err(|e| anyhow::anyhow!("invalid http.proxy.{field} `{raw}`: {e}"))
    };
    Ok(defect_http::ProxySettings {
        http_proxy: settings
            .http_proxy
            .as_deref()
            .map(|raw| parse_uri(raw, "http_proxy"))
            .transpose()?,
        https_proxy: settings
            .https_proxy
            .as_deref()
            .map(|raw| parse_uri(raw, "https_proxy"))
            .transpose()?,
        no_proxy: settings.no_proxy.clone(),
    })
}
