//! Built-in `fetch` tool: reads a URL, renders content (markdown / html / text), enforces
//! timeout and size limits.

use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_client_protocol_schema::{
    Content, ContentBlock, TextContent, ToolCallContent, ToolCallUpdateFields, ToolKind,
};
use defect_agent::error::BoxError;
use defect_agent::http::{HttpClient, HttpClientError, HttpRequest, HttpResponse};
use defect_agent::tool::{
    SafetyClass, Tool, ToolCallDescription, ToolContext, ToolError, ToolEvent, ToolSchema,
    ToolStream,
};
use defect_config::{FetchFormat, FetchToolConfig};
use futures::future::BoxFuture;
use futures::stream;
use serde::{Deserialize, Serialize};
use serde_json::json;

mod render;

#[cfg(test)]
mod tests;

const TITLE_TRUNC: usize = 80;

/// Built-in implementation of the `fetch` tool. Stateless — a singleton
/// `Arc::new(FetchTool::new(cfg))` suffices.
pub struct FetchTool {
    schema: ToolSchema,
    config: FetchToolConfig,
}

impl FetchTool {
    /// Constructs using [`FetchToolConfig::default`].
    pub fn new() -> Self {
        Self::from_config(&FetchToolConfig::default())
    }

    /// Constructs from a [`FetchToolConfig`].
    pub fn from_config(config: &FetchToolConfig) -> Self {
        let default_timeout = config.default_timeout_secs.max(1);
        let max_timeout = config.max_timeout_secs.max(default_timeout);
        let default_format = format_to_str(config.default_format);
        let schema = ToolSchema {
            name: "fetch".to_string(),
            description: format!(
                "Fetch a URL and return its content. \
                 Supports HTTP/HTTPS only. Renders HTML to markdown by default; \
                 raw HTML / plain text via `format`. Times out after `timeout_secs` \
                 (default {default_timeout}; max {max_timeout}). \
                 Truncates responses larger than {} bytes.",
                config.max_response_bytes
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Absolute http:// or https:// URL. Other schemes are rejected."
                    },
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "html", "text"],
                        "description": format!(
                            "Output format. Defaults to `{default_format}` (configured in [tools.fetch]). \
                             `markdown` runs the html→markdown pipeline; \
                             `html` returns raw HTML; `text` strips tags but keeps text."
                        )
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": max_timeout as i64,
                        "description": format!(
                            "Per-call timeout in seconds. Defaults to {default_timeout}. \
                             Capped at {max_timeout} (clamped silently)."
                        )
                    }
                },
                "required": ["url"]
            }),
        };
        let mut effective = config.clone();
        effective.default_timeout_secs = default_timeout;
        effective.max_timeout_secs = max_timeout;
        Self {
            schema,
            config: effective,
        }
    }
}

impl Default for FetchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize)]
struct FetchArgs {
    url: String,
    #[serde(default)]
    format: Option<FetchFormat>,
    #[serde(default)]
    timeout_secs: Option<u32>,
}

#[derive(Debug, Serialize)]
struct FetchOutput {
    status: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    content_type: Option<String>,
    bytes_received: u64,
    bytes_returned: u64,
    truncated: bool,
    redirects: u32,
    elapsed_ms: u64,
    final_url: String,
    /// `Some(original_value)` when the per-request `timeout_secs` was clamped to
    /// `max_timeout_secs`.
    #[serde(skip_serializing_if = "Option::is_none")]
    timeout_clamped_from: Option<u32>,
}

impl Tool for FetchTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }

    fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
        // P2 only supports GET; the URL is user-controlled and has no local side effects,
        // so it is ReadOnly.
        SafetyClass::ReadOnly
    }

    fn describe<'a>(
        &'a self,
        args: &'a serde_json::Value,
        _ctx: ToolContext<'a>,
    ) -> BoxFuture<'a, ToolCallDescription> {
        Box::pin(async move {
            let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("");
            let title = format!("Fetch {}", truncate_title(url));
            let mut fields = ToolCallUpdateFields::default();
            fields.title = Some(title);
            fields.kind = Some(ToolKind::Fetch);
            ToolCallDescription { fields }
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: ToolContext<'_>) -> ToolStream {
        let cancel = ctx.cancel.clone();
        let http = ctx.http.clone();
        let config = self.config.clone();
        let fut = async move { run_fetch(args, http, cancel, config).await };
        let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> = Box::pin(stream::once(fut));
        s
    }
}

async fn run_fetch(
    args: serde_json::Value,
    http: Arc<dyn HttpClient>,
    cancel: tokio_util::sync::CancellationToken,
    config: FetchToolConfig,
) -> ToolEvent {
    let parsed: FetchArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(err) => return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(err))),
    };

    // Pre-validate the URL scheme so that non-http/https URLs fail with `InvalidArgs`
    // (§10 #7) rather than `Execution`.
    if let Err(reason) = validate_scheme(&parsed.url) {
        return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            reason,
        ))));
    }

    let format = parsed.format.unwrap_or(config.default_format);
    let requested_timeout = parsed.timeout_secs.unwrap_or(config.default_timeout_secs);
    let timeout_clamped_from =
        (requested_timeout > config.max_timeout_secs).then_some(requested_timeout);
    let timeout_secs = requested_timeout.min(config.max_timeout_secs).max(1);

    let request = HttpRequest {
        url: parsed.url.clone(),
        timeout: Some(Duration::from_secs(u64::from(timeout_secs))),
        follow_redirects: config.follow_redirects,
        max_redirects: 10,
        max_response_bytes: config.max_response_bytes,
    };

    let started = Instant::now();
    let response = tokio::select! {
        biased;
        _ = cancel.cancelled() => {
            return ToolEvent::Failed(ToolError::Canceled);
        }
        res = http.fetch(request) => res,
    };

    let elapsed_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;

    let response = match response {
        Ok(r) => r,
        Err(err) => return map_http_error(err, timeout_secs),
    };

    finalize(response, format, &config, elapsed_ms, timeout_clamped_from)
}

fn map_http_error(err: HttpClientError, timeout_secs: u32) -> ToolEvent {
    let mapped = match err {
        HttpClientError::InvalidUrl(reason) => ToolError::InvalidArgs(BoxError::new(
            std::io::Error::new(std::io::ErrorKind::InvalidInput, reason),
        )),
        HttpClientError::Timeout => ToolError::Execution(BoxError::new(std::io::Error::other(
            format!("timed out after {timeout_secs}s"),
        ))),
        HttpClientError::TooManyRedirects(n) => ToolError::Execution(BoxError::new(
            std::io::Error::other(format!("too many redirects ({n})")),
        )),
        HttpClientError::Transport(source) => ToolError::Execution(source),
        other => ToolError::Execution(BoxError::new(std::io::Error::other(format!("{other}")))),
    };
    ToolEvent::Failed(mapped)
}

fn finalize(
    response: HttpResponse,
    format: FetchFormat,
    config: &FetchToolConfig,
    elapsed_ms: u64,
    timeout_clamped_from: Option<u32>,
) -> ToolEvent {
    let HttpResponse {
        status,
        content_type,
        body,
        bytes_received,
        truncated,
        redirects,
        final_url,
    } = response;

    let render_result = render::render(&body, content_type.as_deref(), format, config);
    let mut text = match render_result {
        Ok(t) => t,
        Err(e) => {
            return ToolEvent::Failed(ToolError::Execution(BoxError::new(std::io::Error::other(
                e,
            ))));
        }
    };

    let bytes_returned = text.len() as u64;

    if truncated {
        let dropped = bytes_received.saturating_sub(config.max_response_bytes);
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(&format!(
            "[response truncated; {dropped} additional bytes dropped]"
        ));
    }
    if status >= 400 {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(&format!("[http status: {status}]"));
    }

    let raw_output = serde_json::to_value(FetchOutput {
        status,
        content_type,
        bytes_received,
        bytes_returned,
        truncated,
        redirects,
        elapsed_ms,
        final_url,
        timeout_clamped_from,
    })
    .unwrap_or(serde_json::Value::Null);

    let mut fields = ToolCallUpdateFields::default();
    fields.content = Some(vec![ToolCallContent::Content(Content::new(
        ContentBlock::Text(TextContent::new(text)),
    ))]);
    fields.raw_output = Some(raw_output);
    ToolEvent::Completed(fields)
}

fn validate_scheme(url: &str) -> Result<(), String> {
    let trimmed = url.trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") {
        return Ok(());
    }
    Err(format!(
        "unsupported URL scheme; only http/https allowed: {url}"
    ))
}

fn format_to_str(f: FetchFormat) -> &'static str {
    match f {
        FetchFormat::Markdown => "markdown",
        FetchFormat::Html => "html",
        FetchFormat::Text => "text",
    }
}

fn truncate_title(s: &str) -> String {
    if s.chars().count() <= TITLE_TRUNC {
        return s.to_string();
    }
    let truncated: String = s.chars().take(TITLE_TRUNC).collect();
    format!("{truncated}…")
}
