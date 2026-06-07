//! An implementation of [`rmcp::transport::streamable_http_client::StreamableHttpClient`]
//! based on [`defect_http::ProxyAwareConnector`] + `hyper-util`.
//!
//! Replaces rmcp's built-in reqwest backend to avoid pulling in a second copy of
//! reqwest and a duplicate TLS stack. The MCP client shares the same connector
//! (proxy / NO_PROXY / default UA / system root certs) used by LLM/fetch, provided
//! by [`defect_http::build_proxy_connector`].
//!
//! Implements only the subset required by v0:
//! - JSON-RPC POST with Accepted / JSON / SSE responses;
//! - GET SSE stream;
//! - DELETE session;
//! - Bearer auth header (transparent passthrough);
//! - WWW-Authenticate 401 / 403 parsed into [`AuthRequiredError`] /
//!   [`InsufficientScopeError`].
//!
//! Not implemented: OAuth flow (rmcp `auth` feature) — that is not enabled in v0.

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use defect_http::ProxyAwareConnector;
use futures::{StreamExt, stream::BoxStream};
use http::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, WWW_AUTHENTICATE};
use http::{HeaderName, HeaderValue, Method, Request, StatusCode};
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper_util::client::legacy::Client as HyperClient;
use rmcp::model::{ClientJsonRpcMessage, JsonRpcMessage, ServerJsonRpcMessage};
use rmcp::transport::streamable_http_client::{
    AuthRequiredError, InsufficientScopeError, SseError, StreamableHttpClient, StreamableHttpError,
    StreamableHttpPostResponse,
};
use sse_stream::SseStream;

const HEADER_SESSION_ID: &str = "Mcp-Session-Id";
const HEADER_LAST_EVENT_ID: &str = "Last-Event-Id";
const HEADER_MCP_PROTOCOL_VERSION: &str = "MCP-Protocol-Version";
const EVENT_STREAM_MIME_TYPE: &str = "text/event-stream";
const JSON_MIME_TYPE: &str = "application/json";

/// Aligns with rmcp's internal `RESERVED_HEADERS` — prevents users from overriding these
/// control-semantic fields with custom headers (`MCP-Protocol-Version` is exempt because
/// the worker injects it after init, so it is allowed).
const RESERVED_HEADERS: &[&str] = &[
    "accept",
    HEADER_SESSION_ID,
    HEADER_MCP_PROTOCOL_VERSION,
    HEADER_LAST_EVENT_ID,
];

/// Shared hyper-util client type for [`ProxyAwareConnector`] — MCP POST bodies only send
/// known-size JSON, so the body uses [`Full<Bytes>`].
type StreamableHttpHyperClient = HyperClient<ProxyAwareConnector, Full<Bytes>>;

/// A hyper-based [`StreamableHttpClient`] implementation.
///
/// `Clone + Send + 'static` are required by the trait. [`HyperClient`] is already
/// `Clone + Send + Sync`, and wrapping the user-agent in an [`Arc`] makes cloning cheap.
#[derive(Clone)]
pub struct HyperStreamableHttpClient {
    inner: StreamableHttpHyperClient,
    user_agent: HeaderValue,
}

impl std::fmt::Debug for HyperStreamableHttpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HyperStreamableHttpClient")
            .field("user_agent", &self.user_agent)
            .finish()
    }
}

impl HyperStreamableHttpClient {
    /// Convenience constructor: builds a client that shares the same connector and
    /// user-agent from a [`defect_http::HttpStackConfig`].
    ///
    /// # Errors
    ///
    /// Returns an error if the connector fails to build (e.g., TLS root loading, proxy
    /// URL parsing) or if the user-agent is invalid.
    pub fn from_stack_config(
        config: &defect_http::HttpStackConfig,
    ) -> Result<Self, defect_http::HttpStackError> {
        let connector = defect_http::build_proxy_connector(&config.proxy)?;
        let inner = HyperClient::builder(hyper_util::rt::TokioExecutor::default())
            .build::<_, Full<Bytes>>(connector);
        let user_agent = match &config.user_agent {
            Some(s) => {
                HeaderValue::from_str(s).map_err(|e| defect_http::HttpStackError::Config {
                    hint: format!("invalid user_agent: {e}"),
                })?
            }
            None => defect_http::default_user_agent(),
        };
        Ok(Self { inner, user_agent })
    }
}

impl StreamableHttpClient for HyperStreamableHttpClient {
    type Error = HyperClientError;

    async fn post_message(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_token: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<Self::Error>> {
        let body = serde_json::to_vec(&message).map_err(StreamableHttpError::Deserialize)?;

        let mut builder = Request::builder()
            .method(Method::POST)
            .uri(uri.as_ref())
            .header(
                ACCEPT,
                format!("{EVENT_STREAM_MIME_TYPE}, {JSON_MIME_TYPE}"),
            )
            .header(CONTENT_TYPE, JSON_MIME_TYPE);

        builder = apply_auth(builder, auth_token.as_deref())?;
        builder = apply_user_agent(builder, &self.user_agent);
        builder = apply_custom_headers(builder, &custom_headers)?;
        let session_was_attached = session_id.is_some();
        if let Some(session_id) = session_id.as_deref() {
            builder = builder.header(HEADER_SESSION_ID, session_id);
        }

        let request = builder
            .body(Full::new(Bytes::from(body)))
            .map_err(|e| StreamableHttpError::Client(HyperClientError::Build(e)))?;

        let response = self
            .inner
            .request(request)
            .await
            .map_err(|e| StreamableHttpError::Client(HyperClientError::Send(Box::new(e))))?;

        let status = response.status();
        let session_id_header = response
            .headers()
            .get(HEADER_SESSION_ID)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let content_type_header = response
            .headers()
            .get(CONTENT_TYPE)
            .map(|ct| String::from_utf8_lossy(ct.as_bytes()).into_owned());

        if status == StatusCode::UNAUTHORIZED
            && let Some(header) = response.headers().get(WWW_AUTHENTICATE)
        {
            let header = header
                .to_str()
                .map_err(|_| {
                    StreamableHttpError::UnexpectedServerResponse(std::borrow::Cow::Borrowed(
                        "invalid www-authenticate header value",
                    ))
                })?
                .to_owned();
            return Err(StreamableHttpError::AuthRequired(AuthRequiredError::new(
                header,
            )));
        }

        if status == StatusCode::FORBIDDEN
            && let Some(header) = response.headers().get(WWW_AUTHENTICATE)
        {
            let header_str = header
                .to_str()
                .map_err(|_| {
                    StreamableHttpError::UnexpectedServerResponse(std::borrow::Cow::Borrowed(
                        "invalid www-authenticate header value",
                    ))
                })?
                .to_owned();
            let scope = extract_scope_from_header(&header_str);
            return Err(StreamableHttpError::InsufficientScope(
                InsufficientScopeError::new(header_str, scope),
            ));
        }

        if matches!(status, StatusCode::ACCEPTED | StatusCode::NO_CONTENT) {
            return Ok(StreamableHttpPostResponse::Accepted);
        }

        if status == StatusCode::NOT_FOUND && session_was_attached {
            return Err(StreamableHttpError::SessionExpired);
        }

        if !status.is_success() {
            let body_bytes = read_body(response.into_body()).await?;
            let body_text = String::from_utf8_lossy(&body_bytes).into_owned();
            if content_type_header
                .as_deref()
                .is_some_and(|ct| ct.as_bytes().starts_with(JSON_MIME_TYPE.as_bytes()))
                && let Some(message) = parse_json_rpc_error(&body_text)
            {
                return Ok(StreamableHttpPostResponse::Json(message, session_id_header));
            }
            return Err(StreamableHttpError::UnexpectedServerResponse(
                std::borrow::Cow::Owned(format!("HTTP {status}: {body_text}")),
            ));
        }

        match content_type_header.as_deref() {
            Some(ct) if ct.as_bytes().starts_with(EVENT_STREAM_MIME_TYPE.as_bytes()) => {
                let stream = sse_body_stream(response.into_body());
                Ok(StreamableHttpPostResponse::Sse(stream, session_id_header))
            }
            Some(ct) if ct.as_bytes().starts_with(JSON_MIME_TYPE.as_bytes()) => {
                let body_bytes = read_body(response.into_body()).await?;
                match serde_json::from_slice::<ServerJsonRpcMessage>(&body_bytes) {
                    Ok(message) => Ok(StreamableHttpPostResponse::Json(message, session_id_header)),
                    Err(e) => {
                        tracing::warn!(
                            "could not parse JSON response as ServerJsonRpcMessage, treating as accepted: {e}"
                        );
                        Ok(StreamableHttpPostResponse::Accepted)
                    }
                }
            }
            other => {
                tracing::error!("unexpected content type: {other:?}");
                Err(StreamableHttpError::UnexpectedContentType(
                    other.map(str::to_owned),
                ))
            }
        }
    }

    async fn delete_session(
        &self,
        uri: Arc<str>,
        session: Arc<str>,
        auth_token: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<(), StreamableHttpError<Self::Error>> {
        let mut builder = Request::builder()
            .method(Method::DELETE)
            .uri(uri.as_ref())
            .header(HEADER_SESSION_ID, session.as_ref());
        builder = apply_auth(builder, auth_token.as_deref())?;
        builder = apply_user_agent(builder, &self.user_agent);
        builder = apply_custom_headers(builder, &custom_headers)?;

        let request = builder
            .body(Full::new(Bytes::new()))
            .map_err(|e| StreamableHttpError::Client(HyperClientError::Build(e)))?;
        let response = self
            .inner
            .request(request)
            .await
            .map_err(|e| StreamableHttpError::Client(HyperClientError::Send(Box::new(e))))?;

        let status = response.status();
        if status == StatusCode::METHOD_NOT_ALLOWED {
            tracing::debug!("server does not support deleting session");
            return Ok(());
        }
        if !status.is_success() {
            let body_bytes = read_body(response.into_body()).await?;
            let body_text = String::from_utf8_lossy(&body_bytes).into_owned();
            return Err(StreamableHttpError::UnexpectedServerResponse(
                std::borrow::Cow::Owned(format!("HTTP {status}: {body_text}")),
            ));
        }
        Ok(())
    }

    async fn get_stream(
        &self,
        uri: Arc<str>,
        session_id: Arc<str>,
        last_event_id: Option<String>,
        auth_token: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<
        BoxStream<'static, Result<sse_stream::Sse, SseError>>,
        StreamableHttpError<Self::Error>,
    > {
        let mut builder = Request::builder()
            .method(Method::GET)
            .uri(uri.as_ref())
            .header(
                ACCEPT,
                format!("{EVENT_STREAM_MIME_TYPE}, {JSON_MIME_TYPE}"),
            )
            .header(HEADER_SESSION_ID, session_id.as_ref());
        if let Some(last_event_id) = last_event_id {
            builder = builder.header(HEADER_LAST_EVENT_ID, last_event_id);
        }
        builder = apply_auth(builder, auth_token.as_deref())?;
        builder = apply_user_agent(builder, &self.user_agent);
        builder = apply_custom_headers(builder, &custom_headers)?;

        let request = builder
            .body(Full::new(Bytes::new()))
            .map_err(|e| StreamableHttpError::Client(HyperClientError::Build(e)))?;
        let response = self
            .inner
            .request(request)
            .await
            .map_err(|e| StreamableHttpError::Client(HyperClientError::Send(Box::new(e))))?;

        let status = response.status();
        if status == StatusCode::METHOD_NOT_ALLOWED {
            return Err(StreamableHttpError::ServerDoesNotSupportSse);
        }
        if !status.is_success() {
            let body_bytes = read_body(response.into_body()).await?;
            let body_text = String::from_utf8_lossy(&body_bytes).into_owned();
            return Err(StreamableHttpError::UnexpectedServerResponse(
                std::borrow::Cow::Owned(format!("HTTP {status}: {body_text}")),
            ));
        }

        match response.headers().get(CONTENT_TYPE) {
            Some(ct) => {
                let bytes = ct.as_bytes();
                if !bytes.starts_with(EVENT_STREAM_MIME_TYPE.as_bytes())
                    && !bytes.starts_with(JSON_MIME_TYPE.as_bytes())
                {
                    return Err(StreamableHttpError::UnexpectedContentType(Some(
                        String::from_utf8_lossy(bytes).into_owned(),
                    )));
                }
            }
            None => return Err(StreamableHttpError::UnexpectedContentType(None)),
        }

        Ok(sse_body_stream(response.into_body()))
    }
}

/// All low-level errors from hyper-util / hyper are unified into this single client
/// error, which wraps the inner `StreamableHttpError::Client(_)`.
#[derive(Debug, thiserror::Error)]
pub enum HyperClientError {
    #[error("failed to build HTTP request: {0}")]
    Build(#[source] http::Error),
    #[error("HTTP transport error: {0}")]
    Send(#[source] Box<dyn std::error::Error + Send + Sync>),
    #[error("failed to read response body: {0}")]
    ReadBody(#[source] Box<dyn std::error::Error + Send + Sync>),
}

fn apply_auth(
    builder: http::request::Builder,
    auth_token: Option<&str>,
) -> Result<http::request::Builder, StreamableHttpError<HyperClientError>> {
    if let Some(token) = auth_token {
        let value = HeaderValue::from_str(&format!("Bearer {token}")).map_err(|_| {
            StreamableHttpError::UnexpectedServerResponse(std::borrow::Cow::Borrowed(
                "invalid auth bearer token (not a valid header value)",
            ))
        })?;
        Ok(builder.header(AUTHORIZATION, value))
    } else {
        Ok(builder)
    }
}

fn apply_user_agent(
    builder: http::request::Builder,
    value: &HeaderValue,
) -> http::request::Builder {
    builder.header(http::header::USER_AGENT, value.clone())
}

fn apply_custom_headers(
    mut builder: http::request::Builder,
    custom_headers: &HashMap<HeaderName, HeaderValue>,
) -> Result<http::request::Builder, StreamableHttpError<HyperClientError>> {
    for (name, value) in custom_headers {
        validate_custom_header(name).map_err(StreamableHttpError::ReservedHeaderConflict)?;
        builder = builder.header(name.clone(), value.clone());
    }
    Ok(builder)
}

/// Validates reserved headers consistently with rmcp internals: users may not override
/// any reserved header except `MCP-Protocol-Version`.
fn validate_custom_header(name: &HeaderName) -> Result<(), String> {
    if RESERVED_HEADERS
        .iter()
        .any(|&r| name.as_str().eq_ignore_ascii_case(r))
    {
        if name
            .as_str()
            .eq_ignore_ascii_case(HEADER_MCP_PROTOCOL_VERSION)
        {
            return Ok(());
        }
        return Err(name.to_string());
    }
    Ok(())
}

/// Matches the behavior of rmcp's internal `extract_scope_from_header`: extracts the
/// `scope=` value from `WWW-Authenticate` (with or without quotes), returning `None` if
/// not found. Logic directly ported from rmcp 1.7.0.
fn extract_scope_from_header(header: &str) -> Option<String> {
    let lower = header.to_ascii_lowercase();
    let key = "scope=";
    let pos = lower.find(key)?;
    let start = pos + key.len();
    let value = header.get(start..)?;
    if let Some(stripped) = value.strip_prefix('"') {
        let end = stripped.find('"')?;
        return Some(stripped[..end].to_string());
    }
    let end = value
        .find(|c: char| c == ',' || c == ';' || c.is_whitespace())
        .unwrap_or(value.len());
    if end == 0 {
        None
    } else {
        Some(value[..end].to_string())
    }
}

fn parse_json_rpc_error(body: &str) -> Option<ServerJsonRpcMessage> {
    match serde_json::from_str::<ServerJsonRpcMessage>(body) {
        Ok(m @ JsonRpcMessage::Error(_)) => Some(m),
        _ => None,
    }
}

async fn read_body(body: Incoming) -> Result<Bytes, StreamableHttpError<HyperClientError>> {
    let collected = body
        .collect()
        .await
        .map_err(|e| StreamableHttpError::Client(HyperClientError::ReadBody(Box::new(e))))?;
    Ok(collected.to_bytes())
}

/// Feeds a hyper [`Incoming`] directly into [`SseStream`] — avoids one layer of frame
/// wrapping compared to the reqwest `bytes_stream → from_byte_stream` path.
fn sse_body_stream(body: Incoming) -> BoxStream<'static, Result<sse_stream::Sse, SseError>> {
    SseStream::new(body).boxed()
}
