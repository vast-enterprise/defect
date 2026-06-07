//! Amazon Bedrock provider.
//!
//! Bedrock chat request bodies still use the Anthropic Messages shape, but the transport
//! uses the AWS Bedrock Runtime SDK. Only the Anthropic messages protocol is supported;
//! no higher-level concepts like `instance` are introduced.

use std::env;
use std::fmt::Debug;

use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::Client as BedrockClient;
use aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError;
use aws_sdk_bedrockruntime::primitives::{Blob, event_stream::EventReceiver};
use aws_sdk_bedrockruntime::types::{ResponseStream, error::ResponseStreamError};
use aws_smithy_runtime_api::client::orchestrator::HttpResponse;
use aws_smithy_runtime_api::client::result::SdkError;
use aws_smithy_types::error::metadata::ProvideErrorMetadata;
use aws_smithy_types::event_stream::RawMessage;
use defect_agent::error::BoxError;
use defect_agent::llm::{
    Capabilities, CompletionRequest, FeatureSupport, LlmProvider, ModelCapabilityOverrides,
    ModelInfo, ProtocolId, ProviderError, ProviderErrorKind, ProviderInfo, ProviderStream,
    RateLimitScope, ThinkingEcho, TimeoutPhase,
};
use futures::FutureExt;
use futures::future::BoxFuture;
use futures::{Stream, stream};
use serde_json::Value;
use sse_stream::Sse;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use crate::protocol::anthropic_messages;
use crate::wire::anthropic::components as wire;

const DEFAULT_AWS_REGION: &str = "us-east-1";
const DEFAULT_VENDOR: &str = "bedrock";
const DEFAULT_DISPLAY_NAME: &str = "Amazon Bedrock";
const ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";
const CONTENT_TYPE_JSON: &str = "application/json";
const AWS_REGION_ENV: &str = "AWS_REGION";
const AWS_PROFILE_ENV: &str = "AWS_PROFILE";
const BODY_MODEL_FIELD: &str = "model";
const BODY_STREAM_FIELD: &str = "stream";
const BODY_ANTHROPIC_VERSION_FIELD: &str = "anthropic_version";
const ERR_ACCESS_DENIED: &str = "AccessDeniedException";
const ERR_VALIDATION: &str = "ValidationException";
const ERR_MODEL_NOT_READY: &str = "ModelNotReadyException";
const ERR_SERVICE_UNAVAILABLE: &str = "ServiceUnavailableException";
const ERR_THROTTLING: &str = "ThrottlingException";
const ERR_INTERNAL_SERVER: &str = "InternalServerException";
const ERR_MODEL_STREAM: &str = "ModelStreamErrorException";
const ERR_MODEL_TIMEOUT: &str = "ModelTimeoutException";
const ERR_RESOURCE_NOT_FOUND: &str = "ResourceNotFoundException";
const ERR_SERVICE_QUOTA_EXCEEDED: &str = "ServiceQuotaExceededException";
const ERR_MODEL_ERROR: &str = "ModelErrorException";

#[derive(Debug, Default, Clone)]
pub struct BedrockConfig {
    pub vendor: Option<String>,
    pub display_name: Option<String>,
    pub base_url: Option<String>,
    pub default_model: Option<String>,
    pub models: Vec<String>,
    pub aws_profile: Option<String>,
    pub aws_region: Option<String>,
}

impl BedrockConfig {
    fn resolve_region(&self) -> String {
        self.aws_region
            .clone()
            .or_else(|| env::var(AWS_REGION_ENV).ok())
            .unwrap_or_else(|| DEFAULT_AWS_REGION.to_owned())
    }

    fn resolve_profile(&self) -> Option<String> {
        self.aws_profile
            .clone()
            .or_else(|| env::var(AWS_PROFILE_ENV).ok())
    }
}

pub struct BedrockProvider {
    client: BedrockClient,
    info: ProviderInfo,
    capabilities: Capabilities,
    models: Vec<ModelInfo>,
}

impl std::fmt::Debug for BedrockProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BedrockProvider")
            .field("info", &self.info)
            .field("capabilities", &self.capabilities)
            .finish_non_exhaustive()
    }
}

impl BedrockProvider {
    /// # Errors
    ///
    /// Returns an error if the AWS SDK configuration fails to load or the Bedrock client
    /// fails to initialize.
    pub async fn new(config: BedrockConfig) -> Result<Self, ProviderError> {
        let region = config.resolve_region();
        let vendor = config
            .vendor
            .clone()
            .unwrap_or_else(|| DEFAULT_VENDOR.to_owned());
        let display_name = config
            .display_name
            .clone()
            .unwrap_or_else(|| DEFAULT_DISPLAY_NAME.to_owned());
        let mut loader =
            aws_config::defaults(BehaviorVersion::latest()).region(aws_config::Region::new(region));
        if let Some(profile) = config.resolve_profile() {
            loader = loader.profile_name(profile);
        }
        if let Some(endpoint) = config.base_url.clone() {
            loader = loader.endpoint_url(endpoint);
        }
        let sdk_config = loader.load().await;
        let client = BedrockClient::new(&sdk_config);

        Ok(Self {
            client,
            info: ProviderInfo {
                vendor,
                protocol: ProtocolId::AnthropicMessages,
                display_name,
            },
            capabilities: Capabilities {
                tool_calls: FeatureSupport::Supported,
                parallel_tool_calls: FeatureSupport::Supported,
                thinking: FeatureSupport::Supported,
                vision: FeatureSupport::Supported,
                prompt_cache: FeatureSupport::Supported,
                thinking_echo: ThinkingEcho::Required,
            },
            models: model_infos_from_config(config.models, config.default_model),
        })
    }
}

fn model_infos_from_config(models: Vec<String>, default_model: Option<String>) -> Vec<ModelInfo> {
    let mut ids = models;
    if let Some(default_model) = default_model
        && !ids.iter().any(|id| id == &default_model)
    {
        ids.insert(0, default_model);
    }
    ids.into_iter()
        .map(|id| ModelInfo {
            id,
            display_name: None,
            context_window: None,
            max_output_tokens: None,
            deprecated: false,
            capabilities_overrides: ModelCapabilityOverrides::default(),
        })
        .collect()
}

impl LlmProvider for BedrockProvider {
    fn info(&self) -> ProviderInfo {
        self.info.clone()
    }

    fn capabilities(&self) -> Capabilities {
        self.capabilities
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>> {
        async move { Ok(self.models.clone()) }.boxed()
    }

    fn model_info(&self, model_id: &str) -> Option<ModelInfo> {
        self.models
            .iter()
            .find(|model| model.id == model_id)
            .cloned()
    }

    fn complete(
        &self,
        req: CompletionRequest,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        async move {
            let body = anthropic_messages::encode_request(&req);
            let payload = serde_json::to_vec(&bedrock_request_body(body)).map_err(|e| {
                ProviderError::new(ProviderErrorKind::BadRequest {
                    hint: Some(e.to_string()),
                })
            })?;

            let resp = tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    return Err(ProviderError::new(ProviderErrorKind::Canceled));
                }
                r = self
                    .client
                    .invoke_model_with_response_stream()
                    .model_id(req.model.clone())
                    .content_type(CONTENT_TYPE_JSON)
                    .accept(CONTENT_TYPE_JSON)
                    .body(Blob::new(payload))
                    .send() => r,
            };

            let output = match resp {
                Ok(output) => output,
                Err(err) => return Err(map_bedrock_error(err, &req.model)),
            };

            let events = bedrock_event_stream(output.body, cancel.clone());
            let chunks = anthropic_messages::decode_stream_provider_errors(events, cancel);
            Ok(Box::pin(chunks) as ProviderStream)
        }
        .boxed()
    }
}

fn bedrock_request_body(body: wire::CreateMessageParams) -> Value {
    let mut value = serde_json::to_value(body).expect("Anthropic wire body should serialize");
    if let Some(obj) = value.as_object_mut() {
        obj.remove(BODY_MODEL_FIELD);
        obj.remove(BODY_STREAM_FIELD);
        obj.insert(
            BODY_ANTHROPIC_VERSION_FIELD.to_owned(),
            Value::String(ANTHROPIC_VERSION.to_owned()),
        );
    }
    value
}

type InvokeModelError = SdkError<InvokeModelWithResponseStreamError, HttpResponse>;
type BedrockStreamError = SdkError<ResponseStreamError, RawMessage>;

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
struct BedrockSdkError {
    message: String,
}

fn map_bedrock_error(err: InvokeModelError, model: &str) -> ProviderError {
    match err {
        SdkError::DispatchFailure(e) => {
            ProviderError::new(ProviderErrorKind::Transport(box_debug_error(e)))
        }
        SdkError::TimeoutError(_) => ProviderError::new(ProviderErrorKind::Timeout {
            phase: TimeoutPhase::Total,
        }),
        SdkError::ConstructionFailure(e) => ProviderError::new(ProviderErrorKind::BadRequest {
            hint: Some(format!("{e:?}")),
        }),
        SdkError::ResponseError(e) => {
            ProviderError::new(ProviderErrorKind::Transport(box_debug_error(e)))
        }
        SdkError::ServiceError(e) => bedrock_service_error(e.err(), Some(model)),
        unknown => ProviderError::new(ProviderErrorKind::Other(box_debug_error(unknown))),
    }
}

fn box_debug_error(error: impl Debug) -> BoxError {
    BoxError::new(BedrockSdkError {
        message: format!("{error:?}"),
    })
}

fn bedrock_event_stream(
    body: EventReceiver<ResponseStream, ResponseStreamError>,
    cancel: CancellationToken,
) -> impl Stream<Item = Result<Sse, ProviderError>> + Send {
    stream::unfold((body, cancel), |(mut body, cancel)| async move {
        loop {
            if cancel.is_cancelled() {
                return None;
            }

            let received = tokio::select! {
                biased;
                _ = cancel.cancelled() => return None,
                item = body.recv() => item,
            };

            let item = match received {
                Ok(Some(ResponseStream::Chunk(chunk))) => bedrock_chunk_to_sse(chunk),
                Ok(Some(event)) if event.is_unknown() => {
                    warn!("bedrock returned an unknown response stream event");
                    continue;
                }
                Ok(Some(event)) => {
                    warn!(
                        ?event,
                        "bedrock returned an unhandled response stream event"
                    );
                    continue;
                }
                Ok(None) => return None,
                Err(err) => Err(map_bedrock_stream_error(err)),
            };

            return Some((item, (body, cancel)));
        }
    })
}

fn bedrock_chunk_to_sse(
    chunk: aws_sdk_bedrockruntime::types::PayloadPart,
) -> Result<Sse, ProviderError> {
    let Some(bytes) = chunk.bytes else {
        return Err(ProviderError::new(ProviderErrorKind::ProtocolViolation {
            hint: "bedrock response chunk did not include bytes".into(),
        }));
    };
    let data = String::from_utf8(bytes.into_inner())
        .map_err(|e| ProviderError::new(ProviderErrorKind::Malformed(BoxError::new(e))))?;
    Ok(Sse {
        event: None,
        data: Some(data),
        id: None,
        retry: None,
    })
}

fn map_bedrock_stream_error(err: BedrockStreamError) -> ProviderError {
    match err {
        SdkError::DispatchFailure(e) => {
            ProviderError::new(ProviderErrorKind::Transport(box_debug_error(e)))
        }
        SdkError::TimeoutError(_) => ProviderError::new(ProviderErrorKind::Timeout {
            phase: TimeoutPhase::ReadBody,
        }),
        SdkError::ConstructionFailure(e) => ProviderError::new(ProviderErrorKind::BadRequest {
            hint: Some(format!("{e:?}")),
        }),
        SdkError::ResponseError(e) => {
            ProviderError::new(ProviderErrorKind::Transport(box_debug_error(e)))
        }
        SdkError::ServiceError(e) => bedrock_service_error(e.err(), None),
        unknown => ProviderError::new(ProviderErrorKind::Other(box_debug_error(unknown))),
    }
}

fn bedrock_service_error(source: &dyn ProvideErrorMetadata, model: Option<&str>) -> ProviderError {
    let hint = source.message().map(str::to_owned);
    match source.code() {
        Some(ERR_ACCESS_DENIED) => ProviderError::new(ProviderErrorKind::AuthRejected { hint }),
        Some(ERR_VALIDATION) => ProviderError::new(ProviderErrorKind::BadRequest { hint }),
        Some(ERR_RESOURCE_NOT_FOUND) => ProviderError::new(ProviderErrorKind::ModelNotFound {
            model: model.unwrap_or(DEFAULT_VENDOR).to_owned(),
        }),
        Some(ERR_SERVICE_QUOTA_EXCEEDED) => {
            ProviderError::new(ProviderErrorKind::QuotaExceeded { hint })
        }
        Some(ERR_THROTTLING) => ProviderError::new(ProviderErrorKind::RateLimit {
            retry_after: None,
            scope: RateLimitScope::Unspecified,
        }),
        Some(ERR_MODEL_TIMEOUT) => ProviderError::new(ProviderErrorKind::Timeout {
            phase: TimeoutPhase::ReadBody,
        }),
        Some(ERR_MODEL_STREAM) => {
            ProviderError::new(ProviderErrorKind::ServerStreamAborted { hint })
        }
        Some(ERR_MODEL_NOT_READY)
        | Some(ERR_SERVICE_UNAVAILABLE)
        | Some(ERR_INTERNAL_SERVER)
        | Some(ERR_MODEL_ERROR) => {
            ProviderError::new(ProviderErrorKind::ServerError { status: None, hint })
        }
        Some(_) | None => ProviderError::new(ProviderErrorKind::ServerError { status: None, hint }),
    }
}

#[cfg(test)]
mod tests;
