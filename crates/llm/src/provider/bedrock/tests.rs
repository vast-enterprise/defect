use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_bedrockruntime::types::PayloadPart;
use defect_agent::llm::{
    CompletionRequest, Message, MessageContent, ProviderChunk, Role, SamplingParams, ToolChoice,
};
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use super::*;

const TEST_MODEL: &str = "anthropic.claude-sonnet-4-5-20250929-v1:0";
const USER_TEXT: &str = "hello";
const MODEL_START: &str = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"anthropic.claude-sonnet-4-5-20250929-v1:0","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":7,"output_tokens":1}}}"#;
const TEXT_START: &str = r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"","citations":[]}}"#;
const TEXT_DELTA: &str =
    r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#;
const TEXT_STOP: &str = r#"{"type":"content_block_stop","index":0}"#;
const MSG_DELTA_END: &str =
    r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}"#;
const MSG_STOP: &str = r#"{"type":"message_stop"}"#;

fn minimal_request() -> CompletionRequest {
    CompletionRequest {
        model: TEST_MODEL.to_string(),
        system: None,
        messages: vec![Message {
            role: Role::User,
            content: vec![MessageContent::Text {
                text: USER_TEXT.to_string(),
            }]
            .into(),
        }],
        tools: Vec::new(),
        tool_choice: ToolChoice::Auto,
        sampling: SamplingParams::default(),
        hosted_capabilities: defect_agent::llm::HostedCapabilities::default(),
    }
}

fn chunk(data: &str) -> PayloadPart {
    PayloadPart::builder()
        .bytes(Blob::new(data.as_bytes()))
        .build()
}

#[test]
fn bedrock_body_adds_version_and_removes_direct_anthropic_fields() {
    let body = anthropic_messages::encode_request(&minimal_request());
    let value = bedrock_request_body(body);
    let obj = value.as_object().expect("bedrock body object");

    assert_eq!(
        obj.get(BODY_ANTHROPIC_VERSION_FIELD)
            .and_then(serde_json::Value::as_str),
        Some(ANTHROPIC_VERSION)
    );
    assert!(!obj.contains_key(BODY_MODEL_FIELD));
    assert!(!obj.contains_key(BODY_STREAM_FIELD));
    assert!(obj.contains_key("messages"));
    assert!(obj.contains_key("max_tokens"));
}

#[test]
fn model_list_comes_from_config_and_includes_default_model() {
    let models = model_infos_from_config(
        vec!["anthropic.claude-opus-4-1".to_string()],
        Some(TEST_MODEL.to_string()),
    );

    let ids = models
        .iter()
        .map(|model| model.id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(ids, [TEST_MODEL, "anthropic.claude-opus-4-1"]);
}

#[tokio::test]
async fn bedrock_chunks_decode_as_anthropic_events() {
    let events = [
        MODEL_START,
        TEXT_START,
        TEXT_DELTA,
        TEXT_STOP,
        MSG_DELTA_END,
        MSG_STOP,
    ]
    .into_iter()
    .map(|event| bedrock_chunk_to_sse(chunk(event)))
    .collect::<Vec<_>>();

    let chunks = anthropic_messages::decode_stream_provider_errors(
        futures::stream::iter(events),
        CancellationToken::new(),
    )
    .collect::<Vec<_>>()
    .await
    .into_iter()
    .map(|item| item.expect("provider chunk"))
    .collect::<Vec<_>>();

    assert!(matches!(
        chunks.as_slice(),
        [
            ProviderChunk::MessageStart { .. },
            ProviderChunk::Usage(_),
            ProviderChunk::TextDelta { text },
            ProviderChunk::Stop { .. },
            ProviderChunk::Usage(_),
        ] if text == "hi"
    ));
}
