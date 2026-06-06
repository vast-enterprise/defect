use agent_client_protocol_schema::{
    BlobResourceContents, ContentBlock, EmbeddedResource, EmbeddedResourceResource, ImageContent,
    ResourceLink, TextContent, TextResourceContents,
};

use super::content_block_to_message_content;
use super::{DEFAULT_MAX_HOOK_CONTINUES, TurnRequestLimit, TurnState};
use crate::llm::{ImageData, MessageContent};
use crate::session::TurnError;

// ----- TurnState: before-turn-end 续命硬上限 -----

#[test]
fn stop_hook_continues_capped() {
    let mut state = TurnState::new(TurnRequestLimit::Unbounded, DEFAULT_MAX_HOOK_CONTINUES);
    // 初始可续命。
    assert!(state.may_stop_hook_continue());
    // 续到上限。
    for _ in 0..DEFAULT_MAX_HOOK_CONTINUES {
        assert!(state.may_stop_hook_continue());
        state.note_stop_hook_continue();
    }
    // 达上限后不再允许——防死循环。
    assert!(!state.may_stop_hook_continue());
}

#[test]
fn stop_hook_continues_respects_custom_cap() {
    // 自定义上限 1：续命一次后即到顶。
    let mut state = TurnState::new(TurnRequestLimit::Unbounded, 1);
    assert!(state.may_stop_hook_continue());
    state.note_stop_hook_continue();
    assert!(!state.may_stop_hook_continue());
}

#[test]
fn stop_hook_continue_counter_starts_zero() {
    let state = TurnState::new(TurnRequestLimit::Unbounded, DEFAULT_MAX_HOOK_CONTINUES);
    assert_eq!(state.stop_hook_continues, 0);
    assert!(state.may_stop_hook_continue());
}

#[test]
fn text_content_stays_text() {
    let content = content_block_to_message_content(ContentBlock::Text(TextContent::new("hello")))
        .expect("text should convert");

    assert_eq!(
        content,
        vec![MessageContent::Text {
            text: "hello".to_string()
        }]
    );
}

#[test]
fn image_content_with_data_becomes_base64_image() {
    let content = content_block_to_message_content(ContentBlock::Image(ImageContent::new(
        "aGVsbG8=",
        "image/png",
    )))
    .expect("image should convert");

    assert_eq!(
        content,
        vec![MessageContent::Image {
            mime: "image/png".to_string(),
            data: ImageData::Base64 {
                encoded: "aGVsbG8=".to_string(),
            },
        }]
    );
}

#[test]
fn image_content_with_uri_becomes_url_image() {
    let content = content_block_to_message_content(ContentBlock::Image(
        ImageContent::new("", "image/png").uri("https://example.com/cat.png"),
    ))
    .expect("image uri should convert");

    assert_eq!(
        content,
        vec![MessageContent::Image {
            mime: "image/png".to_string(),
            data: ImageData::Url {
                url: "https://example.com/cat.png".to_string(),
            },
        }]
    );
}

#[test]
fn resource_link_becomes_descriptive_text() {
    let content = content_block_to_message_content(ContentBlock::ResourceLink(
        ResourceLink::new("spec", "file:///tmp/spec.md")
            .title("API spec")
            .description("Design document")
            .mime_type("text/markdown")
            .size(128_i64),
    ))
    .expect("resource link should convert");

    assert_eq!(
        content,
        vec![MessageContent::Text {
            text: [
                "resource: spec",
                "title: API spec",
                "description: Design document",
                "mime_type: text/markdown",
                "size: 128",
                "uri: file:///tmp/spec.md",
            ]
            .join("\n"),
        }]
    );
}

#[test]
fn text_resource_becomes_text_with_source_header() {
    let content = content_block_to_message_content(ContentBlock::Resource(EmbeddedResource::new(
        EmbeddedResourceResource::TextResourceContents(
            TextResourceContents::new("fn main() {}\n", "file:///tmp/main.rs")
                .mime_type("text/rust"),
        ),
    )))
    .expect("text resource should convert");

    assert_eq!(
        content,
        vec![MessageContent::Text {
            text: "resource: file:///tmp/main.rs\nmime_type: text/rust\n\nfn main() {}\n"
                .to_string(),
        }]
    );
}

#[test]
fn audio_content_is_rejected() {
    let err = content_block_to_message_content(ContentBlock::Audio(
        agent_client_protocol_schema::AudioContent::new("aGVsbG8=", "audio/wav"),
    ))
    .expect_err("audio should be rejected");

    assert!(matches!(err, TurnError::Internal(_)));
    assert_eq!(
        err.to_string(),
        "internal turn error: ACP audio content is not supported yet"
    );
}

#[test]
fn blob_resource_is_rejected() {
    let err = content_block_to_message_content(ContentBlock::Resource(EmbeddedResource::new(
        EmbeddedResourceResource::BlobResourceContents(
            BlobResourceContents::new("aGVsbG8=", "file:///tmp/image.png").mime_type("image/png"),
        ),
    )))
    .expect_err("blob resource should be rejected");

    assert!(matches!(err, TurnError::Internal(_)));
    assert_eq!(
        err.to_string(),
        "internal turn error: embedded binary resource is not supported yet: file:///tmp/image.png"
    );
}
