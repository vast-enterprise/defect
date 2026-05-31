//! ACP `ContentBlock` → 内部 `MessageContent` 的转换。
//!
//! 从 turn 主流程疏散出来的纯函数集合：把客户端送来的 ACP 内容块翻译成 LLM 协议层的
//! `MessageContent`。无 IO、无状态，便于单测（见 `turn/test.rs`）。

use std::io;

use agent_client_protocol_schema::{
    ContentBlock, EmbeddedResource, EmbeddedResourceResource, ImageContent, ResourceLink,
    TextContent, TextResourceContents,
};

use crate::error::BoxError;
use crate::llm::MessageContent;
use crate::session::TurnError;

pub(super) fn content_block_to_message_content(
    cb: ContentBlock,
) -> Result<Vec<MessageContent>, TurnError> {
    match cb {
        ContentBlock::Text(TextContent { text, .. }) => Ok(vec![MessageContent::Text { text }]),
        ContentBlock::Image(image) => Ok(vec![image_content_to_message_content(image)?]),
        ContentBlock::ResourceLink(link) => Ok(vec![MessageContent::Text {
            text: resource_link_to_text(link),
        }]),
        ContentBlock::Resource(resource) => resource_to_message_content(resource),
        ContentBlock::Audio(_) => Err(invalid_prompt_content(
            "ACP audio content is not supported yet",
        )),
        _ => Err(invalid_prompt_content(
            "unsupported ACP content block variant",
        )),
    }
}

fn image_content_to_message_content(image: ImageContent) -> Result<MessageContent, TurnError> {
    let data = if image.data.is_empty() {
        let Some(uri) = image.uri else {
            return Err(invalid_prompt_content(
                "ACP image content must include data or uri",
            ));
        };
        crate::llm::ImageData::Url { url: uri }
    } else {
        crate::llm::ImageData::Base64 {
            encoded: image.data,
        }
    };

    Ok(MessageContent::Image {
        mime: image.mime_type,
        data,
    })
}

fn resource_to_message_content(
    resource: EmbeddedResource,
) -> Result<Vec<MessageContent>, TurnError> {
    match resource.resource {
        EmbeddedResourceResource::TextResourceContents(text) => Ok(vec![MessageContent::Text {
            text: text_resource_to_text(text),
        }]),
        EmbeddedResourceResource::BlobResourceContents(blob) => {
            Err(invalid_prompt_content(&format!(
                "embedded binary resource is not supported yet: {}",
                blob.uri
            )))
        }
        _ => Err(invalid_prompt_content(
            "unsupported embedded resource variant",
        )),
    }
}

fn resource_link_to_text(link: ResourceLink) -> String {
    let mut lines = vec![format!("resource: {}", link.name)];
    if let Some(title) = link.title {
        lines.push(format!("title: {title}"));
    }
    if let Some(description) = link.description {
        lines.push(format!("description: {description}"));
    }
    if let Some(mime_type) = link.mime_type {
        lines.push(format!("mime_type: {mime_type}"));
    }
    if let Some(size) = link.size {
        lines.push(format!("size: {size}"));
    }
    lines.push(format!("uri: {}", link.uri));
    lines.join("\n")
}

fn text_resource_to_text(resource: TextResourceContents) -> String {
    let mut text = format!("resource: {}", resource.uri);
    if let Some(mime_type) = resource.mime_type {
        text.push_str(&format!("\nmime_type: {mime_type}"));
    }
    text.push_str("\n\n");
    text.push_str(&resource.text);
    text
}

pub(super) fn invalid_prompt_content(message: &str) -> TurnError {
    TurnError::Internal(BoxError::new(io::Error::other(message)))
}
