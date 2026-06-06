use std::collections::BTreeSet;
use std::sync::Mutex;

use crate::llm::{CompletionRequest, Message, MessageContent};

const HASH_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const HASH_PRIME: u64 = 0x0000_0001_0000_01b3;

/// 相邻 LLM 请求稳定性审计器。
///
/// 仅输出结构化 tracing 日志，不改变请求内容。
#[derive(Default)]
pub(crate) struct RequestAuditTracker {
    previous: Mutex<Option<RequestAuditSnapshot>>,
}

impl RequestAuditTracker {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn record(&self, req: &CompletionRequest) {
        let snapshot = RequestAuditSnapshot::from_request(req);
        let mut guard = self
            .previous
            .lock()
            .expect("RequestAuditTracker mutex poisoned");
        let previous = guard.replace(snapshot.clone());
        let delta = RequestAuditDelta::between(previous.as_ref(), &snapshot);
        tracing::debug!(
            target: "defect::cache_audit",
            model = %snapshot.model,
            system_hash = %format_hash(snapshot.system_hash),
            messages_hash = %format_hash(snapshot.messages_hash),
            tools_hash = %format_hash(snapshot.tools_hash),
            user_messages = snapshot.user_messages,
            assistant_messages = snapshot.assistant_messages,
            text_blocks = snapshot.text_blocks,
            thinking_blocks = snapshot.thinking_blocks,
            tool_use_blocks = snapshot.tool_use_blocks,
            tool_result_blocks = snapshot.tool_result_blocks,
            total_text_bytes = snapshot.total_text_bytes,
            total_tool_result_bytes = snapshot.total_tool_result_bytes,
            tool_names = %snapshot.tool_names_csv(),
            changed = %delta.changed_summary(),
            changed_count = delta.changed_count(),
            previous_present = previous.is_some(),
            "llm request cache audit"
        );
    }
}

#[derive(Debug, Clone)]
struct RequestAuditSnapshot {
    model: String,
    system_hash: u64,
    messages_hash: u64,
    tools_hash: u64,
    user_messages: usize,
    assistant_messages: usize,
    text_blocks: usize,
    thinking_blocks: usize,
    tool_use_blocks: usize,
    tool_result_blocks: usize,
    total_text_bytes: usize,
    total_tool_result_bytes: usize,
    tool_names: BTreeSet<String>,
}

impl RequestAuditSnapshot {
    fn from_request(req: &CompletionRequest) -> Self {
        let mut message_stats = MessageStats::default();
        for message in &req.messages {
            message_stats.observe(message);
        }

        let tool_names = req
            .tools
            .iter()
            .map(|tool| tool.name.clone())
            .collect::<BTreeSet<_>>();

        Self {
            model: req.model.clone(),
            system_hash: hash_option_str(req.system.as_deref()),
            messages_hash: hash_json(&req.messages),
            tools_hash: hash_json(&req.tools),
            user_messages: message_stats.user_messages,
            assistant_messages: message_stats.assistant_messages,
            text_blocks: message_stats.text_blocks,
            thinking_blocks: message_stats.thinking_blocks,
            tool_use_blocks: message_stats.tool_use_blocks,
            tool_result_blocks: message_stats.tool_result_blocks,
            total_text_bytes: message_stats.total_text_bytes,
            total_tool_result_bytes: message_stats.total_tool_result_bytes,
            tool_names,
        }
    }

    fn tool_names_csv(&self) -> String {
        self.tool_names
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join(",")
    }
}

#[derive(Default)]
struct MessageStats {
    user_messages: usize,
    assistant_messages: usize,
    text_blocks: usize,
    thinking_blocks: usize,
    tool_use_blocks: usize,
    tool_result_blocks: usize,
    total_text_bytes: usize,
    total_tool_result_bytes: usize,
}

impl MessageStats {
    fn observe(&mut self, message: &Message) {
        match message.role {
            crate::llm::Role::User => self.user_messages += 1,
            crate::llm::Role::Assistant => self.assistant_messages += 1,
        }

        for content in message.content.iter() {
            match content {
                MessageContent::Text { text } => {
                    self.text_blocks += 1;
                    self.total_text_bytes += text.len();
                }
                MessageContent::Thinking { text, .. } => {
                    self.thinking_blocks += 1;
                    self.total_text_bytes += text.len();
                }
                MessageContent::ToolUse { .. } => {
                    self.tool_use_blocks += 1;
                }
                MessageContent::ToolResult { output, .. } => {
                    self.tool_result_blocks += 1;
                    self.total_tool_result_bytes += tool_result_bytes(output);
                }
                MessageContent::Image { .. } => {}
                MessageContent::ProviderActivity { .. } => {}
            }
        }
    }
}

fn tool_result_bytes(output: &crate::llm::ToolResultBody) -> usize {
    use crate::llm::{ImageData, ToolResultBody, ToolResultContent};
    match output {
        ToolResultBody::Text { text } => text.len(),
        ToolResultBody::Json { value } => value.to_string().len(),
        ToolResultBody::Content { blocks } => blocks
            .iter()
            .map(|b| match b {
                ToolResultContent::Text { text } => text.len(),
                ToolResultContent::Image { data, .. } => match data {
                    ImageData::Base64 { encoded } => encoded.len(),
                    ImageData::Url { url } => url.len(),
                },
            })
            .sum(),
    }
}

struct RequestAuditDelta {
    changed: Vec<&'static str>,
}

impl RequestAuditDelta {
    fn between(previous: Option<&RequestAuditSnapshot>, current: &RequestAuditSnapshot) -> Self {
        let Some(previous) = previous else {
            return Self {
                changed: vec!["initial_request"],
            };
        };

        let mut changed = Vec::new();
        if previous.model != current.model {
            changed.push("model");
        }
        if previous.system_hash != current.system_hash {
            changed.push("system");
        }
        if previous.messages_hash != current.messages_hash {
            changed.push("messages");
        }
        if previous.tools_hash != current.tools_hash {
            changed.push("tools");
        }
        if previous.tool_names != current.tool_names {
            changed.push("tool_names");
        }
        if previous.user_messages != current.user_messages {
            changed.push("user_message_count");
        }
        if previous.assistant_messages != current.assistant_messages {
            changed.push("assistant_message_count");
        }
        if previous.text_blocks != current.text_blocks {
            changed.push("text_blocks");
        }
        if previous.thinking_blocks != current.thinking_blocks {
            changed.push("thinking_blocks");
        }
        if previous.tool_use_blocks != current.tool_use_blocks {
            changed.push("tool_use_blocks");
        }
        if previous.tool_result_blocks != current.tool_result_blocks {
            changed.push("tool_result_blocks");
        }
        if previous.total_text_bytes != current.total_text_bytes {
            changed.push("text_bytes");
        }
        if previous.total_tool_result_bytes != current.total_tool_result_bytes {
            changed.push("tool_result_bytes");
        }
        if changed.is_empty() {
            changed.push("none");
        }
        Self { changed }
    }

    fn changed_summary(&self) -> String {
        self.changed.join(",")
    }

    fn changed_count(&self) -> usize {
        if self.changed.as_slice() == ["none"] {
            0
        } else {
            self.changed.len()
        }
    }
}

fn hash_option_str(value: Option<&str>) -> u64 {
    let mut hasher = StableHasher::new();
    if let Some(value) = value {
        hasher.write_str(value);
    }
    hasher.finish()
}

fn hash_json<T>(value: &T) -> u64
where
    T: serde::Serialize,
{
    let Ok(bytes) = serde_json::to_vec(value) else {
        return 0;
    };
    let mut hasher = StableHasher::new();
    hasher.write_bytes(&bytes);
    hasher.finish()
}

fn format_hash(value: u64) -> String {
    format!("{value:016x}")
}

struct StableHasher {
    state: u64,
}

impl StableHasher {
    fn new() -> Self {
        Self {
            state: HASH_OFFSET_BASIS,
        }
    }

    fn write_str(&mut self, value: &str) {
        self.write_bytes(value.as_bytes());
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.state ^= u64::from(*byte);
            self.state = self.state.wrapping_mul(HASH_PRIME);
        }
        self.state ^= u64::from(b'\n');
        self.state = self.state.wrapping_mul(HASH_PRIME);
    }

    fn finish(self) -> u64 {
        self.state
    }
}

#[cfg(test)]
mod tests;
