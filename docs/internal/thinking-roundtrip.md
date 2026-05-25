# Thinking / Reasoning 多轮 round-trip

`defect` 把 LLM 的"思考过程"统一抽象成 [`ProviderChunk::ThinkingDelta`] /
[`ProviderChunk::ThinkingSignature`]（详见 [`llm-trait.md`](./llm-trait.md)
§1.1）。**入流**这条线已经通了——Anthropic `thinking_delta` /
`signature_delta` 与 OpenAI 兼容厂商的 `delta.reasoning_content` 都正确解码。

本文补的是**出流**——把上一轮 assistant 回复的 thinking 内容回写到下一轮
请求里。这条路当前是断的：`assistant_message()`
（`crates/agent/src/session/turn.rs`）只把文本和工具调用塞进 history，
`thinking_buf` / `thinking_signature` 一律丢弃。后果是：

- **Anthropic extended thinking + tool_use 多轮**：上一轮 assistant 回了
  thinking + tool_use；这一轮把 tool_result 送回模型时，必须把上一轮的
  `thinking` block（含 `signature`）原样回放，否则 Anthropic 拒收
  （400 `unexpected_thinking_signature_mismatch`）。
- **DeepSeek v4-pro thinking + tool_use 多轮**：上一轮 assistant 出了
  `reasoning_content` + `tool_calls`；这一轮请求里这条 assistant
  message 必须带 `reasoning_content` 字段，否则 DeepSeek 直接 400：
  > The `reasoning_content` in the thinking mode must be passed back to the API.
- **DeepSeek-R1**（`deepseek-reasoner`）官方文档**禁止**回放
  `reasoning_content`——回放反而 400。所以"回放 vs 不回放"是
  per-provider / per-model 的开关，不是普世真理。

## 1. 设计原则

1. **Thinking 是 assistant 消息的一阶 content**——和 text / tool_use 平起
   平坐。在内部 `MessageContent` 上加一个 `Thinking` variant，而不是
   藏在 `Message` 之外的旁路状态。Anthropic 的 wire 模型（`content` 数组
   按顺序排 thinking / text / tool_use 三种 block）已经把它当成一阶
   content；我们的内部模型同步对齐，不发明第二种表达。
2. **Thinking 内容是 opaque 的**——主循环不读、不改、不裁剪 thinking
   文本。它对主循环唯一的语义是"原样回放给同一 provider 的下一轮"。
   裁剪/压缩历史时，整块 `Thinking` 要一起丢，不能只丢文本留 signature
   也不能反过来。
3. **是否回放由 provider 决定，不由 thinking 字段本身决定**——
   `Capabilities` 上加一位 `thinking_echo`，protocol 层 encode 时按这位
   做差异：Anthropic / DeepSeek-v4-pro 设 `Required`，DeepSeek-R1 / OpenAI
   官方 o1 / o3 设 `Forbidden`，未配置默认 `Forbidden`（保守）。
4. **Signature 与 thinking text 同生共死**——不能只回放文本不带
   signature（Anthropic 直接拒），也不能只发 signature（无意义）。
   `MessageContent::Thinking` 的 `signature` 是 `Option<String>`，但**只
   有 `signature` 缺失而 text 在的情况**才出现在 DeepSeek 这种没有
   signature 概念的 provider 上。

## 2. 内部数据模型

`crates/agent/src/llm/request.rs` 的 `MessageContent` 加一支：

```rust
#[non_exhaustive]
pub enum MessageContent {
    Text { text: String },
    /// 上一轮模型产出的思考链。仅出现在 [`Role::Assistant`] 消息里。
    ///
    /// `signature` 是 Anthropic extended thinking 的防伪签名：必须与
    /// 文本同进同出。DeepSeek-v4-pro 等纯文本 echo 的 provider 这里
    /// 为 [`None`]。
    Thinking {
        text: String,
        signature: Option<String>,
    },
    ToolUse { id: String, name: String, args: serde_json::Value },
    ToolResult { tool_use_id: String, output: ToolResultBody, is_error: bool },
    Image { mime: String, data: ImageData },
}
```

构造时机：`session/turn.rs` 的 `assistant_message(outcome: &DrainOutcome)`
在 `text_buf` / `tool_uses` 之外加一段：

```rust
if !outcome.thinking_buf.is_empty() || outcome.thinking_signature.is_some() {
    content.push(MessageContent::Thinking {
        text: outcome.thinking_buf.clone(),
        signature: outcome.thinking_signature.clone(),
    });
}
```

**顺序约束**：`Thinking` 必须排在同条 assistant message 内 `Text` /
`ToolUse` **之前**——Anthropic 的 wire 顺序是 `thinking → text → tool_use`，
顺序错位会被拒。OpenAI 兼容侧不在乎顺序（reasoning_content 是 message
顶级字段，不进 content 数组），但保持同样顺序便于阅读。

## 3. Capabilities 扩展

`crates/agent/src/llm/capabilities.rs` 的 `Capabilities` 新增字段：

```rust
pub struct Capabilities {
    // ... 既有字段 ...

    /// thinking 内容回放策略。
    ///
    /// `Required` —— 上一轮 assistant 的 thinking 必须出现在下一轮
    /// 请求里（Anthropic extended thinking、DeepSeek-v4-pro）。
    /// `Forbidden` —— 回放会被服务端拒（DeepSeek-R1、OpenAI o1 / o3
    /// 官方）。
    /// `Optional` —— 服务端容忍两种行为（暂未观察到此类）。
    pub thinking_echo: ThinkingEcho,
}

#[non_exhaustive]
pub enum ThinkingEcho {
    Forbidden,
    Required,
    Optional,
}

impl Default for ThinkingEcho {
    fn default() -> Self { ThinkingEcho::Forbidden }
}
```

`thinking_echo` 与 `thinking: FeatureSupport` 是两件事：

- `thinking` 回答"模型**会不会**产 thinking 内容"——影响 codec 是否
  解码 `ThinkingDelta`、`Capabilities.thinking_echo` 是否生效。
- `thinking_echo` 回答"产了 thinking 内容**该不该**回放"——影响 encode
  路径。

`ModelCapabilityOverrides` 同样加 `thinking_echo: Option<ThinkingEcho>`，
让 DeepSeek 同一 provider 下 `deepseek-reasoner`（Forbidden）和
`deepseek-v4-pro`（Required）能分别表达。

## 4. Provider 层 encode 规则

### 4.1 Anthropic Messages

protocol 层 `encode_content` 加一支：

```rust
MessageContent::Thinking { text, signature } => {
    wire::ContentBlockParam::ThinkingBlockParam(wire::ThinkingBlockParam {
        thinking: text.clone(),
        signature: signature.clone().unwrap_or_default(),
        r#type: wire::ThinkingBlockParamType::Thinking,
    })
}
```

注：

- Anthropic wire 上 `signature` 是 required 字段，缺失等价于"伪造"，
  服务端拒。所以走 Anthropic 这条 encode 路径而 `signature: None` 时，
  上层应在更早处（按 `thinking_echo == Required` + provider 是 Anthropic）
  保证 signature 已被填充——若仍为 None，**不要回放**，整块跳过。
- `thinking_echo: Forbidden` 时整块 `MessageContent::Thinking` 不写入
  wire（哪怕 provider 是 Anthropic）。这种状态在 Anthropic 下不会真的
  出现，但保留兜底以防错配。

### 4.2 OpenAI 兼容（DeepSeek 等）

OpenAI 官方 wire schema 没有 `reasoning_content` 字段。两条路径选其一：

**路径 A（采纳）**：在 `scripts/llm-codegen/src/openai_strip.rs` 增补
patch，给 `ChatCompletionRequestAssistantMessage` 加 `reasoning_content:
Option<String>` 字段。codegen 后 `wire::ChatCompletionRequestAssistantMessage`
带这个字段，protocol 层 encode 时按 `thinking_echo` 决定填不填。

**路径 B（拒）**：encode 出去后用 `serde_json::Value` 后处理插字段。
拒因：把 wire 类型重新拆回 untyped 是退步，且未来加更多 quirk
（`prompt_cache` 等）会越来越糟。

protocol 层 `encode_assistant_message_into` 改造：

```rust
let mut text_parts: Vec<String> = Vec::new();
let mut tool_calls = Vec::new();
let mut reasoning_text = String::new();   // ← 新增

for c in &m.content {
    match c {
        MessageContent::Text { text } => text_parts.push(text.clone()),
        MessageContent::Thinking { text, .. } => reasoning_text.push_str(text),
        MessageContent::ToolUse { id, name, args } => { /* 同前 */ }
        _ => {}
    }
}

let reasoning_content = match (echo_mode, reasoning_text.is_empty()) {
    (ThinkingEcho::Required, false) => Some(reasoning_text),
    _ => None,
};

out.push(wire::ChatCompletionRequestAssistantMessage {
    content,
    tool_calls,
    reasoning_content,         // ← 新增字段（codegen patch 后存在）
    // ...
});
```

`echo_mode` 由 protocol 层从 `ProviderConfig` 读取——构造期一次性配死，
不每条消息判别。

`MessageContent::Thinking` 的 `signature` 字段在 OpenAI 路径上**忽略**
（DeepSeek 不要、OpenAI 自己也不要）。

### 4.3 厂商配置入口

`crates/llm/src/provider/openai.rs` 的 `OpenAiConfig` 加：

```rust
pub struct OpenAiConfig {
    // ... 既有 base_url / api_key 等 ...
    pub default_thinking_echo: ThinkingEcho,
    pub model_thinking_echo: HashMap<String, ThinkingEcho>,
}
```

DeepSeek provider（`provider/deepseek.rs`）在 `DeepSeekProvider::new`
里把：

- `default_thinking_echo = ThinkingEcho::Forbidden`（兼容老 R1）
- `model_thinking_echo` 中 `"deepseek-v4-pro"` → `Required`
  （往后新模型默认走 Forbidden 直到验证通过）

写硬编码表的位置在 `provider/deepseek.rs::HARDCODED_MODELS`，与现有
`context_window` 等一并维护。

Anthropic provider 全模型 `thinking_echo = Required`——extended
thinking 只要开启就要回放。

## 5. 主循环视角

主循环对 thinking 的语义是"opaque token 序列，按 provider 要求回放"：

- **取消**：thinking 流到一半被取消，已累计的 `thinking_buf` 不写进
  history（这条对话失败，没有"半截 thinking"的概念）。
- **裁剪**：context 压缩时，`MessageContent::Thinking` 整体剔除——
  如果保留要保留整块；不能拆 signature/text。理由：剔除是"这条对话
  我们不再回放"，部分剔除等于伪造。
- **跨 provider 切换**：从 Anthropic 切到 OpenAI 走的对话，旧的
  `MessageContent::Thinking` 还留在 history。OpenAI 路径按 §4.2 处理
  （Forbidden 默认就丢弃；Required 把文本写进 reasoning_content，
  signature 弃掉）。**反向**（OpenAI → Anthropic 切回）时 thinking
  没有 signature，Anthropic 路径会跳过整块——这是预期行为，
  Anthropic 拒收无 signature thinking。

## 6. 测试策略

最低要求：

1. **协议层多轮 round-trip 测试**（`crates/llm/src/protocol/openai_chat/tests.rs`）：
   - 构造一个含 `MessageContent::Thinking { text: "abc", signature: None }`
     的 history，echo_mode = Required，断言 wire body 含
     `assistant.reasoning_content == "abc"`。
   - 同样输入但 echo_mode = Forbidden，断言 wire body **不含**
     `reasoning_content` 字段。
2. **协议层多轮 round-trip 测试**（`crates/llm/src/protocol/anthropic_messages/tests.rs`）：
   - history 含 `Thinking { text, signature: Some(s) }`，断言 wire body
     的 `messages[i].content` 第一个 block 是
     `{ type: "thinking", thinking: text, signature: s }`。
   - signature 为 None 时整块被跳过。
3. **DeepSeek smoke** (`crates/llm/examples/deepseek_smoke.rs`)：
   - 加 `scenario_thinking_tool_multi_turn` —— 用 `deepseek-v4-pro`
     先让模型调一个工具，再把工具结果送回，验证第二轮请求**带**
     `reasoning_content`，得到合法响应（不是上文那条 400）。

## 7. 与现有文档的关系

- [`llm-trait.md`](./llm-trait.md) §1.1 已经定义 `ThinkingDelta` /
  `ThinkingSignature`，本文是对"出流"的补充，不修订入流约束。
- [`llm-trait.md`](./llm-trait.md) §3 `MessageContent` 列表本应同步
  加 `Thinking` variant；改动落地时一并修。
- [`llm-trait.md`](./llm-trait.md) §5 `Capabilities` 与
  `ModelCapabilityOverrides` 字段表本应同步加 `thinking_echo`；
  改动落地时一并修。
- [`llm-openai.md`](../outbound/llm-openai.md) §1.2 上游 patches 列表
  本应加上"reasoning_content 字段补丁"；落地时一并修
  `scripts/llm-codegen/src/openai_strip.rs`。
- [`llm-openai.md`](../outbound/llm-openai.md) §5.4 已经写过
  "reasoning_content 是兼容厂商扩展点"，但只覆盖入流；本文补出流。

## 8. 演进口子

- **OpenAI 官方 o1 / o3 演进**：当前 OpenAI 官方不通过
  `delta.reasoning_content` 暴露 thinking——它把 thinking 扣在服务端
  仅算 token。若官方未来公开这部分内容（路线图里有，参见 OpenAI
  Responses API），新增 `Capabilities.thinking_echo: Required`，按
  本文路径自然生效。
- **Anthropic redacted_thinking**：Anthropic 在某些安全过滤后会发
  `redacted_thinking` block 替代 `thinking` block，含 signature 但
  不含明文。当前我们解码这个 block 时**已经**走 ThinkingDelta（文本为空）
  + ThinkingSignature 路径——等于把它还原成空文本 thinking + 真
  signature。Anthropic 接受空 thinking + 有 signature 的 echo，所以
  本文规则零修改即覆盖。
- **新的回放策略**：若出现"必须回放但带额外字段"的 provider
  （如假想中的 reasoning_summary），扩 `MessageContent::Thinking`
  时加可选字段，不破坏既有 variant。

[`ProviderChunk::ThinkingDelta`]: ../../crates/agent/src/llm/chunk.rs
[`ProviderChunk::ThinkingSignature`]: ../../crates/agent/src/llm/chunk.rs
