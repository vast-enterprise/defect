# OpenAI 兼容 provider 落地

`provider/openai.rs` 实现 [`LlmProvider`](../internal/llm-trait.md)，对接 OpenAI 官方 + 所有遵循 Chat Completions 协议的兼容服务（DeepSeek、Qwen、Together、本地 vllm 等）。一个文件一个 provider，**不**走前缀路由——通过 `OpenAiConfig::base_url` 切目标。

设计前提见 [`docs/outbound/llm.md`](./llm.md) §1 与 [`docs/internal/llm-trait.md`](../internal/llm-trait.md) §4.2。本文聚焦与 [`llm-anthropic.md`](./llm-anthropic.md) 的**差异**——客户端方案、codegen、auth、错误映射的总体形状那边已经写过，不在这里复述。

---

## 1. OAS 来源

不像 Anthropic 没有官方 OAS——OpenAI 在 [`openai/openai-openapi`](https://github.com/openai/openai-openapi) 公开维护一份**官方**规范。但全量约 13k 行、覆盖 chat / responses / images / audio / files / batches / fine-tuning / assistants / vector_stores / threads / runs / fine_tuning / uploads / 还在加，完全用不到。

策略：**fork 到 `crates/llm/oas/openai.yaml`，本地裁剪**。每次 OpenAI 上游加 chat 字段时手动 sync 这一段。

裁剪后保留：

| 用途 | 方法 | 路径 | 对应 `LlmProvider` 方法 |
| --- | --- | --- | --- |
| 流式 + 非流式生成 | POST | `/v1/chat/completions` | `complete` |
| 列出模型 | GET | `/v1/models` | `list_models` |

`/v1/models/{model}` 不要（同 [`llm-anthropic.md`](./llm-anthropic.md) §2.1）。

`/v1/responses` 在 OpenAI 那边是新一代接口，**v0 不做**——加 `responses` 协议要新一份 codec，是独立工作。

### 1.1 schema 裁剪范围

只留 chat 相关：

| 留 | 删 |
| --- | --- |
| `CreateChatCompletionRequest` 全字段（OpenAI Chat 接口字段不算太多） | `assistants` / `audio` / `images` / `embeddings` / `files` / `batches` / `vector_stores` / `responses` / `fine_tuning` / `uploads` 整支 schema 树 |
| `ChatCompletionRequestMessage`（user / assistant / tool / system / developer 五支） | `MessageObject`（assistants 那个 thread message） |
| `ChatCompletionStreamResponse` + 嵌套 `Choice` / `Delta` / `ToolCallChunk` | 非 chat 的 response schema |
| `Model`（`/v1/models` 返回的元素） | `FineTuningJob` 等无关 |

裁剪与 OpenAI 上游 sync 的工具：在 `scripts/llm-codegen` 里加一个 `openai-strip` 子命令，跑 `oas3` 只 keep `paths.{/v1/chat/completions, /v1/models}` + 它们引用的 `$ref` 闭包，输出到 `oas/openai.yaml`。这条命令不每次跑，**只在上游同步时人工触发**。

### 1.2 上游 patches

复用 toac 的 OpenAI example 在 `build.rs` 里做的两条 patches（[来源](https://github.com/4t145/tower-openapi-client/blob/master/examples/openai/build.rs)）：

1. **`9223372036854776000` → `i64::MAX`**：`seed` 字段的 `minimum`/`maximum` 上游写超过 `i64::MAX` 193，`oas3` parser 拒收。
2. **`exclusiveMinimum: true` / `exclusiveMaximum: true` 行删除**：OAS 3.1 / JSON Schema 2020-12 期望数字而非 bool，上游混着写。

这两条迁到我们的 `scripts/llm-codegen` 里，每次 sync 时 strip + patch 一次性出 `oas/openai.yaml`。

我们另外在 strip 阶段注入一条非标字段：

3. **`ChatCompletionRequestAssistantMessage.reasoning_content`**（可选 nullable string）：OpenAI 官方 wire schema 不含此字段，但 DeepSeek-v4-pro / SiliconFlow 等兼容厂商在 thinking 模式下要求把上一轮 reasoning 文本**回放**给服务端（否则 400 `reasoning_content must be passed back to the API`）。strip 把它挂进 schema 后，protocol 层 encode 时按 [`Capabilities.thinking_echo`](../internal/llm-trait.md#5-capabilities) 决定是否填值；OpenAI 官方收到额外字段会忽略，不影响现有路径。实现位置：`scripts/llm-codegen/src/openai_strip.rs::inject_assistant_reasoning_content`，详见 [`thinking-roundtrip.md`](../internal/thinking-roundtrip.md) §4.2。

## 2. 鉴权

```
Authorization: Bearer ${OPENAI_API_KEY}
OpenAI-Organization: ${OPENAI_ORG}      # optional
OpenAI-Project: ${OPENAI_PROJECT}       # optional
```

OAS 里 `securitySchemes.ApiKeyAuth` 是 HTTP Bearer，codegen 出来的 `AuthConfig::builder().api_key_auth(token)` 直接把 `Bearer ` 前缀加好。`OpenAI-Organization` / `OpenAI-Project` 走 OAS（OpenAI 官方 spec 里有），codegen 出对应字段；用户没配就发空 header（toac 的 codegen 对 `None` 字段省略不发）。

兼容厂商配置（DeepSeek / Together）通过 `OpenAiConfig::base_url` 改终点；同一个 provider 实例承载——**没必要每家厂商一个 provider 文件**，差异只是 base_url + 模型清单 + 末帧字段处理（reasoning_content）。

## 3. transport 装配

形态与 Anthropic 一致（[`llm-anthropic.md`](./llm-anthropic.md) §4），不再重述。差别：

- `base_url` 默认 `https://api.openai.com/v1`，可用 `OPENAI_BASE_URL` 覆盖。
- 兼容厂商常用 base_url：`https://api.deepseek.com`、`https://api.together.xyz/v1`、`http://localhost:8000/v1`（本地 vllm）。

## 4. `complete` 实现

骨架与 [`llm-anthropic.md`](./llm-anthropic.md) §5 相同，差异在请求构造与 SSE 协商：

```rust
let mut request = wire::operations::chat::completions::post::Request {
    body: encode_request(&req)?,                        // body.stream = true
};
request = request.with_accept(http::HeaderValue::from_static("text/event-stream"));

let resp = self.client.clone().call(request).await?;
let stream = match resp {
    Response::Status200Sse(s) => s,
    Response::Status200Json(_) => return Err(ProtocolViolation { ... }),
};
Ok(Box::pin(decode_stream(stream, cancel)))
```

注意 OpenAI Chat 的 200 同时声明 JSON + SSE 两支 content-type（同一个 `200`），跟 [toac OpenAI example](https://github.com/4t145/tower-openapi-client/blob/master/examples/openai/src/main.rs) 一样靠 `Accept` header 选分支。

## 5. 协议层：差异表

`crates/llm/src/protocol/openai_chat.rs` 暴露 `encode_request` / `decode_stream`。

### 5.1 字段映射差异（vs Anthropic）

| `CompletionRequest` 字段 | OpenAI wire | 关键差异 |
| --- | --- | --- |
| `system: Option<String>` | 第一条 `messages[0]` `role=system` `content=text` | **不是 top-level**——Anthropic 是 top-level `system: String`，OpenAI 必须打进 messages 数组首位 |
| `messages` | `messages` | tool_use / tool_result 拆开成独立 message，见下 |
| `tools: Vec<ToolSchema>` | `tools: [{ type: "function", function: { name, description, parameters: input_schema } }]` | 多一层 `function` wrap |
| `tool_choice: Auto` | `"auto"` 字符串 | **字符串而非 object** |
| `tool_choice: Required` | `"required"` 字符串 | |
| `tool_choice: Named(s)` | `{ type: "function", function: { name: s } }` | |
| `tool_choice: None` | `"none"` 字符串 | OpenAI 真有这个语义，Anthropic 没有 |
| `sampling.max_tokens` | `max_tokens` | 可选；OpenAI 默认按模型 |
| `sampling.thinking: Enabled` | `reasoning_effort: "medium"` 或厂商扩展 | OpenAI 官方仅 o1/o3 系列认这个字段；DeepSeek 走 `reasoning_content` 不通过 request 字段 |
| `sampling.thinking: Disabled` | 不发字段 | |
| `stop_sequences: Vec<String>` | `stop: array<string>` 或 `string` | 1 个时 OpenAI 接受 string，array 也接受；统一发 array |

### 5.2 messages 数组结构差异

Anthropic 把 tool_use（assistant 出的）和 tool_result（user 回的）当作 message **content 数组里的 block**；OpenAI 拆成**独立 message**。映射规则：

```
内部 Message { role: Assistant, content: [Text("…"), ToolUse{id,name,args}] }
  ↓ 编码 OpenAI
{ role: "assistant", content: "…", tool_calls: [{ id, type: "function", function: { name, arguments: serde_json::to_string(&args) } }] }

内部 Message { role: User, content: [ToolResult{tool_use_id, output}] }
  ↓ 编码 OpenAI
{ role: "tool", tool_call_id: tool_use_id, content: serialize(output) }
```

要点：

- **同一条内部 Assistant message 携带 ToolUse 时，会被同时**：保留 `content` 文本 + 出 `tool_calls` 数组。两者并存。
- **`ToolResult` 必须独占一条 message**（OpenAI 的 `role: tool`），同一内部 User message 里有多个 ToolResult 时拆成多条。
- 如果同一内部 `User` message 同时带 `ToolResult` 和普通用户文本，**编码顺序必须先发所有 `role: tool`，再发 `role: user`**，否则 OpenAI 兼容服务会拒绝这轮 history。
- **`ToolResultBody::Json` 必须 stringify**——OpenAI tool message 的 `content` 只接受 string，与 Anthropic（接受 content block 数组）不同。

### 5.3 stream 解码状态机

输入：`SseEventStream` 上每条事件 `data:` 行是 `ChatCompletionChunk` JSON。OpenAI 没用 `event:` 名字 tag，所有 chunk 都是匿名事件（toac 的 SSE codec 把 `event:` 缺失视作默认事件）；流末以 `data: [DONE]` 收尾。

状态机维持：

```rust
struct DecoderState {
    started: bool,
    /// tool_calls 索引 → { tool_use_id, accumulated_args }（防御性 args
    /// 累计仅用于决定 ToolUseEnd 时机；ArgsDelta 直接 yield 不等齐）。
    tool_calls: HashMap<u32, ToolCallSlot>,
    /// 累计 finish_reason，用于 Stop 推断。
    finish_reason: Option<wire::FinishReason>,
}

struct ToolCallSlot {
    tool_use_id: String,
    name_emitted: bool,
}
```

事件处理：

| chunk 字段 | 状态 / 动作 | 产出 chunk |
| --- | --- | --- |
| 第 1 条 chunk（任意 `delta`） | `started=true` | `MessageStart{id=chunk.id, model=chunk.model}` |
| `delta.content: Some(text)` | – | `TextDelta{text}` |
| `delta.reasoning_content: Some(text)` | – | `ThinkingDelta{text}`（DeepSeek / o1 扩展，OpenAI 官方未来 o1 也走这个路） |
| `delta.tool_calls[i]` 首次出现，含 `id` 与 `function.name` | `tool_calls[i] = ToolCallSlot{ id, name_emitted: true }` | `ToolUseStart{id, name}` |
| `delta.tool_calls[i].function.arguments: Some(s)` | 查 `tool_calls[i].id` | `ToolUseArgsDelta{id, fragment: s}` |
| `delta.tool_calls[i]` 出现但缺 `id` / `name`（后续片段） | – | 仅 `ToolUseArgsDelta` |
| `choice.finish_reason: stop` | – | 先对所有未关 `tool_calls[*]` emit `ToolUseEnd{id}`（防御）；再 `Stop{EndTurn}` |
| `choice.finish_reason: length` | – | 同上关 + `Stop{MaxTokens}` |
| `choice.finish_reason: tool_calls` | – | 对所有 `tool_calls[*]` emit `ToolUseEnd{id}`，再 `Stop{ToolUse}` |
| `choice.finish_reason: content_filter` | – | `Stop{Refusal}` |
| 末帧（`stream_options.include_usage=true` 时存在）`usage: { ... }` | – | `Usage{input_tokens, output_tokens}` |
| `data: [DONE]` | – | （流终结） |

要点：

- **`stream_options.include_usage=true` 协议层强制开**——内部 `Usage` 字段需要末帧才能填，关闭就拿不到 token 计数。
- **tool_calls 没有显式 end**——OpenAI 不发 per-tool `stop`，必须靠 `finish_reason` 触发"统一关闭所有未关闭的 tool_use"。
- **`finish_reason: tool_calls` 与 `delta.tool_calls` 共存**——OpenAI 在最后一条带 finish 的 chunk 里仍可能带 `delta.tool_calls` 末段。先处理 delta，再处理 finish。
- **首 chunk 仅 `delta.role=assistant` 没有 content**——`MessageStart` 触发条件是"第 1 条 chunk"，不是"首次出现 content"。

### 5.4 reasoning_content 的兼容厂商扩展点

`reasoning_content` 是 OpenAI Chat 协议族里**双向**的兼容厂商扩展——入流（response）和出流（request）都要处理。

**入流（decode）**：OpenAI 官方不发 `reasoning_content`；DeepSeek / SiliconFlow / 月之暗面等都用这个名字。我们**直接接受**——codec 看见就 emit `ThinkingDelta`，不报错。OpenAI 自己往 o1 演进上不再用 `delta.content` 而用 `delta.reasoning_content` 时，零修改。

**出流（encode）**：上一轮 `MessageContent::Thinking { text, .. }` 是否回放进 `assistant.reasoning_content` 由 [`Capabilities.thinking_echo`](../internal/llm-trait.md#5-capabilities) 决定：

| `thinking_echo` | encode 行为 |
| --- | --- |
| `Required` | 把所有 `Thinking { text }` 的文本拼起来写进 `reasoning_content`（DeepSeek v4 系列这条是必需的） |
| `Forbidden` | 整块 `Thinking` 跳过，不写 `reasoning_content`（OpenAI 官方 / 默认） |
| `Optional` | 同 `Required` |

`echo_mode` 由 protocol 层从 `OpenAiConfig` 读取一次性配死，不每条消息判别。`signature` 字段在 OpenAI 路径上忽略（DeepSeek 不要、OpenAI 自己也不要）。详细设计见 [`thinking-roundtrip.md`](../internal/thinking-roundtrip.md) §4.2。

`Capabilities` 的 `thinking` 字段对 OpenAI 默认报 `Unsupported`、`thinking_echo` 默认 `Forbidden`；DeepSeek base_url 时通过 `OpenAiConfig` 在构造时传入 `Capabilities` 覆盖（`thinking: Supported`），并按模型在 `HARDCODED_MODELS` 里设 `ModelCapabilityOverrides::thinking_echo`——`deepseek-v4-flash` / `deepseek-v4-pro` → `Required`。这层覆盖在构造期发生，不影响每请求判定。

DeepSeek prompt cache / smoke 的排障经验另见 [`../testing/deepseek-cache-smoke.md`](../testing/deepseek-cache-smoke.md)。

## 6. 错误映射差异

整体结构同 [`llm-anthropic.md`](./llm-anthropic.md) §7，OpenAI 特有的：

| 来源 | 触发 | 映射 |
| --- | --- | --- |
| HTTP 400 + body `error.code = "context_length_exceeded"` | – | `ContextOverflow{}` |
| HTTP 400 + body `error.type = "invalid_request_error"` + `param: "max_tokens"` | – | `MaxTokensInvalid{}` |
| HTTP 401 | – | `AuthRejected{ hint: error.message }` |
| HTTP 403 + `error.code = "insufficient_quota"` | – | `QuotaExceeded{}` |
| HTTP 404 + `error.code = "model_not_found"` | – | `ModelNotFound{ model }` |
| HTTP 429 with `Retry-After` | – | `RateLimit{ retry_after, scope: 看 error.type 判 RPM/TPM }` |
| HTTP 429 without `Retry-After` + `error.type = "tokens_per_min_exceeded"` | – | `RateLimit{ retry_after: None, scope: Tpm }` |
| HTTP 5xx | – | `ServerError{ status }` |
| 流中 `data: { error: ... }` | – | 映射上述错误规则 |

`request_id` 从 `x-request-id` header 抽取。

## 7. `list_models`

```rust
fn list_models(&self) -> BoxFuture<...> {
    Box::pin(async move {
        if let Some(v) = self.models.read().await.clone() { return Ok(v); }
        let resp = self.client.clone().call(wire::operations::models::get::Request {}).await?;
        // OpenAI Model 元素只有 id / created / owned_by — 不带 context_window
        let mapped: Vec<_> = resp.data.into_iter().map(|m| ModelInfo {
            id: m.id,
            display_name: None,
            context_window: None,
            max_output_tokens: None,
            deprecated: false,
            capabilities_overrides: ModelCapabilityOverrides::default(),
        }).collect();
        // 与硬编码表合并，把 context_window 等填进去
        let mapped = merge_with_hardcoded(mapped, &HARDCODED_MODELS);
        *self.models.write().await = Some(mapped.clone());
        Ok(mapped)
    })
}
```

`HARDCODED_MODELS` v0 列：`gpt-4o-mini`, `gpt-4o`, `o1-mini`, `o1`, `o3-mini`, `o3`, `deepseek-v4-flash`, `deepseek-v4-pro`。其余无表项，`context_window: None`。

兼容厂商的 `/v1/models` 字段在不同厂商间不一致（DeepSeek 多、Together 不全），共同字段只有 `id`——其余按 `None` 处理是当前最稳。

## 8. Capabilities

```rust
// OpenAI 官方默认值
Capabilities {
    tool_calls: Supported,
    parallel_tool_calls: Supported,  // OpenAI 默认开；可通过 sampling 关
    thinking: Unsupported,           // 官方：通过 reasoning_effort 字段；非通用，按 Unsupported 上报
    vision: Supported,
    prompt_cache: Supported,         // GPT-4o 等自动启用 prompt cache
}
```

`OpenAiConfig::with_capabilities_override(...)` 让 DeepSeek 等改 `thinking: Supported`；这层覆盖在构造期发生，不影响每请求判定。

`ModelCapabilityOverrides` v0 用法：`o1-*` / `o3-*` 系列覆 `parallel_tool_calls: Unsupported`（这些模型不允许并发工具调用）。

## 9. 测试策略

同 [`llm-anthropic.md`](./llm-anthropic.md) §10：

- 单测：`crates/llm/src/protocol/openai_chat/test.rs`，覆盖 tool_calls 隐式 end、`finish_reason` 五种值、`reasoning_content` 走 ThinkingDelta、末帧 usage、`[DONE]` 终结。
- 集测：`crates/llm/tests/openai_e2e.rs`，wiremock 模拟 stream。**不打真的 OpenAI**。

## 10. Codegen 工作流

跟 Anthropic 共用 `scripts/llm-codegen`：

```bash
# 平时改 OAS 后重新生成
cargo run -p defect-llm-codegen -- openai

# 上游同步 OpenAI 官方 OAS
cargo run -p defect-llm-codegen -- openai-strip --upstream path/to/openai/openapi.yaml
```

第二条命令做 1.1 的 keep-paths-and-refs 裁剪 + 1.2 的 patches，输出到 `crates/llm/oas/openai.yaml`。第一条命令读 `oas/openai.yaml` 出 `src/wire/openai.rs`。

## 11. 共享 vs 分文件

考虑过 OpenAI / DeepSeek / Together / Qwen / vllm 各开一个 `provider/<vendor>.rs` 文件，否决：

- **真差异只有 base_url + 模型清单 + capabilities 覆盖**——三件事都是构造参数，扩成单独文件得复制 200 行 transport 装配代码。
- 真出现 wire 层差异（如 vllm 的 `tool_choice` 解析有怪异）→ 在 `OpenAiConfig` 里加 quirk flag，由 codec 内部分支。
- 后续如果某家厂商有 transport 层差异（OAuth、签名）→ 那时再开新文件。

跟 [`docs/outbound/llm.md`](./llm.md) §1.3 的"两层架构每加一家厂商只新增一个文件"看似冲突——不冲突。`provider/openai.rs` 已经是**一家厂商**（"OpenAI 兼容"是 wire 协议族，而非厂商身份）。Bedrock / Vertex 这种 transport 完全异质的才会真新增文件。
