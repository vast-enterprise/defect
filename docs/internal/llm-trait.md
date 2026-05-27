# `LlmProvider` trait 设计

`LlmProvider` 是 `defect-agent` 中暴露给主循环的 LLM 抽象，由 `defect-llm` 中的具体厂商实现接入（详见 [`docs/outbound/llm.md`](../outbound/llm.md)）。本文档分阶段沉淀 trait 的各组成部分，先从 **`ProviderChunk`**（流式输出的统一形状）开始——它是 trait 的核心耦合面，决定了主循环、provider、ACP 桥接三方的事件交互。

## 1. ProviderChunk

### 1.1 定义

```rust
#[non_exhaustive]
pub enum ProviderChunk {
    /// 流的第一个事件，仅含会话级元信息。
    MessageStart {
        id: String,
        model: String,
    },

    /// 助手文本增量。
    TextDelta { text: String },

    /// 思考链文本增量（Anthropic extended thinking / DeepSeek
    /// reasoning_content / o1-style 等的统一抽象）。
    ThinkingDelta { text: String },

    /// 思考链签名（Anthropic 多轮保留 thinking 时的校验数据，
    /// 不是文本，不应与 ThinkingDelta 合并）。
    ThinkingSignature { signature: String },

    /// 工具调用开始：声明一个新 tool_use，后续 ToolUseArgsDelta
    /// 与 ToolUseEnd 都通过 id 关联到本次调用。
    ToolUseStart { id: String, name: String },

    /// 工具调用参数片段。`fragment` 为裸字节片段，**不保证是
    /// 合法 JSON 子串**；调用方必须在收到对应 ToolUseEnd 之后
    /// 才能整体解析为 JSON。
    ToolUseArgsDelta { id: String, fragment: String },

    /// 工具调用结束：此 id 对应的所有 ArgsDelta 已发完。
    ToolUseEnd { id: String },

    /// 本次生成结束。stop_reason 见 [`StopReason`]。
    Stop { reason: StopReason },

    /// token 使用统计。可能在一次流中多次到达（部分 provider
    /// 分开发送 input / output / cache 部分），调用方应累加而非覆盖。
    Usage(Usage),
}

#[non_exhaustive]
pub enum StopReason {
    /// 模型自然结束本轮。
    EndTurn,
    /// 命中 max_tokens。
    MaxTokens,
    /// 命中 stop_sequence。
    StopSequence,
    /// 模型请求工具调用，调用方应执行 tool_use 后续轮。
    ToolUse,
    /// 安全策略拒绝输出。
    Refusal,
}

pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
}
```

### 1.2 语义约束

- **首事件总是 `MessageStart`**：流第一个有效事件必定是 `MessageStart`，否则视为协议错误
- **`Stop` 是最后一个语义事件**：之后只可能再出现 `Usage`（部分 provider 把 usage 放在 stop 之后），不会再有任何 Delta
- **tool_use id 唯一且贯穿**：从 `ToolUseStart` 到 `ToolUseEnd`，同一 id 间的 `ToolUseArgsDelta` 按到达顺序拼接得到完整参数 JSON
- **并发 tool_use**：OpenAI 协议允许多个 tool_use 的 `ArgsDelta` 在 wire 上交错；调用方按 id 分桶累积，不依赖 wire 顺序
- **`fragment` 是裸字节，不增量解析**：直到 `ToolUseEnd` 收齐后才整体 `serde_json::from_str`
- **`Usage` 可累加**：多次到达时调用方逐字段相加，不覆盖
- **`ping` 等 keep-alive 在协议层吞掉**：协议层不向上传

### 1.3 错误模型

provider 的 `complete` 返回 `Stream<Item = Result<ProviderChunk, ProviderError>>`：

- 单条 `Err` 视为终止——调用方收到错误后丢弃流，不再消费后续事件
- 流中可恢复的协议噪声（解析失败的单条 SSE event、未知 wire 类型）不应抛 Err，由协议层选择"吞掉并 tracing::warn"或"上抛 Malformed"——细则在协议层文档中规定

`ProviderError` 的精确分类在本文档第 4 节定义（待写）。

## 2. trait 主签名

```rust
use futures::{Stream, future::BoxFuture};
use std::pin::Pin;
use tokio_util::sync::CancellationToken;

pub type ProviderStream =
    Pin<Box<dyn Stream<Item = Result<ProviderChunk, ProviderError>> + Send>>;

pub trait LlmProvider: Send + Sync {
    /// 厂商元信息（厂商名、API 风格、tracing 标签等）。
    fn info(&self) -> ProviderInfo;

    /// 厂商级能力矩阵。模型级差异通过 [`ModelInfo::capabilities_overrides`]
    /// 表达，主循环按需合并。
    fn capabilities(&self) -> Capabilities;

    /// provider adapter 自报家门的 hosted capability 集合（与
    /// [`Self::capabilities`] 不同——前者是模型属性，这里是当前 adapter
    /// 实现状态：能否在 wire 上声明 hosted search / fetch 等）。
    ///
    /// 默认实现返回全 `false`。session 启动期与 `capabilities.search.mode`
    /// 一起做能力来源裁决；详见 [`capabilities.md`](./capabilities.md)。
    fn hosted_capabilities(&self) -> HostedCapabilities {
        HostedCapabilities::default()
    }

    /// 列出此 provider 当前可用的模型。
    ///
    /// 实现可能产生网络调用（如 OpenAI `/v1/models`），结果应在
    /// provider 内部缓存以供 [`Self::model_info`] 同步查询。
    fn list_models(&self)
        -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>>;

    /// 同步查询某个模型的元信息。
    ///
    /// 用于主循环裁剪 context 时的快路径，不应触发网络调用。
    /// 若 provider 缓存里没有，返回 `None`；调用方可决定是先调
    /// [`Self::list_models`] 再重试，还是按未知模型处理。
    fn model_info(&self, model_id: &str) -> Option<ModelInfo>;

    /// 启动一次流式生成。
    ///
    /// `cancel` 由调用方持有，可在任意时刻 `cancel()` 终止此次
    /// 调用与下游 stream。drop 返回的 stream 同样视为取消。
    fn complete(
        &self,
        req: CompletionRequest,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>>;
}
```

### 2.1 类型擦除选择

trait 全员 `BoxFuture` + `ProviderStream` 类型擦除，目的是让 `dyn LlmProvider` 直接可用——`Session` 需要在运行期持有"当前 provider"，多家 provider 都要塞进同一个槽。

代价：每次方法调用一次 box。在 LLM 场景（每次调用都是网络 IO + 数百 ms）下不可测，可以忽略。

### 2.2 取消语义

- `cancel.cancel()` 后，`complete` 返回的 future / stream **应该**很快终止
- 终止形态由 provider 自由选择：
  - 直接 `Poll::Ready(None)` 终结流（静默取消）
  - 或先 yield 一次 `Err(ProviderError::Canceled)` 再终结（显式取消）
- drop stream 仍是合法取消方式；`cancel` 只是显式入口，不替代 drop
- provider 实现负责把 `cancel` 接到底层 transport（reqwest 用 `tokio::select!`、AWS event-stream 用原生 abort、OAuth 拦截器在请求前检查）

### 2.3 返回值传值

`info()` / `capabilities()` 返回值而非引用——避免方法签名带生命周期，使 trait object 的 vtable 形态最简单。要求 `ProviderInfo` 与 `Capabilities` 都 `Clone`，每次调用克隆少量字段（开销可忽略）。

## 3. 入参与元信息类型

> 字段集合是初版，会在后续厂商落地中按需扩展；标 *(P2)* 的字段 v0 不必填充。

```rust
pub struct CompletionRequest {
    pub model: String,
    pub system: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSchema>,
    pub tool_choice: ToolChoice,
    pub sampling: SamplingParams,
    /// 本轮允许 provider 自行使用的 hosted capability 集合。session
    /// 启动期一次性裁决，turn 装配时透传。详见 [`capabilities.md`](./capabilities.md) §9。
    pub hosted_capabilities: HostedCapabilities,
}

pub struct Message {
    pub role: Role,
    pub content: Vec<MessageContent>,
}

pub enum Role { User, Assistant }

pub enum MessageContent {
    Text(String),
    /// 上一轮模型产出的思考链，仅出现在 [`Role::Assistant`] 消息里。
    /// 是否回放由 [`Capabilities::thinking_echo`] 决定（详见
    /// [`thinking-roundtrip.md`](./thinking-roundtrip.md)）。`signature`
    /// 仅 Anthropic extended thinking 使用，DeepSeek 等纯文本 echo 的
    /// provider 这里为 [`None`]。同条 assistant message 内 `Thinking`
    /// 必须排在 `Text` / `ToolUse` 之前——Anthropic wire 顺序约定。
    Thinking { text: String, signature: Option<String> },
    /// 历史轮次的工具调用：发出请求时把上一轮 tool_use 与 tool_result
    /// 都放在 messages 里，让 provider 重建上下文。
    ToolUse { id: String, name: String, args: serde_json::Value },
    ToolResult { tool_use_id: String, output: ToolResultBody, is_error: bool },
    /// provider-hosted 能力产生的活动（hosted search 调用 + 结果）。
    /// agent 主循环不解释 `payload`；codec 在重发同 provider 时透传，
    /// 切 provider 时由 codec 决定如何降级。`#[serde(skip)]` 不持久化。
    /// 详见 [`capabilities.md`](./capabilities.md) §7。
    ProviderActivity {
        provider_id: String,
        kind: ProviderActivityKind,
        payload: serde_json::Value,
    },
    /// 多模态输入。 *(P2)*
    Image { mime: String, data: ImageData },
}

pub enum ToolResultBody {
    Text(String),
    Json(serde_json::Value),
}

pub enum ToolChoice {
    /// 模型自行决定。
    Auto,
    /// 强制至少调用一个工具。
    Required,
    /// 强制调用指定工具。
    Named(String),
    /// 禁止工具调用，只产文本。
    None,
}

pub struct SamplingParams {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub stop_sequences: Vec<String>,
    /// 启用思考链（Anthropic extended thinking / DeepSeek reasoning）。
    pub thinking: ThinkingConfig,
}

pub enum ThinkingConfig {
    Disabled,
    /// 启用，并给一个 token 预算上限（Anthropic）。
    /// 不支持 budget 概念的 provider 忽略该字段。
    Enabled { budget_tokens: Option<u32> },
}

pub struct ProviderInfo {
    /// 厂商标识（"anthropic" / "openai" / "bedrock" / ...）。
    pub vendor: String,
    /// 底层协议（"anthropic_messages" / "openai_chat"）。
    pub protocol: ProtocolId,
    /// 用于 tracing 的人读名（"Anthropic Claude"）。
    pub display_name: String,
}

pub enum ProtocolId {
    AnthropicMessages,
    OpenAiChat,
}

pub struct ModelInfo {
    pub id: String,
    pub display_name: Option<String>,
    pub context_window: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub deprecated: bool,
    /// 此模型相对 provider 全局 capabilities 的差异。
    /// 例如同一 Anthropic provider 下，Sonnet 4 支持 thinking
    /// 而 Haiku 不支持。
    pub capabilities_overrides: ModelCapabilityOverrides,
}
```

`Capabilities` / `ModelCapabilityOverrides` / `FeatureSupport` 三态枚举的字段集合在第 5 节定义（待写）。

`ToolSchema` 定义在 `defect-agent` 的 tool 模块（详见 [`tool-trait.md`](./tool-trait.md)），不在本文档展开。

### 3.1 缓存与 list_models 协同

- v0 实现：`list_models` 返回硬编码表（Anthropic 的几个 Claude 模型 / OpenAI 当前活跃模型），`model_info` 直接查同一张表
- 后续可演进：lazy fetch + 内部 `RwLock<Vec<ModelInfo>>` 缓存
- **不强制** `list_models` 必须先于 `model_info` 调用——provider 实现自己保证 `model_info` 至少能查到硬编码模型

## 4. 与两家 wire 的映射

### 4.1 Anthropic Messages SSE

| wire 事件 | 产生的 ProviderChunk |
| --- | --- |
| `message_start` | `MessageStart { id, model }` + `Usage { input_tokens }` |
| `content_block_start { type: "thinking" }` | （无，记录内部状态） |
| `content_block_delta { thinking_delta }` | `ThinkingDelta { text }` |
| `content_block_delta { signature_delta }` | `ThinkingSignature { signature }` |
| `content_block_start { type: "text" }` | （无） |
| `content_block_delta { text_delta }` | `TextDelta { text }` |
| `content_block_start { type: "tool_use", id, name }` | `ToolUseStart { id, name }` |
| `content_block_delta { input_json_delta }` | `ToolUseArgsDelta { id, fragment }` |
| `content_block_stop`（tool_use block） | `ToolUseEnd { id }` |
| `content_block_stop`（text/thinking block） | （无） |
| `message_delta { stop_reason, output_tokens }` | `Stop { reason }` + `Usage { output_tokens }` |
| `message_stop` | （无） |
| `ping` | （吞掉） |

注：`content_block index` 是 wire 内部索引，**不向上暴露**；codec 内部维护 `index → (kind, tool_use_id)` 表来确定 `content_block_stop` 该不该发 `ToolUseEnd`。

### 4.2 OpenAI Chat Completions stream

| wire 事件 | 产生的 ProviderChunk |
| --- | --- |
| 第一个 chunk（带 `role: assistant`） | `MessageStart { id, model }` |
| `delta.content` | `TextDelta { text }` |
| `delta.reasoning_content`（DeepSeek 等扩展） | `ThinkingDelta { text }` |
| `delta.tool_calls[i]`（首次出现 i，含 id/name） | `ToolUseStart { id, name }` |
| `delta.tool_calls[i].function.arguments` | `ToolUseArgsDelta { id, fragment }` |
| `finish_reason: tool_calls` | 对所有 in-progress tool emit `ToolUseEnd { id }` + `Stop { ToolUse }` |
| `finish_reason: stop` / `length` / `content_filter` | `Stop { 对应 reason }` |
| 末尾 `usage` chunk | `Usage { ... }` |

注：OpenAI 不发 `content_block_stop`，codec 必须在 `finish_reason` 时统一关闭所有未关闭的 tool_use。

## 5. Capabilities

`Capabilities` 是 provider 自描述的能力矩阵；`ModelCapabilityOverrides` 表达模型级差异。主循环按需合并：模型级 `Some(_)` 覆盖 provider 级，`None` 沿用 provider 级。

```rust
pub struct Capabilities {
    /// 工具调用（content_block 含 tool_use / tool_calls 字段）。
    pub tool_calls: FeatureSupport,

    /// 同一轮内并发多个 tool_use。
    pub parallel_tool_calls: FeatureSupport,

    /// 思考链（Anthropic extended thinking / DeepSeek reasoning_content
    /// / o1-style）。回答"模型**会不会**产 thinking 内容"，即 codec 入流
    /// 是否解码 `ThinkingDelta`；与 [`thinking_echo`](#thinking_echo) 配套
    /// 但语义不同。
    pub thinking: FeatureSupport,

    /// 多模态输入（图片）。
    pub vision: FeatureSupport,

    /// prompt cache。
    pub prompt_cache: FeatureSupport,

    /// thinking 内容回放策略——"产了 thinking 内容**该不该**回放给
    /// 服务端"。`Required` 强制把上一轮 thinking 写进下一轮请求
    /// （Anthropic extended thinking、DeepSeek-v4-pro），`Forbidden`
    /// 直接丢弃（DeepSeek-R1、OpenAI o1 / o3 官方），`Optional` 服务端
    /// 两种都容忍。详见 [`thinking-roundtrip.md`](./thinking-roundtrip.md)。
    pub thinking_echo: ThinkingEcho,
}

/// 模型级覆写。`None` 表示沿用 provider 级 [`Capabilities`] 字段。
///
/// 字段集合按"现实中真的会按模型变化"的属性限定，不机械与
/// `Capabilities` 一一对应。后续如出现新差异点再加。
pub struct ModelCapabilityOverrides {
    pub thinking: Option<FeatureSupport>,
    pub vision: Option<FeatureSupport>,
    pub prompt_cache: Option<FeatureSupport>,
    pub parallel_tool_calls: Option<FeatureSupport>,
    /// 让同一 provider 下不同模型表达不同回放策略——例如 DeepSeek
    /// 把 v4 系列设为 Required，预留口子让未来 Forbidden 模型可以单独
    /// 覆盖。
    pub thinking_echo: Option<ThinkingEcho>,
}

#[non_exhaustive]
pub enum ThinkingEcho {
    /// 默认。回放会被服务端拒。
    Forbidden,
    /// 必须把上一轮 thinking 原样写进下一轮请求。
    Required,
    /// 服务端两种行为都容忍。
    Optional,
}

#[non_exhaustive]
pub enum FeatureSupport {
    Supported,
    Unsupported,
    /// 通过适配伪支持。
    ///
    /// 例如某 provider 没有原生 web_search，但 agent 把它包装成
    /// 一个工具暴露给 LLM，借此"假装"支持。三态枚举允许这类
    /// "通过适配伪支持"状态用类型层面表达，bool 表达力不足。
    PassthroughAsTool,
}
```

### 5.1 字段筛选记录

`Capabilities` 故意**不列**以下字段：

| 候选字段 | 不列的理由 |
| --- | --- |
| `streaming` | trait 强制流式，所有 provider 必然 `Supported` |
| `streaming_usage` | `ProviderChunk::Usage` 已表达，主循环按出现累加 |
| `system_prompt` | 两家 wire 都支持，OpenAI 通过第一条 message 实现，codec 内部消化 |
| `forced_tool_choice` | trait 已有 [`ToolChoice::Required`] / `Named`，能用即视为支持 |
| `seed` / `logit_bias` / `response_format` | `SamplingParams` 字段，主循环传或不传即可，无需能力声明 |
| `web_search` / `image_generation` | 这是 capability（hosted vs local 来源协商），不是模型属性——走 [`HostedCapabilities`] + `[capabilities.*]` 配置树，详见 [`capabilities.md`](./capabilities.md) |
| `max_context_tokens` / `max_output_tokens` | per-model 信息，已在 [`ModelInfo`] 字段中 |

## 6. 设计取舍记录

- **不暴露 ContentBlock 边界**：text/thinking 切换靠"事件类型变化"隐式表达。tool_use 是唯一需要显式 Start/End 的，因为它有"按 id 累积参数"的需求
- **`Usage` 是独立事件而非 `Stop` 字段**：两家 provider 的 usage 发送时机不同（Anthropic 分两次：input 在 start、output 在 message_delta；OpenAI 在末尾），独立事件让 codec 自由发送，主循环累加
- **`MessageStart` 不包含 `system_fingerprint` 等 vendor-specific 字段**：这些走 tracing，不进 chunk
- **`Stop` 不带文本原因，只用枚举**：原始 wire 字段（"end_turn"/"stop"/"length"）若需要可走 tracing，主循环只关心语义类别
- **`fragment` 用 `String` 而非 `Bytes`**：两家 wire 都是 UTF-8 文本，`String` 更易用；如未来出现二进制片段（unlikely），再升级类型
- **不留 vendor 逃生通道（`Map<String, Value>`）**：盘点真实用例（prompt cache 标记、`seed`、`response_format`、`logit_bias`、`service_tier`、Anthropic beta、Bedrock guardrail 等）后发现，每条都属于"能力差异"或"provider 配置"两类，应通过 `Capabilities` + 命名字段表达，或在 provider 构造时配死，而非 per-request 通用 escape hatch。注释级"非迫不得已不得用"的警告在工程实践中会失败——类型本身就是文档（`claude.md` §41）。真要绕开 trait 的极少数 vendor-only 调用，应通过具体 provider 的 ext-trait（如 `impl Anthropic { pub async fn complete_anthropic(...) }`）暴露，不污染 `LlmProvider`

## 7. ProviderError

设计原则：**能明确分出的情况尽量分出，并保留兜底**（`Other`）。每个 variant 的 `retry_hint` 不同，合并会丢失信息。

### 7.1 顶层结构

把 kind 与 cross-cutting 的诊断信息（`request_id` 等）分开，避免在每个 variant 里重复字段：

```rust
pub struct ProviderError {
    pub kind: ProviderErrorKind,
    /// 服务端返回的 request id（Anthropic `request-id` header / OpenAI
    /// `x-request-id` 等）。排障第一信号源，应尽力填充。
    pub request_id: Option<String>,
}

// BoxError 见 crate::error::BoxError —— 全 crate 统一的 dyn-error newtype。
```

### 7.2 错误分类

```rust
#[non_exhaustive]
pub enum ProviderErrorKind {
    // ---------- 认证 ----------
    /// 没配凭证。
    AuthMissing { var_hint: Option<String> },
    /// 凭证格式错误（API key 不像 key、JWT 不能 decode）。
    AuthMalformed { hint: Option<String> },
    /// 凭证被服务端拒绝（401 InvalidApiKey）。
    AuthRejected { hint: Option<String> },
    /// OAuth/STS token 过期，可刷新后重试。
    AuthExpired,

    // ---------- 配额 ----------
    /// 请求级速率限制（RPM / TPM 命中）。
    RateLimit { retry_after: Option<Duration>, scope: RateLimitScope },
    /// 余额不足 / 月度配额耗尽。
    QuotaExceeded { hint: Option<String> },

    // ---------- 输入 ----------
    /// context window 撑爆。
    ContextOverflow { used: Option<u64>, limit: Option<u64> },
    /// 单次 max_tokens 超过模型上限或被服务端拒。
    MaxTokensInvalid { requested: Option<u64>, limit: Option<u64> },
    /// 模型 ID 不存在 / 不可用。
    ModelNotFound { model: String },
    /// 请求体被 wire 服务校验拒绝（schema 错误、互斥字段冲突）。
    BadRequest { hint: Option<String> },
    /// 请求里引用的工具 schema 自身被服务端拒绝。
    InvalidToolSchema { tool: String, hint: Option<String> },

    // ---------- 安全/合规 ----------
    /// 输入触发安全过滤器。
    InputBlocked { policy: Option<String> },
    /// 模型生成被安全过滤器中断。
    OutputBlocked { policy: Option<String> },

    // ---------- 协议/服务端故障 ----------
    /// 5xx 或服务端报告的内部错误。
    ServerError { status: Option<u16>, hint: Option<String> },
    /// 服务端在生成中切流。
    ServerStreamAborted { hint: Option<String> },
    /// wire JSON / SSE 解析失败。
    Malformed(BoxError),
    /// 服务端响应了未在协议规范内的 wire 类型/字段（强 schema 失败）。
    ProtocolViolation { hint: String },

    // ---------- 传输 ----------
    /// DNS / TCP / TLS / HTTP 层错误（reqwest::Error 等）。
    Transport(BoxError),
    /// 请求超时（连接 / 读 / 总）。
    Timeout { phase: TimeoutPhase },

    // ---------- 控制流 ----------
    /// 用户/上层主动取消（CancellationToken 或 drop stream 触发）。
    Canceled,

    // ---------- 兜底 ----------
    /// 仅用于"能确认是错误但不能确定语义类别"。
    /// 实现新增分类时优先把此处的 case 提取出去，**不让兜底变成默认值**。
    Other(BoxError),
}

pub enum RateLimitScope {
    /// 每分钟请求数。
    Rpm,
    /// 每分钟 token 数。
    Tpm,
    /// 服务端报告但未细分。
    Unspecified,
}

pub enum TimeoutPhase {
    Connect,
    ReadHeaders,
    ReadBody,
    Idle,
    Total,
}
```

### 7.3 重试建议

`retry_hint` 挂在外层 `ProviderError` 上，返回结构化建议（不是 bool）：

```rust
pub enum RetryHint {
    /// 不可重试。
    No,
    /// 立刻重试一次（瞬时故障）。
    Immediate,
    /// 等服务端建议时长后重试。
    After(Duration),
    /// 退避重试（无服务端建议）。
    Backoff,
    /// 需先做某事再重试。
    AfterAction(RetryAction),
}

pub enum RetryAction {
    RefreshAuth,
    SwitchModel,
    ReduceContext,
}

impl ProviderError {
    pub fn retry_hint(&self) -> RetryHint {
        use ProviderErrorKind::*;
        match &self.kind {
            // 永久错误
            AuthMissing { .. }
            | AuthMalformed { .. }
            | AuthRejected { .. }
            | ModelNotFound { .. }
            | BadRequest { .. }
            | InvalidToolSchema { .. }
            | InputBlocked { .. }
            | OutputBlocked { .. }
            | ProtocolViolation { .. }
            | MaxTokensInvalid { .. }
            | QuotaExceeded { .. }
            | Canceled
            | Other(_) => RetryHint::No,

            // 需先动作
            AuthExpired => RetryHint::AfterAction(RetryAction::RefreshAuth),
            ContextOverflow { .. } => {
                RetryHint::AfterAction(RetryAction::ReduceContext)
            }

            // 限流
            RateLimit { retry_after: Some(d), .. } => RetryHint::After(*d),
            RateLimit { retry_after: None, .. } => RetryHint::Backoff,

            // 瞬时
            ServerError { .. }
            | ServerStreamAborted { .. }
            | Malformed(_)
            | Transport(_)
            | Timeout { .. } => RetryHint::Backoff,
        }
    }

    /// 便捷判断：是否值得让 agent 自动重试。
    pub fn is_retryable(&self) -> bool {
        !matches!(self.retry_hint(), RetryHint::No)
    }
}
```

### 7.4 设计取舍记录

- **`Malformed` 归 `Backoff` 而非 `No`**：偶发 SSE 字节抖动重试可能恢复；codec 真正写错时上层应靠总重试次数限制兜底
- **`ServerStreamAborted` 与 `ServerError` 分开**：流中切断时上层可能想保留已 yield 的部分内容，这是请求级错误没有的状态
- **`Auth` 四态保留不合并**：四者 `RetryHint` 不同（前三个 `No`，`AuthExpired` 是 `AfterAction(RefreshAuth)`）
- **`Other` 不做 sealed 构造约束**：兜底就是兜底，定期 grep 审计 `Other` 出现位置即可
- **`request_id` 上提到外层 struct**：避免每个 variant 都加同一字段；它是诊断信息而非分类信息

## 8. 待续

- `ImageData`（多模态输入载荷）的具体形态——v0 不实现，标 *(P2)*
