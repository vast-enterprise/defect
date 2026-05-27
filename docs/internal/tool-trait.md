# `Tool` trait 设计

`Tool` 是 `defect-agent` 中暴露给主循环的工具抽象。内置工具（`defect-tools`）与 MCP 适配器（`defect-mcp`）都通过实现这个 trait 接入。本文沉淀 trait 各组成部分的设计与取舍。

设计的根本原则是 **"以 ACP 为导向"**：工具产出的字段直接对位 [agent-client-protocol](https://agentclientprotocol.com/) 的 wire 类型（`ToolCallUpdateFields` / `RequestPermissionRequest`），避免重复造一份内部字段再做映射。

`Tool` 仅服务于本地工具——`fetch` / `fs` / `bash` / `capabilities.search.mode = "local"` 时的 `search`。**provider-hosted 能力（hosted search / hosted fetch 等）不实现 `Tool` trait**，由 provider adapter 在 wire 层直接处理，详见 [`capabilities.md`](./capabilities.md) §8。本地 `fetch` 工具的形状见 [`tools-fetch.md`](./tools-fetch.md)。

## 1. ToolSchema

工具的"对外名片"，仅描述参数形状，不带任何执行能力。

```rust
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    /// 参数 JSON Schema（Draft 2020-12 子集）。
    pub input_schema: serde_json::Value,
}
```

[`CompletionRequest::tools`](./llm-trait.md#3-请求侧类型) 接受 `Vec<ToolSchema>`——provider 不持有 `dyn Tool`，只把 schema 序列化进 wire JSON。这样工具的"声明"与"执行"解耦：同一份 schema 可以喂给 N 个 provider，而工具实例的所有权与生命周期归属主循环 / Session。

## 2. Tool trait 主签名

```rust
pub trait Tool: Send + Sync {
    fn schema(&self) -> &ToolSchema;

    fn safety_hint(&self, args: &serde_json::Value) -> SafetyClass;

    fn describe(&self, args: &serde_json::Value) -> ToolCallDescription;

    fn execute(&self, args: serde_json::Value, ctx: ToolContext<'_>) -> ToolStream;
}

pub type ToolStream = Pin<Box<dyn Stream<Item = ToolEvent> + Send>>;
```

四个方法对应一次工具调用的四个阶段：

| 方法 | 时机 | 是否做 IO |
|------|------|-----------|
| `schema()` | 装配工具集时（每轮开始前） | 否（返回引用） |
| `safety_hint(args)` | LLM 请求调用后、执行前，喂给 sandbox policy | 否（纯函数） |
| `describe(args)` | 执行前、推送 `ToolCall` / `RequestPermission` 时 | 否（纯函数） |
| `execute(args, ctx)` | policy 放行后实际执行 | 是 |

类型擦除（`Pin<Box<dyn Stream>>`）的考虑与 `LlmProvider::ProviderStream` 同源——主循环要把若干异构 `dyn Tool` 装进同一个 registry，关联类型会让上层签名带上无尽的泛型参数。

## 3. ToolCallDescription

```rust
use agent_client_protocol::schema::ToolCallUpdateFields;

pub struct ToolCallDescription {
    pub fields: ToolCallUpdateFields,
}
```

**直接复用 ACP 的 `ToolCallUpdateFields`**。这一份数据同时驱动三种 ACP 消息：

1. 首次推送 `ToolCall`（status = Pending）
2. `RequestPermissionRequest::tool_call`（请权限时给客户端展示的内容）
3. `ToolEvent::Progress` 的基线（增量更新只改差量字段）

字段约定：

- **`tool_call_id` 不在此结构中**。ACP 要求 `ToolCallId` 在 session 内唯一；由主循环统一分配（首选用 LLM 给的 `tool_use_id`，否则自生成 UUID），工具实现完全不感知 ID。
- **`raw_input` 不由工具填**。主循环在外层一次性把传入的 args 原样塞进去，避免工具实现自己塞导致与 wire 上的真实参数发散。
- **`status` 不由工具填**。从 [`ToolEvent`](#5-toolevent--toolstream) 的 variant 推断（`Progress` → `InProgress`，`Completed` → `Completed`，`Failed` → `Failed`）。

字段中由 `describe()` 真正需要填的是：

- `title`（必填）：给客户端 UI 展示的标题，如 `"Reading src/main.rs"`。
- `kind`：`Read` / `Edit` / `Delete` / `Move` / `Search` / `Execute` / `Think` / `Fetch` / `SwitchMode` / `Other`。
- `locations`：受影响的文件路径（与可选 `line`），让客户端的 "follow-along" 功能能跟随。
- `content`：通常此时为空；执行期通过 `Progress` 增量填入（终端输出、diff 等）。

### 3.1 为什么不自定义字段而直接抄 ACP？

考虑过定义内部 `ToolCallDescription { title, kind, locations }` 然后由桥接层映射到 ACP 字段。否决理由：

1. ACP 的 `ToolCallUpdateFields` 已经覆盖了我们能想到的所有字段，且字段集随协议演化由 ACP 维护者推进；我们再造一份只会延迟拿到新字段。
2. ACP 的 `ToolCallUpdate` / `ToolCall` / `RequestPermissionRequest` 共用同一组字段（前者是后者的 patch 形态）——映射层必须把内部类型分别拼成三种 wire 形态，工程成本高于直接复用。
3. 主循环想在 ACP 之外接入别的前端（CLI 直接打印、HTTP API 等）时，把 `ToolCallUpdateFields` 当成事实标准也比内部类型可移植性更强。

代价：`defect-agent` 直接依赖 `agent-client-protocol`。已经接受——`defect-acp` 反正要依赖它，而 `defect-agent` 的事件模型也终归要对位 ACP，没必要假装解耦。

## 4. SafetyClass 与 sandbox policy

```rust
#[non_exhaustive]
pub enum SafetyClass {
    ReadOnly,
    Mutating,
    Destructive,
    Network,
}
```

工具自己**只回答"我想做什么"**（safety_hint），**不决定"能不能做"**（policy）。Allow / Deny / Ask 由外部 sandbox policy 综合下列输入决定：

- `safety_hint(args)` 的返回值
- 用户配置的工具白名单 / 黑名单
- 当前 session 的历史授权（`PermissionOptionKind::AllowAlways` 记录）
- 工作目录与路径白名单

### 4.1 为什么 `safety_hint` 接收 args？

同一个工具的安全等级常常依参数而异。`bash(command="ls")` 是 `ReadOnly`，`bash(command="rm -rf /")` 是 `Destructive`。把 args 作为入参允许工具做最小限度的语义判别，比固定一个等级更安全。实现必须保持纯函数（不能做 IO），否则 policy 决策就被工具实现侧的副作用污染了。

### 4.2 为什么不在 trait 上加 `requires_permission(&self) -> bool`？

那是 policy 的语义而不是工具的语义。把"是否要 Ask"放在工具上会让用户配置失效——用户想把所有 `bash` 都设成 AllowAlways 时，工具自己说"我永远要 Ask"就把用户的策略架空了。

## 5. ToolEvent / ToolStream

```rust
#[non_exhaustive]
pub enum ToolEvent {
    /// 进度增量。映射到 ACP `session/update` 的 tool_call_update。
    Progress(ToolCallUpdateFields),

    /// 成功结束。fields 里通常携带最终的 content / locations / raw_output。
    Completed(ToolCallUpdateFields),

    /// 失败结束。携带 Rust 侧错误便于上层 retry / log；
    /// 主循环把 status 设为 Failed 并把文本塞进 content。
    Failed(ToolError),
}
```

终态语义：

- 流中**至多一个** `Completed` 或 `Failed`，且必须是流的**最后一个**事件。
- 主循环看到终态后即视为本次工具调用结束，不再消费后续元素。
- drop 流视为取消，等价于 `ctx.cancel.cancel()`。

### 5.1 为什么 Progress / Completed 都是 `ToolCallUpdateFields`？

复用 ACP 的"patch"语义：`ToolCallUpdateFields` 所有字段都是 `Option<T>`，工具想改什么字段就 `Some(...)`，主循环原样转发到 ACP wire 不做翻译。这避免了"工具内部事件 → 中间结构 → wire patch"的两跳映射。

### 5.2 为什么 Failed 不合并进 Completed？

考虑过 `Finished { update, ok: bool }` 的合并形态。否决理由：

1. `update.status` 已经能区分 Completed / Failed；`ok` 字段是冗余字段，违背"减少冗余"的设计原则。
2. `Failed` 需要携带 Rust 侧的 [`ToolError`](#7-toolerror) 类型给主循环做差异化处理（重试 / 让 LLM 修正参数 / 中断 turn），把它压平成 `bool` 会丢失信息。
3. 调用方匹配语义更清楚：`match event { Failed(e) => log_error(e), Completed(_) => ... }` 比 `if !ok { ... }` 可读。

### 5.3 为什么用 `Stream` 而不是 `async fn -> ToolResult`？

工具调用的 UX 需要流式反馈：

- `bash` / `read_terminal` 的输出应当增量推到客户端，而不是等命令结束才显示。
- `edit_file` 想在写入前先推一个 `Progress(content=Diff(...))` 让客户端预览。
- 长跑任务（构建、网络抓取）需要进度条 / 阶段提示。

如果用 `async fn -> ToolResult`，要么这些场景全部退化成"无反馈直到结束"，要么我们再发明一套 channel 注入到 `ToolContext`，徒增复杂度。`Stream<Item = ToolEvent>` 就是工具实现写流式输出最自然的形态，并且 `futures::stream::once` / `tokio_stream::wrappers::ReceiverStream` 都能让简单工具的实现也保持一行的简洁度。

## 6. ToolContext

```rust
#[non_exhaustive]
pub struct ToolContext<'a> {
    pub cwd: &'a Path,
    pub cancel: CancellationToken,
}
```

**显式 struct 注入**而非 thread-local / 环境变量。理由：

- 测试时构造 `ToolContext` 一目了然（不用 mock 全局态）。
- 避免不同 session 在多线程下相互污染。
- 字段标 `non_exhaustive`，后续追加（sandbox 句柄、`fs` trait、ACP 反向通道用于发出 `terminal/create` 等）不破坏现有实现。

工具实现应在长循环 / await 点检查 `ctx.cancel.is_cancelled()` 并尽快退出；通过 `select!` 等待 `cancel.cancelled()` 也是常见模式。

## 7. ToolError

```rust
#[non_exhaustive]
pub enum ToolError {
    Canceled,
    InvalidArgs(BoxError),
    Execution(BoxError),
}
```

粒度故意保持粗——主循环只需要对这三类做差异化处理：

- `Canceled`：不报告失败，不消耗 retry 预算。
- `InvalidArgs`：可以把错误信息送回 LLM 让模型修正参数后重试（不消耗 turn 预算）。
- `Execution`：照常计入失败，按 sandbox policy / 用户配置决定是否重试。

更细粒度的错误类型由内置工具自己塞进 `Execution(BoxError)` 的 source 里携带（`BoxError` 见 [`crate::error::BoxError`](#备注boxerror)）。和 [`ProviderError`](./llm-trait.md#7-providererror) 不同的是，`ToolError` 不承担 retry hint 的责任——工具的可重试性由 policy 决定（同一个 `Execution` 错误，对 `bash` 可重试、对 `edit_file` 不可重试）。

### 备注：BoxError

`crate::error::BoxError` 是 `defect-agent` 全 crate 统一的"类型擦除错误来源" newtype，封装 `Box<dyn std::error::Error + Send + Sync>`。`ProviderError` / `ToolError` 等 enum 中需要透传任意 std error 时一律用它，不在公共类型里写裸 `Box<dyn ...>`。设计与扩展原则见 `crates/agent/src/error.rs` 的 doc comment。

## 8. 与 ACP 消息的整体映射

下面给出主循环把 `Tool` 产出的事件转换成 ACP wire 消息的概念性流程：

```text
LLM 流出 ToolUseStart{id, name} + ToolUseArgsDelta* + ToolUseEnd
        │
        ▼
 主循环：args = parse_json(deltas), id = allocate_tool_call_id(llm_id)
        │
        ▼
 tool.safety_hint(&args) ──► policy 决策
        │                          │
        │                  ┌───────┴───────┐
        │                  ▼               ▼
        │             Allow / Deny    Ask
        │                  │               │
        │                  │               ▼
        │                  │     ACP RequestPermissionRequest {
        │                  │       tool_call: ToolCallUpdate {
        │                  │         id, fields: tool.describe(&args).fields
        │                  │       },
        │                  │       options: [...]
        │                  │     }
        │                  ▼
        │           ACP session/update (ToolCall { id, status: Pending,
        │             ...tool.describe(&args).fields, raw_input: args })
        ▼
 tool.execute(args, ctx)
        │
        ├──► ToolEvent::Progress(fields)  ──► session/update (ToolCallUpdate { id, fields })
        │       (终端输出、diff、location 变化、阶段标题等)
        │
        └──► ToolEvent::Completed(fields)  ──► session/update (ToolCallUpdate {
                                                  id, fields: { status: Completed, ...fields }
                                              })
              ToolEvent::Failed(error)     ──► session/update (ToolCallUpdate {
                                                  id, fields: { status: Failed,
                                                  content: [error.to_string()] }
                                              })
        │
        ▼
 主循环把工具结果封装成 MessageContent::ToolResult，加入 history，进入下一轮 LLM
```

## 9. 待办与延伸

- **`ToolError` 的取消语义**：当 `ctx.cancel` 因 `session/cancel` 触发时，工具发 `Failed(Canceled)` 还是直接 drop 流？倾向后者，但需要在 ACP 桥接层定一个明确约定——见 `docs/internal/event-model.md`（待写）。
- **多模态返回**：`ToolCallContent` 已经支持 `Content(ContentBlock)` / `Diff` / `Terminal` 三种；具体的工具内置实现何时用哪种，留给各工具自己的设计文档。
- **工具组合**：proxy / wrapper 工具（如 "在 sandbox 里跑另一个工具"）的设计延后到具体场景出现时再讨论。
- **MCP 桥接**：`defect-mcp` 把 `rmcp::Tool` 包成 `defect_agent::tool::Tool` 的具体映射在 `docs/outbound/mcp.md`（待写）。
