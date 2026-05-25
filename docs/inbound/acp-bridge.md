# ACP 桥接层设计

`defect-acp` 是 `defect-agent` 的协议适配层，把 [Zed Agent Client Protocol](https://agentclientprotocol.com) 的 JSON-RPC wire 形态翻译成 agent 内部的 `Session` / `AgentEvent` 接口。本文沉淀这一层的形状与边界。

## 1. 定位与边界

### 1.1 形态：薄壳

`defect-acp` 只暴露**一个入口** + **一个 trait 实现**：

```rust
// 公共入口
pub async fn serve(agent: Arc<dyn Bridgeable>) -> Result<(), AcpError>;
```

内部把 `agent-client-protocol` 的 builder 全配好、起 `Stdio` 传输、注册各方法回调。`defect-cli` 几乎只需要：

```rust
let agent = build_agent_core(...);   // defect-agent + defect-llm + defect-tools 装配
defect_acp::serve(agent).await?;
```

**不是**可组装框架——v0 只一种前端形态（stdio）。出现新对接（HTTP / socket / unix-domain）再抽 trait。

### 1.2 边界：协议适配，不参与业务

`defect-acp` 职责清单：

- ✓ 接收 ACP request / notification，按方法分发
- ✓ 调用 `Session` 上的对应方法，把内部错误映射成 ACP `Error`
- ✓ 订阅 `Session` 的 `AgentEvent` 流，按[投影表](./acp-bridge.md#3-agentevent--sessionupdate-翻译表)推 `session/update`
- ✓ 桥接 `request_permission`：发 wire request、等 client 响应、回写到 `AgentEvent` / 主循环

**不**做的事：

- ✗ 任何业务决策（要不要重试 / 要不要压缩 context / 该不该执行某工具）—— 全在主循环
- ✗ 持久化 —— 走 `defect-storage`，订阅同一条 `AgentEvent` 流
- ✗ 配置加载 —— 走 `defect-config`

## 2. ACP 方法分发

ACP 0.12 的 v2 schema 列出了一大票方法。v0 只实现真正必要的子集：

| 方法                                                    | 类型         | 方向               | v0   | 实现位置                            |
| ------------------------------------------------------- | ------------ | ------------------ | ---- | ----------------------------------- |
| `initialize`                                            | request      | client → agent     | ✓    | `handlers::initialize`              |
| `authenticate`                                          | request      | client → agent     | stub | 不开 auth capability，直接 reject   |
| `session/new`                                           | request      | client → agent     | ✓    | `handlers::session_new`             |
| `session/load`                                          | request      | client → agent     | P1   | 持久化做完再上                      |
| `session/prompt`                                        | request      | client → agent     | ✓    | `handlers::session_prompt`          |
| `session/cancel`                                        | notification | client → agent     | ✓    | `handlers::session_cancel`          |
| `session/update`                                        | notification | **agent → client** | ✓    | 由 AgentEvent 流投影                |
| `session/request_permission`                            | request      | **agent → client** | ✓    | `permission::request`               |
| `fs/read_text_file`                                     | request      | agent → client     | P1   | 走 [`acp-fs.md`](./acp-fs.md)；按 client 的 `FileSystemCapabilities` 决定走 `AcpFsBackend` 还是 `LocalFsBackend` |
| `fs/write_text_file`                                    | request      | agent → client     | P1   | 同上                                |
| `terminal/*`                                            | request      | agent → client     | P2   | 嵌入终端复杂，先不做                |
| `session/fork` / `resume` / `close` / `list` / `delete` | request      | client → agent     | P2   | 不声明对应 capability               |
| `providers/*` / `session/set_*`                         | request      | client → agent     | P2   | 不声明对应 capability               |

### 2.1 `initialize`

```rust
async fn initialize(req: InitializeRequest) -> Result<InitializeResponse, Error> {
    Ok(InitializeResponse::new(req.protocol_version)
        .agent_capabilities(capabilities()))
}

fn capabilities() -> AgentCapabilities {
    AgentCapabilities::new()
        // v0 声明的 PromptCapabilities 子项（image / audio / embedded_context）
        // 取决于我们对多模态的支持，初期保守：仅 text + resource_link
        // load_session: false（无持久化）
        // 不声明 session_capabilities 的 list/fork/resume/close/delete
}
```

具体能力位的选择见 [`acp-handshake.md`](./acp-handshake.md)（待写——可与本文合并或拆出）。

### 2.2 `session/new`

```rust
async fn session_new(req: NewSessionRequest) -> Result<NewSessionResponse, Error> {
    let session = agent.create_session(req.cwd, req.mcp_servers).await?;
    Ok(NewSessionResponse::new(session.id()))
}
```

`Session` 实例的所有权归 `defect-agent`（见[岔路 2 决议](#关键决议)）。`defect-acp` 持有 `Arc<dyn AgentCore>`，通过 trait 方法访问。

### 2.3 `session/prompt` —— 长 request

ACP 的 `session/prompt` 是 request：**有返回值** `PromptResponse { stop_reason }`。语义上：

```
client ──prompt──► agent
                   │
                   │  agent 持续推 session/update（流式）
                   ▼
                   ...
                   │
agent ──response── │  turn 结束，return PromptResponse
       (stop_reason)
```

也就是 **prompt request 的处理时长 ≈ 整个 turn**。SDK 的 `on_receive_request` 回调是 `async fn`，天然支持。骨架：

```rust
async fn session_prompt(
    req: PromptRequest,
    cx: ConnectionTo<Client>,
) -> Result<PromptResponse, Error> {
    let session = agent.session(&req.session_id).await?;

    // 启动 turn，拿到事件流
    let mut events = session.start_turn(req.prompt).await?;

    while let Some(event) = events.next().await {
        match project(&event) {
            Project::Update(u)     => cx.send_notification(u).await?,
            Project::Permission(p) => handle_permission(&cx, &session, p).await?,
            Project::EndTurn(r)    => return Ok(PromptResponse::new(r)),
            Project::None          => {}
        }
    }

    // 事件流被外部 cancel 而无 TurnEnded：按 Cancelled 返回
    Ok(PromptResponse::new(StopReason::Cancelled))
}
```

`project` 是[投影表](#3-agentevent--sessionupdate-翻译表)的实现。`handle_permission` 见[§4](#4-请权限的双向流程)。

### 2.4 `session/cancel` —— notification

```rust
async fn session_cancel(req: CancelNotification) {
    if let Some(session) = agent.session(&req.session_id).await {
        session.cancel_turn();
    }
}
```

`Session::cancel_turn` 触发当前 turn 的 `CancellationToken`。主循环捕获取消后，事件流以 `TurnEnded { reason: Cancelled }` 收尾，§2.3 的循环看到它后正常 respond。

ACP 规范要求：`session/cancel` 之后所有 pending `session/request_permission` 必须 respond `RequestPermissionOutcome::Cancelled`。这一点由 `permission::request` 实现（监听 `cancel.cancelled()` future + `cx.send_request` 的 race），见 §4。

## 3. AgentEvent → SessionUpdate 翻译表

事件模型见 [`event-model.md`](../internal/event-model.md)。翻译规则：

| `AgentEvent` 变体                             | 是否上 wire                                                                | 翻译目标                                                                                  |
| --------------------------------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `TurnStarted`                                 | ✗                                                                          | （仅审计）                                                                                |
| `TurnEnded { reason, .. }`                    | wire 不发 update；驱动 `prompt` request 返回 `PromptResponse::new(reason)` | —                                                                                         |
| `AssistantText { content }`                   | ✓                                                                          | `SessionUpdate::AgentMessageChunk(ContentChunk::new(content))`                            |
| `AssistantThought { content }`                | ✓                                                                          | `SessionUpdate::AgentThoughtChunk(ContentChunk::new(content))`                            |
| `ToolCallStarted { id, fields }`              | ✓                                                                          | `SessionUpdate::ToolCall(ToolCall { tool_call_id: id, ..fields_into_tool_call(fields) })` |
| `ToolCallProgress { id, fields }`             | ✓                                                                          | `SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(id, fields))`                          |
| `ToolCallFinished { id, fields }`             | ✓                                                                          | `SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(id, fields))`                          |
| `PolicyDecision { decision: Ask, .. }`        | ✓                                                                          | 触发 `session/request_permission` 流程（见 §4）；非 update                                |
| `PolicyDecision { decision: Allow/Deny, .. }` | ✗                                                                          | （仅审计）                                                                                |
| `PermissionResolved`                          | ✗                                                                          | （仅审计——回写决策已经在 ACP request 的返回里完成）                                       |
| `LlmCallStarted` / `Finished`                 | ✗                                                                          | （仅 storage / tracing）                                                                  |
| `ContextCompressed`                           | ✗                                                                          | （仅 storage / tracing）                                                                  |

实现细节：

- 投影函数返回 `enum Project { Update(SessionNotification), Permission(...), EndTurn(StopReason), None }`，按变体分流。
- 字段为什么不直接借用：`ContentChunk` / `ToolCall` 还有 `_meta` / `unstable_message_id` 等字段，由桥接层在投影时填默认值，`AgentEvent` 不感知 wire 元信息。

## 4. 请权限的双向流程

ACP 的 `session/request_permission` 是**反向 request**（agent → client，等 client respond）。流程：

```
主循环                 桥接层                  ACP wire             client
  │                      │                       │                    │
  │ Event PolicyDecision │                       │                    │
  │  { decision: Ask }   │                       │                    │
  ├─────────────────────►│                       │                    │
  │                      │ cx.send_request       │                    │
  │                      │  RequestPermissionReq │                    │
  │                      ├──────────────────────►│ ──────────────────►│
  │                      │                       │                    │  (用户决定)
  │                      │                       │ ◄──────────────────│
  │                      │ outcome ◄─────────────│                    │
  │                      │                       │                    │
  │ ◄────────────────────┤ session.resolve_perm  │                    │
  │   resolve(outcome)   │   ition(id, outcome)  │                    │
  │                      │                       │                    │
  │ Event PermissionResol│                       │                    │
  │  ved (审计)          │                       │                    │
  ├─────────────────────►│                       │                    │
```

实现要点：

- `request_permission` 的 wire payload 直接用 `tool.describe(args)` 已经产出的 `ToolCallUpdateFields`（[Tool trait §3](../internal/tool-trait.md#3-toolcalldescription)）—— `fields` 与 `id` 一拼即可。
- `cx.send_request(...)` 与 `session.cancel_token.cancelled()` 在 `tokio::select!` 中竞速；cancel 触发时返回 `RequestPermissionOutcome::Cancelled` 给主循环，符合 ACP 规范。
- `PermissionOption` 列表由主循环组装（`Allow once` / `Allow always` / `Reject once` / `Reject always`）。具体选项决定属于 sandbox policy 设计，见 `docs/internal/sandbox-policy.md`（待写）。

## 5. `Session` 的接口需求

桥接层需要 `defect-agent` 暴露一组 trait（具体形状在 [`session.md`](../internal/session.md) 定，本文先列出来源需求）：

```rust
trait AgentCore: Send + Sync {
    async fn create_session(&self, cwd: PathBuf, mcp: Vec<McpServer>) -> Result<SessionHandle>;
    async fn session(&self, id: &SessionId) -> Option<SessionHandle>;
}

trait SessionHandle: Send + Sync {
    fn id(&self) -> &SessionId;
    async fn start_turn(&self, prompt: Vec<ContentBlock>) -> Result<EventStream>;
    fn cancel_turn(&self);
    /// 把 ACP 反向 request 的回执回写给主循环。
    fn resolve_permission(&self, id: ToolCallId, outcome: PermissionResolution);
}
```

不在 trait 上暴露任何 wire 类型——`AgentCore` / `SessionHandle` 都用 `defect-agent` 自己的类型；桥接层负责拼装 ACP 的 wire 形态。

## 6. 错误映射

`defect-agent` 内部错误（`ProviderError` / `ToolError` / 主循环错误）映射成 ACP `Error`：

| agent 错误                    | ACP `ErrorCode`                                             | 备注                                             |
| ----------------------------- | ----------------------------------------------------------- | ------------------------------------------------ |
| `Session not found`           | `InvalidParams`                                             | `session/prompt` / `cancel` 找不到 session 时    |
| `ProviderError::AuthRejected` | `InternalError` + data                                      | 客户端无法修，按 internal 报；详细写 data 字段   |
| `ProviderError::Canceled`     | （不报错——直接 respond `PromptResponse{Cancelled}`）        | 取消是正常路径                                   |
| `ProviderError::*` 其他       | `InternalError`                                             | data 带 retry hint，便于客户端展示               |
| `ToolError::*`                | （不报错——通过 `ToolCallFinished{status: Failed}` 上 wire） | 工具失败是 turn 内事件，不是 prompt request 失败 |

ACP 把 `ErrorCode::AuthRequired` 等专用错误码留给真正的鉴权流程；v0 我们不开 auth，不会用到。

## 7. v0 不做的事（明确划线）

- `session/load` / `fork` / `resume` —— 等 storage 落地
- `fs/*` —— 由 [`acp-fs.md`](./acp-fs.md) 落地：客户端声明 `FileSystemCapabilities { read_text_file, write_text_file }` 全开时走 `AcpFsBackend`，否则降级到 `LocalFsBackend`
- `terminal/*` —— 嵌入终端 UI 复杂度高，v1 再上
- `providers/*` / `session/set_model` / `session/set_mode` / `session/set_config_option` —— 不声明对应 capability，client 自然不会调
- `_meta` 透传 —— `AgentEvent` 不感知 `_meta`，桥接层投影时一律填 `None`

## 关键决议

回顾本文之前的讨论结论：

1. **A：薄壳形态**。`defect-acp` 不抽 trait 让上层注入；只暴露 `serve` 入口。新前端形态出现再抽。
2. **X：Session 归属 `defect-agent`**。协议适配层不持有业务态，`defect-acp` 通过 `AgentCore` trait 访问。
3. **Q + 局部复用：内部 `AgentEvent` enum**。变体我们定义（持久化稳、能表达 turn 边界与 LLM 调用）；字段类型尽量直接借 ACP 类型（`ToolCallUpdateFields` / `ContentBlock` / `StopReason`），避免重新发明。

## 8. 后续相关文档

- [`acp-handshake.md`](./acp-handshake.md) —— `initialize` 能力位的具体清单（v0 声明哪些）
- [`acp-session.md`](./acp-session.md) —— `session/new` / `session/load` 的字段语义
- [`acp-prompt.md`](./acp-prompt.md) —— `session/prompt` 的字段语义、image / resource_link 处理
- [`acp-permission.md`](./acp-permission.md) —— `RequestPermissionRequest` 的 PermissionOption 编排
- [`acp-cancel.md`](./acp-cancel.md) —— 取消语义的边界 case（pending permission / in-flight tool）
- [`acp-fs.md`](./acp-fs.md) —— P1：`fs/read_text_file` / `fs/write_text_file` 委托
