# `Session` 设计

`Session` 是 `defect-agent` 中"一次对话"的生命周期单元。本文沉淀 Session、AgentCore、History、ToolRegistry 这一组核心抽象的形状与边界。

## 1. 抽象层级

```
defect-cli
    │ 装配
    ▼
Arc<dyn AgentCore>
    │ holds
    ├── ToolRegistry (内置工具，进程级共享)
    ├── LLM provider 注册表
    └── 全局配置
    │
    │ create_session(cwd, mcp_servers)
    ▼
Arc<dyn Session>  ── (per session)
    │ holds
    ├── id: SessionId
    ├── cwd: PathBuf
    ├── Box<dyn History>            ← 历史
    ├── Box<dyn ToolRegistry>       ← per-session 工具表（含 MCP）
    ├── Arc<TurnLock>               ← 单 turn 互斥
    ├── CancellationToken           ← 当前 turn 的取消信号
    └── 事件流广播总线
```

| 层级 | 抽象 | 作用域 | 持有者 |
| --- | --- | --- | --- |
| 进程 | `AgentCore` | 整个 CLI 生命周期 | `defect-cli` |
| 会话 | `Session` | `session/new` ~ `session/close` | `AgentCore` 内部表 |
| 子组件 | `History` / `ToolRegistry` | 跟随 `Session` | `Session` 内部 |

**所有 4 个核心类型都以 trait 暴露**。理由：

- 测试时可注入 mock，不必拉真实 LLM / 文件系统
- 未来形态扩展不动 `defect-acp` 的代码（嵌入式 agent / 多租户 / 远程 session 等）
- "v0 没必要" ≠ "永远没必要"——抽象的代价主要是 trait object 的间接调用，可以接受

## 2. AgentCore

```rust
pub trait AgentCore: Send + Sync {
    fn create_session(
        &self,
        cwd: PathBuf,
        mcp_servers: Vec<McpServer>,
    ) -> BoxFuture<'_, Result<Arc<dyn Session>, AgentError>>;

    fn session(&self, id: &SessionId) -> Option<Arc<dyn Session>>;
}
```

要点：

- **不暴露 LLM provider 选择**。哪个 provider / 模型由 `AgentCore` 实现内部根据全局配置决定，避免 `defect-acp` 接触 provider 类型。
- **`mcp_servers` 来自 ACP `NewSessionRequest`**。`AgentCore` 实现负责拉起 MCP server 子进程 / SSE 连接，把每个 MCP 工具按 `mcp.<server>.<name>` 命名空间包装成 `Arc<dyn Tool>` 加入会话工具表（详见 [`capabilities.md`](./capabilities.md) §6.2）。
- **session 启动期一次性裁决能力来源**。`create_session` 内部读 `(capabilities.search, provider.hosted_capabilities())` 确定本 session 的 search 来源（hosted / local / disabled），失败返回 [`AgentError::Init(SessionInitError::CapabilityUnsatisfied)`]；详见 [`capabilities.md`](./capabilities.md) §5。
- **session 表的并发模型**由实现决定（典型实现：`DashMap<SessionId, Arc<dyn Session>>`）；trait 不暴露。

[`AgentError::Init(SessionInitError::CapabilityUnsatisfied)`]: ./capabilities.md

## 3. Session

```rust
pub trait Session: Send + Sync {
    fn id(&self) -> &SessionId;

    fn subscribe(&self) -> EventStream;

    fn run_turn(
        &self,
        prompt: Vec<ContentBlock>,
    ) -> BoxFuture<'_, Result<StopReason, TurnError>>;

    fn cancel_turn(&self);

    fn resolve_permission(&self, id: ToolCallId, outcome: PermissionResolution);
}
```

### 3.1 双轨：subscribe + run_turn future

acp-bridge.md 与 event-model.md 已经定下事件流的语义。但事件流上**没有"turn 失败"的语义**（`StopReason` 只有 EndTurn / MaxTokens / MaxTurnRequests / Refusal / Cancelled），fatal 错误（鉴权过期、模型不可用、内部 invariant 破坏）找不到地方塞。

解法是**双轨**：

- `subscribe()` 返回事件流——给所有消费者订阅，发 `AgentEvent::*` 包括 `TurnEnded`
- `run_turn(prompt)` 返回 future——这个 future 的 outcome 是 `Result<StopReason, TurnError>`，acp 桥接据此决定 respond `PromptResponse` 还是 JSON-RPC `Error`

事件流上的 `TurnEnded` 仍然发——给 storage / tracing 看；但 acp 桥接**不**用它驱动响应，用 `run_turn` future 的 outcome。

桥接层用法：

```rust
let events = session.subscribe();
let turn = session.run_turn(prompt);
tokio::pin!(turn);
loop {
    tokio::select! {
        result = &mut turn => return result.map(PromptResponse::new).map_err(map_to_acp_error),
        Some(e) = events.next() => forward(&cx, e).await?,
    }
}
```

### 3.2 单 turn 互斥

ACP 规范要求一个 session 同时只能有一个 turn。`run_turn` 在另一个 turn 还没结束时被调用：返回 `TurnError::TurnInProgress`。客户端想串行就自己排队 await。

实现层面用 `tokio::sync::Mutex` 还是原子 flag 由实现决定；trait 只约定语义。

### 3.3 取消语义

`cancel_turn()` 触发当前 turn 的 `CancellationToken`：

- 主循环检查到 cancel 后中止 LLM 调用、中止 in-flight 工具、清空待发的 LLM 调用
- 主循环以 `Ok(StopReason::Cancelled)` 结束（**不是** `TurnError`——取消是正常路径）
- 事件流以 `TurnEnded { reason: Cancelled }` 收尾
- 所有 pending `request_permission` 的 wire 等待 future 收到 cancel 信号后 respond `RequestPermissionOutcome::Cancelled`（acp 桥接层实现）

幂等：没有 turn 在跑时调 `cancel_turn()` 是 no-op。

### 3.4 resolve_permission

ACP 反向 request `session/request_permission` 是 acp 桥接层自己发出去等回执的——但回执必须**回写**到主循环（主循环根据 outcome 决定是否执行工具）。

`Session::resolve_permission(id, outcome)` 是这条回写通道：acp 桥接层拿到 `RequestPermissionResponse` 后立刻调用，主循环内部按 `id` 找到等待的工具调用 future，唤醒它继续。

具体的等待机制由实现决定（典型：`DashMap<ToolCallId, oneshot::Sender<PermissionResolution>>`）。

## 4. History

```rust
pub trait History: Send + Sync {
    fn append(&self, msg: Message);
    fn snapshot(&self) -> Vec<Message>;
    fn replace(&self, messages: Vec<Message>);
    fn record_input_tokens(&self, tokens: u64);
    fn token_estimate(&self) -> Option<u64>;
}
```

**History 是纯存储 + token 计量，压缩编排不在这里。** 早先把 `compact()` 设在
trait 上是设计错位——摘要要调 LLM，存储抽象够不到 provider。改为：trait 提供
`replace`（整体回写）、`splice_prefix`（前缀替换，后台压缩回写用）与
`record_input_tokens`（喂真实用量），压缩的「选边界 → 调 LLM 摘要 → 重建消息列表」
放在 turn 主循环（见 [`turn-loop.md`](./turn-loop.md) §4 与
`crates/agent/src/session/turn/{compact,microcompact,compaction_slot}.rs`）。这与
codex / opencode / Claude Code 三家一致：摘要编排都在 session/turn 层而非消息存储里。

`splice_prefix(drop_count, summary)` 是后台压缩的回写原语：后台任务在旧 snapshot 上
算出 `drop_count`，但摘要 LLM 调用期间前台仍在尾插，故不能 `replace(整表)`——
`splice_prefix` 只换掉**当前**列表前 `drop_count` 条、保留其后全部（含期间新增的尾部）。
其并发安全前提（飞行期间不增删中段）由后台压缩 single-flight 保证，详见 turn-loop §4.1。
`history` 字段由此从 `Box<dyn History>` 改为 `Arc<dyn History>`——后台压缩任务要
`'static` 持有它跨 turn。

为什么仍抽 trait：

- **token 计数 / resume 仍需钩子**，不抽 trait 会以裸函数四散在主循环里
- **测试**：`MockHistory` 比改具体 struct 字段更可控

### token 估算（`VecHistory`）

不引入 tokenizer 依赖（对齐 opencode：trigger 用真实 usage，内部估算用字符启发式）。
`token_estimate` 两段拼接：

- **基线**：上一次 LLM 调用回报的真实输入 token（`input + cache_read +
  cache_creation`），由主循环每次调用后经 `record_input_tokens` 喂入——最准的一段
- **增量**：基线之后 `append` 的消息按 `chars/4` 估算累加（图片记 ~2000 常量）

`replace`（压缩后回写）会清空基线，等下一次真实调用回报；基线缺失时（新建 / 刚
replace）整份 snapshot 走字符启发式兜底，空历史返回 `None`。

## 5. 事件流：mpsc bounded + fan-out

event-model.md §5 已经定义"事件不丢"是硬约束——慢消费者必须 backpressure，不能 drop。technically 这意味着：

```text
                                 ┌─► acp_subscriber (mpsc bounded N)
                                 │
   主循环 ──► [fan-out task] ────┼─► storage_subscriber (mpsc bounded N)
                                 │
                                 └─► tracing_subscriber (mpsc bounded N)
```

实现要点：

- `subscribe()` 内部新建一个 `mpsc::channel(N)`，把发送端注册到 fan-out 表
- fan-out task：对每条事件，串行 `send().await` 到所有订阅者
- 一个慢订阅者填满了自己的 channel，`send().await` 在它身上阻塞，**主循环也跟着阻塞**——这是我们要的 backpressure：宁可慢，不能丢
- 订阅者 drop receiver 时从 fan-out 表里清掉

容量 N：v0 取 256，按实际监控调整。

为什么不用 `tokio::sync::broadcast`：broadcast 在订阅者跟不上时会标 `Lagged` 并跳过事件——直接违背"不丢事件"约束。容量调大可以减少概率但不解决问题。

## 6. ToolRegistry：进程级 + 会话级

```rust
pub trait ToolRegistry: Send + Sync {
    fn schemas(&self) -> Vec<ToolSchema>;
    fn get(&self, name: &str) -> Option<Arc<dyn Tool>>;
}
```

两层注册表：

- **进程级**：`AgentCore` 持有，内置工具（`defect-tools` 暴露的 fs / bash / grep ...）。无状态，`Arc<dyn Tool>` 可被所有 session 直接共享。注册名是裸名（`fetch` / `bash` / ...）。本地 `search` tool 仅在 session 启动时 `capabilities.search.mode = "local"` 才注入会话级。
- **会话级**：`Session` 持有，每个 session 自己的 MCP 工具（每个 MCP server 是 per-session 的子进程）。**所有 MCP 工具一律以 `mcp.<server>.<name>` 命名空间注册**——不区分名字、不区分 capability mode、不区分本地工具 enabled。详见 [`capabilities.md`](./capabilities.md) §6.2。

主循环通过 `Session` 暴露的 **composite registry** 查工具——`Session` 实现内部把进程级 + 会话级两个 registry 串起来（`get` 时先查会话级、再查进程级）。这样 turn 主循环只接触一个统一接口。

为什么不让 `Session` 本身实现 `ToolRegistry`：单一职责。`Session` 已经管 history / cancel / events 了，再加 schema 查询语义会让 trait 膨胀。组合优于继承。

## 7. 错误划分

### 7.1 AgentError —— `create_session` 用

```rust
pub enum AgentError {
    InvalidCwd(PathBuf),
    McpStartup { server: String, source: BoxError },
    Other(BoxError),
}
```

`create_session` 是少数几个接口级错误的地方。MCP 启动失败要带上 server 名字便于排障，因为 `mcp_servers` 列表里多个失败时上层得知道是哪个。

### 7.2 TurnError —— `run_turn` 的 fatal 退出

```rust
pub enum TurnError {
    TurnInProgress,
    Provider(ProviderError),
    Internal(BoxError),
}
```

划线规则：**只有"导致 turn 无法继续推进"的错误才进 TurnError**。

具体来说：

| 情况 | 归宿 |
| --- | --- |
| 用户 `session/cancel` | `Ok(StopReason::Cancelled)` |
| LLM 单次调用失败但可重试 | 主循环内重试，事件流发 `LlmCallFinished{error}`，不进 TurnError |
| 重试用尽仍失败 | `Err(TurnError::Provider(_))` |
| 模型拒绝输出（refusal） | `Ok(StopReason::Refusal)` |
| 工具执行失败 | `ToolCallFinished{status: Failed}` 事件，turn 继续，不进 TurnError |
| 主循环 invariant 被破坏（理应 bug） | `Err(TurnError::Internal(_))` |

`TurnInProgress` 表达"已经有一个 turn 在跑"，便于 acp 桥接转 `Error::InvalidRequest`。

## 8. 演进口子

- **多 turn 并发**：v0 拒绝。真要支持时换 trait 方法签名（返回 turn handle 而不是 future），不影响现有调用方。
- **session 持久化**：jsonl 持久化由 `defect-storage` 订阅事件流实现，与 Session trait 无关；resume 时由 `AgentCore` 实现回放事件流重建 `History` + `ToolRegistry`，trait 不动。
- **session/load**：trait 上加 `load_session(id) -> Result<Arc<dyn Session>, _>`；v0 不实现。
- **session/fork**：trait 上加 `fork_session(parent_id, ...) -> ...`；v0 不实现。
- **rewind / 回滚改动**：先不纳入本期 session 设计。它更像独立的 workspace snapshot / patch 回放能力，后续单独设计，不和 session 持久化混在一起。
- **远程 session**（gRPC 后的 `AgentCore` 实现）：trait 已经是 trait object，零改动。

## 9. 落地节奏

trait 已经在 `crates/agent/src/session.rs` 落地。具体实现按下列顺序：

1. `VecHistory`（trivial 实现）
2. `StaticToolRegistry`（注册表 + composite）
3. `DefaultSession`（持有 history / registry / 事件总线 / cancel token）
4. `DefaultAgentCore`（session 表 + 装配）
5. Turn 主循环（由 `DefaultSession::run_turn` 持有）

具体内部实现细节属于落地阶段，不进 trait 文档。
