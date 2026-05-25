# 无头 ACP 客户端设计

目标：为 `defect` 的端到端测试提供一个**无 UI、可断言**的 ACP client 使用方案。

它解决的不是“如何再写一个前端”，而是“如何稳定地驱动 `defect`，并验证 wire 上真实发生了什么”。

当前结论优先级很明确：

1. **先直接用 `agent-client-protocol` 原库写测试**
2. **只有出现稳定重复样板时，再提炼极薄 helper**
3. **不要先做一层新的 headless client 库**

## 1. 设计目标

无头客户端需要同时满足四件事：

1. **驱动真实 ACP 会话**：至少覆盖 `initialize` / `session/new` / `session/load` / `session/prompt` / `session/cancel`。
2. **接住 agent→client 反向请求**：至少覆盖 `session/request_permission` 与 `fs/read_text_file` / `fs/write_text_file`。
3. **记录并断言事件流**：测试应能检查 `SessionUpdate`、`PromptResponse`、wire error，而不是只看“命令有没有退出”。
4. **同时支持两种传输**：
   - 进程内 `Channel::duplex`：快，适合 crate 级集成测试。
   - 子进程 `stdio`：真，适合 `defect-cli` 黑箱 e2e。

非目标：

- 不做交互式 UI。
- 不做面向最终用户的通用 ACP client。
- 不手写 JSON-RPC parser；优先复用 `agent-client-protocol` SDK。

## 2. 总体方案

首选方案不是新 crate，而是直接复用 ACP SDK 自带能力：

- `Client.builder()`
- `AcpAgent`
- `ByteStreams`
- `ConnectionTo::build_session(...)`
- `ActiveSession::send_prompt(...)`
- `ActiveSession::read_to_string(...)`

只有在以下重复稳定出现时，才考虑补一个内部 helper：

- `Channel` / `stdio` 传输接线样板
- notification 收集
- 反向请求 handler 样板

如果后续确实需要 helper，推荐它只是下面这些原语的组合，而不是一个新协议抽象层。

核心对象可以收敛为：

```text
Scenario
  ├─ ClientCapabilities
  ├─ ReverseRequestScript
  ├─ PromptSteps
  └─ Expectations

HeadlessAcpClient
  ├─ transport
  ├─ recorder
  ├─ fs_delegate
  └─ permission_delegate
```

设计原则：

- **协议层复用 SDK**：用 `agent_client_protocol::Client` builder 注册 request / notification handler。
- **测试层提供脚本 DSL**：测试写“收到权限请求时返回 AllowOnce”，而不是自己拼 handler。
- **记录优先于打印**：所有 request / response / notification 都进 recorder，断言基于结构化记录。

## 3. 模块拆分

如果未来重复足够多，再拆模块；在那之前，优先把测试直接写在调用点。

### 3.1 `transport/`

职责：把“怎么连上 agent”抽象掉。

首批实现：

- `ChannelTransportHarness`
  - 基于 `Channel::duplex`
  - 给 `defect_acp::serve_on(...)` 的白盒/灰盒测试用
- `ChildProcessTransportHarness`
  - 优先直接用 ACP 原库自带的 `AcpAgent`
  - stdin/stdout 接 ACP，stderr 单独收日志
  - 给真实 `defect-cli` 黑箱测试用

这里的关键不是定义 trait，而是把**传输差异收敛到最外层**。如果 `AcpAgent` 和 `Channel` 已经够用，就不再包装一层。

### 3.2 `script/`

职责：描述测试想让 client 怎么表现。

但首版不应先做完整脚本 DSL。先直接写 handler，等重复显著时再提炼。

建议对象：

- `ClientScript`
  - `capabilities: ClientCapabilities`
  - `fs: FsScript`
  - `permissions: PermissionScript`
  - `notifications: NotificationScript`

- `FsScript`
  - `read(path) -> outcome`
  - `write(path, content) -> outcome`
  - 支持“固定返回”“按顺序消费脚本”“闭包动态计算”三种模式

- `PermissionScript`
  - `allow_once`
  - `deny_once`
  - `cancel_once`
  - `by_tool_name(...)`
  - `by_safety_class(...)` 如果后面 wire 上能稳定拿到足够信息

脚本层应允许“未声明的调用直接失败”，避免测试静默通过。

### 3.3 `recorder/`

职责：记录所有可观测事实，供断言使用。

建议记录：

- `InitializeResponse`
- `NewSessionResponse` / `LoadSessionResponse`
- 每条 `SessionNotification`
- 每次反向请求：
  - `RequestPermissionRequest`
  - `ReadTextFileRequest`
  - `WriteTextFileRequest`
- `PromptResponse`
- JSON-RPC error
- 子进程模式下的 `stderr`

建议结构：

```rust
struct Transcript {
    notifications: Vec<SessionNotification>,
    permission_requests: Vec<RequestPermissionRequest>,
    fs_reads: Vec<ReadTextFileRequest>,
    fs_writes: Vec<WriteTextFileRequest>,
    prompt_outcomes: Vec<Result<PromptResponse, agent_client_protocol::Error>>,
    stderr: String,
}
```

`Transcript` 应提供高层 helper，例如：

- `assistant_text()`
- `tool_calls_named("bash")`
- `last_stop_reason()`
- `expect_permission_requested()`

### 3.4 `runner/`

职责：把脚本、传输、录制器串起来，跑完整场景。

建议入口：

```rust
async fn run_scenario(
    transport: impl AcpTestTransport,
    scenario: Scenario,
) -> Result<ScenarioOutcome, TestClientError>
```

`Scenario` 建议支持多步：

1. `initialize`
2. `new_session` 或 `load_session`
3. 一次或多次 `prompt`
4. 可选 `cancel`
5. 收集 transcript

这样后面测试 resume、多轮上下文、取消时，不需要每个测试手搓流程。

## 4. 为什么不用“直接手写 JSON 行”

不建议自己维护一个低层 JSON-RPC 客户端。理由：

1. 仓库已经使用 `agent-client-protocol` crate，继续复用能减少 schema 漂移。
2. 本项目真正要测的是 **Defect 对 ACP 语义的实现**，不是 JSON 编解码。
3. 反向请求很多，自己维护 request id、response race、cancel 语义很容易偏离协议。

唯一值得保留的“低层模式”是：

- 在子进程黑箱测试里，允许额外记录原始 stdin/stdout 文本，作为 debug artifact；
- 但驱动逻辑本身仍走 SDK。

## 5. 关键协议能力

首版必须支持：

### 5.1 正向请求

- `initialize`
- `session/new`
- `session/load`
- `session/prompt`
- `session/cancel`

### 5.2 反向请求

- `session/request_permission`
- `fs/read_text_file`
- `fs/write_text_file`

### 5.3 通知

- `session/update`

其中 `session/update` 需要重点断言这些变体：

- `AgentMessageChunk`
- `AgentThoughtChunk`
- `ToolCall`
- `ToolCallUpdate`

## 6. 断言模型

建议不要一开始就补高层断言助手。首版先允许测试直接遍历原始记录。

建议能力：

- 断言 turn 终态
  - `expect_stop_reason(EndTurn | Cancelled | MaxTokens | ...)`
- 断言 assistant 文本包含某片段
- 断言至少发生一次某工具调用
- 断言某工具调用最终 `Completed` / `Failed`
- 断言出现权限请求，且用户选择被正确投射
- 断言 fs 委托确实走了 client，而不是回退到本地盘
- 断言子进程 stderr 不含意外 panic / backtrace

这层 helper 不应该替代原始 transcript，而是建立在 transcript 之上。

## 7. 两级测试策略

无头客户端建好后，测试分两层：

### 7.1 crate 内灰盒测试

位置：

- `crates/acp/tests/*.rs`
- 未来需要时的 `crates/cli/tests/*.rs`

特点：

- transport 用 `Channel::duplex`
- server 直接 `serve_on(...)`
- provider / tool / policy 可注入假实现
- 速度快，定位精确

适合验证：

- ACP 桥接逻辑
- 权限桥
- fs 委托
- cancel 边界
- error projection

### 7.2 CLI 黑箱 e2e

位置：

- 根目录 `tests/*.rs`

特点：

- transport 用子进程 stdio
- 跑真实 `defect` 二进制
- 配合假 LLM server / test container / 临时目录

适合验证：

- 参数装配
- tracing 到 stderr，不污染 stdout wire
- 真实工作目录、真实文件落盘
- CLI 配置与 ACP server 的整链闭环

## 8. 建议的最小实现顺序

### Phase 1：先写真实 `stdio` 黑箱测试

直接用：

- `AcpAgent`
- `Client.builder()`
- `build_session(...)`
- request / notification handler

目标：证明原库 API 已经足够简单，至少能无额外包装地跑通 `defect`。

### Phase 2：仅在重复出现后提炼 helper

加入：

- notification recorder
- permission handler helper
- fs handler helper

目标：让权限/文件系统类测试不再各写一套 handler，但 helper 仍然紧贴 ACP 原语。

## 9. 具体落点建议

当前建议新增的首先不是 crate，而是黑箱测试：

- `tests/acp_stdio_smoke.rs`
- `tests/acp_permission_roundtrip.rs`
- `tests/acp_fs_delegation_stdio.rs`

## 10. 风险与边界

### 10.1 不要把它做成“第二个 ACP SDK”

这个库的职责是**测试编排**，不是再封装一遍完整协议客户端。

### 10.2 不要把实现绑死在 `defect-cli`

子进程模式需要能接“任意实现了 ACP over stdio 的 agent 可执行文件”，这样它既能测 `defect`，也能对照别的实现做兼容性测试。

### 10.3 Cancel 语义必须按真实协议跑

尤其是 pending `request_permission` 与 fs 反向请求期间的取消，不能在 harness 里“假装同步完成”，否则测不出真正的 race。

## 11. 最终建议

结论：

1. **优先直接用 `agent-client-protocol` SDK 写测试**。
2. **不要先做新的 headless client 库**。
3. **先补真实子进程 `stdio` 黑箱测试**，再看哪些样板值得抽象。
4. **如果要抽象，只抽 transport / recorder / 反向请求 handler 这三类最小 helper**。

这样做之后，`defect` 的 ACP 端到端测试会形成一条清晰分层：

- 内层：`Channel` 灰盒，快
- 外层：`stdio` 黑箱，真
- 中间共享同一套无头 client 脚本与断言模型
