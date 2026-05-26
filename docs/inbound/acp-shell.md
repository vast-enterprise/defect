# ACP Shell（Terminal）委托设计

ACP 的 `terminal/create` / `terminal/output` / `terminal/release` / `terminal/wait_for_exit` / `terminal/kill` 是**反向请求**（agent → client）：让 agent 把 shell 执行委托给客户端去做，而不是自己在 agent 进程内 `sh -c`。在 zed / vscode 这类有集成终端 UI 的客户端里，这条委托链让 agent 的 shell 操作在客户端的 PTY 里跑——用户能看到实时输出、能 Ctrl+C、客户端能做资源回收。

当前 `bash` 工具（[`tools-bash.md`]）是纯本地实现：`tokio::process::Command` spawn 到 `/bin/sh -c`。本文设计"shell 委托"模式——与 `fs` 委托（[`acp-fs.md`]）走同一架构，引入 [`ShellBackend`] trait 让工具层不感知后端。

设计原则：

1. **shell 委托是后端选择，工具实现不变**——改造后的 `bash` 工具只看 [`ShellBackend`] trait，由 [`defect-acp`] 在 session 创建时决定塞 [`LocalShellBackend`] 还是 [`AcpShellBackend`]，工具层完全不感知。
2. **能力协商是单向硬契约**——客户端在 `initialize` 里声明终端能力；agent 严格按清单选后端，不试探。
3. **保守降级**——客户端没声明完整 terminal 能力则整组退回本地。不混用（"create 走客户端、output 走本地"会导致状态撕裂）。
4. **工具层的工作区边界由 agent 自己守**——即便客户端有 PTY 隔离，agent 不依赖客户端的边界检查。
5. **v0 范围：非交互命令**——ACP terminal 协议是为交互式 PTY 设计的，但 v0 只用它跑非交互命令（`stdin=null`），与当前 `bash` 工具语义对齐。交互式 terminal 作为独立工具后续引入。

[`tools-bash.md`]: ../internal/tools-bash.md
[`acp-fs.md`]: ./acp-fs.md
[`ShellBackend`]: #3-shellbackend-抽象
[`LocalShellBackend`]: #4-localshellbackend
[`AcpShellBackend`]: #5-acpshellbackend
[`defect-acp`]: ./acp-bridge.md

## 1. 能力协商

### 1.1 ACP 的 terminal 能力

ACP `InitializeRequest` 携带 [`ClientCapabilities`]；v0 关注的 terminal 相关字段（schema 0.13.2，`v2/client.rs::ClientCapabilities`）：

```rust
pub struct ClientCapabilities {
    pub fs: FileSystemCapabilities,
    /// Whether the Client supports all `terminal/*` methods.
    pub terminal: bool,
    // ...
}
```

`terminal` 是单一 `bool`——schema 已经把"能否处理整组 terminal/* 反向请求"压成一个开关。v0 的决策粒度是"全有或全无"：声明 `true` 即用 [`AcpShellBackend`]，否则回退 [`LocalShellBackend`]。不逐方法拆分——与 `fs` 的"read+write 全才委托"逻辑一致（[`acp-fs.md` §1.2]）。

[`ClientCapabilities`]: https://docs.rs/agent-client-protocol-schema/0.13/agent_client_protocol_schema/v2/struct.ClientCapabilities.html
[`acp-fs.md` §1.2]: ./acp-fs.md#12-决策表

### 1.2 决策表

```
┌──────────────────────────────────────┬──────────────────────────────────┐
│ 客户端 terminal 能力                   │ defect-acp 装配的 ShellBackend   │
├──────────────────────────────────────┼──────────────────────────────────┤
│ 声明支持 terminal/*                   │ AcpShellBackend                 │
│ 未声明 / 字段缺失                      │ LocalShellBackend（降级）        │
└──────────────────────────────────────┴──────────────────────────────────┘
```

### 1.3 实现位置

```rust
// crates/acp/src/serve.rs
enum ShellMode { Local, Delegated }

fn decide_shell_mode(client_caps: &ClientCapabilities) -> ShellMode {
    if client_caps.terminal {
        ShellMode::Delegated
    } else {
        ShellMode::Local
    }
}
```

连接级状态存储：与 `FsMode` 并列存进 `ServeState`（[`serve.rs::ServeState`]）——`initialize` handler 写入，`session/new` / `session/load` handler 读取后构造对应的 `Arc<dyn ShellBackend>`，注入给 `AgentCore::create_session`。

[`serve.rs::ServeState`]: ../../crates/acp/src/serve.rs

## 2. ACP Terminal 反向请求：形状回顾

ACP wire 形态（基于 schema 0.13.2，`v1/client.rs`）：

### 2.1 `terminal/create`

```rust
pub struct CreateTerminalRequest {
    pub session_id: SessionId,
    pub command: String,
    pub args: Vec<String>,          // 默认空
    pub env: Vec<EnvVariable>,      // 默认空
    pub cwd: Option<PathBuf>,       // 绝对路径
    pub max_output_bytes: Option<u64>, // 输出字节上限，客户端负责截断
}

pub struct CreateTerminalResponse {
    pub terminal_id: TerminalId,
}
```

- **`command` + `args` 分离**——与 `bash` 工具的 `command: String`（走 `sh -c`）不同。`AcpShellBackend` 需要把 shell 行拆成 `command` + `args`。v0 方案：`command = "/bin/sh"`, `args = ["-c", user_command]`，保持与当前 `bash` 工具一致。
- **`cwd` 必须绝对**——agent 端先 `resolve_workspace_path` 校验边界，再塞进请求。
- **`max_output_bytes`**——默认不设（由客户端自行决定上限）；如果用户传了 `timeout_ms` 相关约束，不在此字段体现（超时由 agent 侧 `select!` 管理）。

### 2.2 `terminal/output`

```rust
pub struct TerminalOutputRequest {
    pub session_id: SessionId,
    pub terminal_id: TerminalId,
}

pub struct TerminalOutputResponse {
    pub output: String,
    pub truncated: bool,
    pub exit_status: Option<TerminalExitStatus>,
}
```

- **轮询模式**——agent 调 `terminal/output` 拿当前已累积的输出。可多次调用（中间取进度），或等到进程退出后一次性拿全量。
- `exit_status` 为 `None` 时表示进程还在跑。
- `truncated` 为 `true` 时表示客户端的输出已超过 `max_output_bytes` 上限被截断。

### 2.3 `terminal/wait_for_exit`

```rust
pub struct WaitForTerminalExitRequest {
    pub session_id: SessionId,
    pub terminal_id: TerminalId,
}

pub struct WaitForTerminalExitResponse {
    pub exit_status: TerminalExitStatus,
}
```

- **阻塞等待**——客户端在进程退出后才回复。agent 端在 `select!` 里 race `cancel.cancelled()` vs `wait_for_exit`。

### 2.4 `terminal/release`

```rust
pub struct ReleaseTerminalRequest {
    pub session_id: SessionId,
    pub terminal_id: TerminalId,
}
// response 无字段
```

- **幂等语义**——重复 release 同一个 terminal_id 不应报错。

### 2.5 `terminal/kill`

```rust
pub struct KillTerminalRequest {
    pub session_id: SessionId,
    pub terminal_id: TerminalId,
}
// response 无字段（ACP 用 impl_jsonrpc_request! 宏定义）
```

- **取消路径用**——`ctx.cancel` 触发时 agent 调 `kill` 而非直接 drop future。

## 3. `ShellBackend` 抽象

在 `defect-agent` 中定义，与 [`FsBackend`] 同层：

```rust
// crates/agent/src/shell.rs（新文件）

use std::path::PathBuf;
use futures::future::BoxFuture;

/// Shell 执行后端抽象。
///
/// v0 语义：每条命令创建独立 terminal，跑完后取输出再 release。
/// 不暴露"持久 terminal 跨 turn 复用"——等交互式 terminal 工具再做。
pub trait ShellBackend: Send + Sync {
    /// 创建 terminal 并启动命令。返回客户端分配的 terminal_id。
    fn create(
        &self,
        command: String,
        cwd: PathBuf,
    ) -> BoxFuture<'_, Result<TerminalId, ShellError>>;

    /// 轮询 terminal 当前输出。
    ///
    /// # 输出语义
    /// - 可多次调用（中间取进度）
    /// - `exit_status = Some(_)` 表示进程已退出
    fn output(
        &self,
        id: &TerminalId,
    ) -> BoxFuture<'_, Result<ShellOutput, ShellError>>;

    /// 阻塞等待 terminal 进程退出。
    fn wait_for_exit(
        &self,
        id: &TerminalId,
    ) -> BoxFuture<'_, Result<TerminalExitStatus, ShellError>>;

    /// 释放 terminal 资源。
    fn release(
        &self,
        id: &TerminalId,
    ) -> BoxFuture<'_, Result<(), ShellError>>;

    /// 强制终止 terminal 进程。
    fn kill(
        &self,
        id: &TerminalId,
    ) -> BoxFuture<'_, Result<(), ShellError>>;
}

#[derive(Debug, Clone)]
pub struct ShellOutput {
    pub text: String,
    pub truncated: bool,
    pub exit_status: Option<TerminalExitStatus>,
}

#[derive(Debug, Clone)]
pub struct TerminalExitStatus {
    /// 进程 exit code。被信号杀掉时为 `None`，看 `signal`。
    ///
    /// 内部用 `i32` 与 `BashOutput.exit_code`（[`tools-bash.md` §1]）一致；
    /// `AcpShellBackend` 收到 ACP schema 的 `Option<u32>` 时按位转 `i32`，
    /// 0..=i32::MAX 范围内值不变，超过的退化为 -1（实际 exit code 域是
    /// 0..=255，不会越界）。
    pub exit_code: Option<i32>,
    /// 信号名（如 `SIGKILL`）。本地后端来自 `signal_name(sig)`；
    /// ACP 后端直接透传 schema 的 `signal: Option<String>`。
    pub signal: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TerminalId(String);

impl TerminalId {
    pub fn new(id: impl Into<String>) -> Self { Self(id.into()) }
}

#[derive(Debug, thiserror::Error)]
pub enum ShellError {
    #[error("terminal not found: {0}")]
    NotFound(TerminalId),
    #[error("shell execution failed: {0}")]
    Execution(BoxError),
    #[error("backend error: {0}")]
    Backend(BoxError),
}
```

设计取舍：

- **`create` 不暴露 `args` / `env` / `max_output_bytes`**——v0 一律 `sh -c`，env 继承 agent 进程。与当前 `bash` 工具的字段取舍（[`tools-bash.md` §1]）完全一致。
- **`TerminalId` 是 newtype 而非 ACP schema 的 `TerminalId`**——解耦 agent 与 ACP 协议层。`LocalShellBackend` 用自己的 id 生成策略（进程 PID + 单调计数器），`AcpShellBackend` 映射到 schema 的 `TerminalId`。
- **`output` 与 `wait_for_exit` 分开**——前者是轮询（非阻塞），后者是阻塞。v0 的 `bash` 工具只用 `wait_for_exit`（跑完一次性取），`output` 留给后续流式输出需求。
- **不引入 `async fn`**——用 `BoxFuture` 与项目其他 trait 保持一致。

[`FsBackend`]: ../internal/tools-fs.md#2-fsbackend-抽象
[`tools-bash.md` §1]: ../internal/tools-bash.md#1-工具名片

## 4. `LocalShellBackend`

在 `defect-tools` 中实现，直接复用现有 `bash` 工具的进程管理逻辑：

```rust
// crates/tools/src/shell.rs（新文件）

use std::collections::HashMap;
use std::sync::Mutex;

pub struct LocalShellBackend {
    terminals: Mutex<HashMap<TerminalId, Arc<TerminalState>>>,
}

/// 单个 terminal 的运行态。reader task 持续把 stdout/stderr 灌进
/// `output`，`output()` / `wait_for_exit()` 读这份共享缓冲。
struct TerminalState {
    child: Mutex<tokio::process::Child>,
    output: Mutex<OutputBuffer>,
    /// reader task 关掉两个流后切到 `wait()`，最终把 ExitStatus 通过这个
    /// `Notify` + `Mutex<Option<...>>` 暴露出来。
    exit: tokio::sync::Mutex<Option<std::process::ExitStatus>>,
    exit_notify: tokio::sync::Notify,
    timed_out: AtomicBool,
}
```

实现要点：

- **`create`**：spawn `sh -c command`（同现有 [`tools-bash.md` §4.1]），`stdout`/`stderr` 走 `Stdio::piped()`，`kill_on_drop(true)`。立刻 `tokio::spawn` 一个 **reader task** 持续 `next_line()` 写入 `output` buffer（1 MiB 上限，与 [`tools-bash.md` §4.2] 同款），两个流都关闭后调 `child.wait()` 把 `ExitStatus` 写进 `exit` 并 `notify_waiters()`。terminal 状态存到 `HashMap`，返回 `TerminalId`。
- **`output`**：从 `output` buffer 取当前已累积文本，读 `exit` 是否已就位（已退出则填 `Some(...)`），返回 `ShellOutput`。**幂等可重复调用**——返回的是当时的快照，buffer 不被 drain。
- **`wait_for_exit`**：`exit_notify.notified()` 等到 reader task 写入 `exit`，再返回。
- **`release`**：从 `HashMap` 移除，drop `Arc<TerminalState>`（reader task 与 child 一并收尾）。
- **`kill`**：`child.start_kill()`，**不**移除 `HashMap` 项——`kill` 的语义是"强杀但不释放资源"，后续仍可调 `output` / `wait_for_exit`。释放由 `release` 负责。

> **为什么需要 reader task**：ACP 协议里 `output` 是一次性快照查询，但本地实现必须持续从 piped fd 读出来——否则一旦子进程输出超过 pipe buffer 就会被阻塞写。这点与现有 bash 工具的"边跑边读 + select! 同步聚合"等价，只是把读循环从工具的主 future 里搬到了 backend 的 reader task。

与现有 `bash` 工具的关系：改造 `BashTool`——去掉内部的 `Command::new("sh")`，改为通过 [`ToolContext::shell`] 拿后端调 `create` / `wait_for_exit` / `output` / `release`。BashTool 仍然无状态（[`tools-bash.md` §1]），后端依赖走 ctx 注入而非 struct 字段——与 fs 后端取舍一致（[`crate::tool::ToolContext::fs`]）。

[`tools-bash.md` §4.1]: ../internal/tools-bash.md#41-进程生成
[`tools-bash.md` §4.2]: ../internal/tools-bash.md#42-输出捕获策略
[`ToolContext::shell`]: ../../crates/agent/src/tool.rs
[`crate::tool::ToolContext::fs`]: ../../crates/agent/src/tool.rs

## 5. `AcpShellBackend`

完全镜像 [`AcpFsBackend`]（[`acp-fs.md` §3]）：

```rust
// crates/acp/src/shell.rs（新文件）

use agent_client_protocol::schema::{
    CreateTerminalRequest, KillTerminalRequest, ReleaseTerminalRequest,
    TerminalId as AcpTerminalId, TerminalOutputRequest, SessionId,
};
use agent_client_protocol::{Client, ConnectionTo};

pub struct AcpShellBackend {
    cx: ConnectionTo<Client>,
    session_id: SessionId,
    workspace_root: PathBuf,
}

impl ShellBackend for AcpShellBackend {
    fn create(&self, command: String, cwd: PathBuf) -> BoxFuture<'_, Result<TerminalId, ShellError>> {
        Box::pin(async move {
            let abs_cwd = resolve_workspace_path(&self.workspace_root, &cwd)?;
            let req = CreateTerminalRequest::new(self.session_id.clone(), "/bin/sh")
                .args(vec!["-c".into(), command])
                .cwd(abs_cwd);
            let resp = self.cx.send_request(req).block_task().await
                .map_err(map_wire_error)?;
            Ok(TerminalId::new(resp.terminal_id.to_string()))
        })
    }

    fn output(&self, id: &TerminalId) -> BoxFuture<'_, Result<ShellOutput, ShellError>> {
        Box::pin(async move {
            let req = TerminalOutputRequest::new(
                self.session_id.clone(),
                AcpTerminalId::new(id.0.clone()),
            );
            let resp = self.cx.send_request(req).block_task().await
                .map_err(map_wire_error)?;
            Ok(ShellOutput {
                text: resp.output,
                truncated: resp.truncated,
                exit_status: resp.exit_status.map(map_acp_exit_status),
            })
        })
    }

    // wait_for_exit / release / kill 同理
}

/// 把 ACP schema 的 `TerminalExitStatus`（`exit_code: Option<u32>`）映射到
/// agent 内部的 `TerminalExitStatus`（`exit_code: Option<i32>`）。
///
/// exit code 的标准值域是 0..=255（POSIX）/ 0..=u32::MAX（Windows GetExitCodeProcess）；
/// 实际用户语义只取 0..=255。i32 域足够装下，超过 i32::MAX 退化为 -1。
fn map_acp_exit_status(s: agent_client_protocol::schema::TerminalExitStatus) -> TerminalExitStatus {
    TerminalExitStatus {
        exit_code: s.exit_code.map(|n| i32::try_from(n).unwrap_or(-1)),
        signal: s.signal,
    }
}
```

- **`ConnectionTo<Client>` 是 `Clone`**——它是 `Arc<...>` 的 newtype，`AcpShellBackend` 持一份即可。
- **`block_task`** 是 ACP SDK 的 await 模型。
- **工作区边界**：agent 自己再调一次 `resolve_workspace_path` 校验（与 [`acp-fs.md` §4] 一致）——bash 工具层已经用 `resolve_workdir` 做过一次。这是有意的"双层栅栏"：fs 工具家族不在工具层做边界（路径直传 backend），shell 在工具层做了一次（基于现状），backend 层再校验一次让 `AcpShellBackend` / `AcpFsBackend` 的安全保证对称。落地时校验通过的成本可忽略，多一道栅栏避免未来加新 shell 工具时漏掉边界。

[`AcpFsBackend`]: ./acp-fs.md#3-acpfsbackend
[`acp-fs.md` §3]: ./acp-fs.md#3-acpfsbackend
[`acp-fs.md` §4]: ./acp-fs.md#4-工作区边界agent-自我约束

## 6. `bash` 工具改造

### 6.1 变更点

当前 `BashTool` 的 `execute` 直接调 `tokio::process::Command::new("sh")` ([`tools-bash.md` §4])。改造后 `BashTool` struct **不变**——仍然只持 `schema` + 两个 timeout 配置（保持 [`tools-bash.md` §1] "工具单例无状态"的取舍）。后端依赖通过新增的 [`ToolContext::shell`] 注入：

```rust
// crates/agent/src/tool.rs（扩展 ToolContext）
pub struct ToolContext<'a> {
    pub cwd: &'a Path,
    pub cancel: CancellationToken,
    pub fs: Arc<dyn FsBackend>,
    pub shell: Arc<dyn ShellBackend>,   // 新增
}
```

`execute` 流程变为：

```text
1. 解析 args（command / workdir / timeout_ms）
2. resolve_workdir(ctx.cwd, args.workdir) → 边界校验（沿用 bash.rs 现有逻辑）
3. ctx.shell.create(command, cwd) → 拿到 terminal_id
4. 主 select!：
   ├── cancel：shell.kill(id); shell.release(id); ToolEvent::Failed(Canceled)
   ├── 超时：shell.kill(id); shell.wait_for_exit(id);
   │         shell.output(id); shell.release(id); ToolEvent::Completed(timeout=true)
   └── 正常：shell.wait_for_exit(id); shell.output(id); shell.release(id);
             ToolEvent::Completed(...)
5. release 在所有分支上幂等执行（包括失败路径）
```

工具的行为与现有 `bash` 完全一致——LLM 感知不到后端差异。

> **路径校验留在工具层**：当前 bash 自己实现了 `resolve_workdir`（bash.rs:347-377），与 fs 工具家族用的 `resolve_workspace_path`（[`agent/src/fs.rs`]）几乎重叠。本次改造**不**合并这两个函数——bash 的 workdir 必须存在（要 canonicalize 整路径），fs 的目标可能尚未存在（只 canonicalize 父目录）。差异够大，强行合并反而割裂。

### 6.2 不变的部分

- `safety_hint` 仍然返回 `Destructive`（[`tools-bash.md` §2]）
- `describe` 仍然产 `title="$ command"`、`kind=Execute`
- 输出格式仍然合并 stdout/stderr，1 MiB 上限
- 非零退出仍然是 `Completed`（不是 `Failed`）
- 超时 / 取消行为一致

[`tools-bash.md` §2]: ../internal/tools-bash.md#2-安全等级safety_hint
[`tools-bash.md` §4]: ../internal/tools-bash.md#4-execute
[`agent/src/fs.rs`]: ../../crates/agent/src/fs.rs

## 7. e2e 测试

放在 `crates/acp/tests/shell_delegation.rs`。结构与现有 `fs_delegation.rs` 一致。

| # | 场景 | 验证 |
| --- | --- | --- |
| 1 | 客户端声明 terminal 能力 → 跑 `bash` 工具 `echo hello` | 收到一条 `terminal/create` 反向请求；agent 正确拿到 output；`ToolCallFinished` |
| 2 | 同上 → 命令以非零退出 | `terminal/output` 返回 exit_code≠0；agent 产出 `Completed`（非 `Failed`） |
| 3 | 同上 → 超时 | agent 发 `terminal/kill`；`Completed` 含 timeout 信息 |
| 4 | 同上 → turn 中途 cancel | agent 发 `terminal/kill` + `terminal/release`；`ToolCallFinished` status=Failed（`ToolError::Canceled`），随后 `TurnEnded = Cancelled` |
| 5 | 客户端未声明 terminal 能力 → 跑 `bash` | 不发 `terminal/*` 请求（退回 LocalShellBackend）；命令正常执行 |
| 6 | 委托模式下 workdir 越界 | agent 端 `resolve_workdir` 报错（`ToolError::InvalidArgs`）；**不**发 `terminal/create` |
| 7 | 委托模式下 client 返回 wire error | `ToolCallFinished` status=Failed；content 含 wire error 信息 |

## 8. 后续演进（不在 v0）

- **交互式 terminal 工具**——新增独立的 `terminal` 工具，暴露 `create` / `input` / `output` / `release`，让 LLM 能管理持久 PTY session（对位 ACP `terminal/create` 的交互式用途）。
- **流式输出**——当前 `bash` 工具单发 `Completed`（[`tools-bash.md` §4.2]）。委托模式下可以在 `wait_for_exit` 之前多次轮询 `output` 发 `Progress`，实现边跑边看。
- **后台任务**——terminal 生命周期跨 turn 存活，让 LLM 在后续 turn 里通过 `terminal/output` 查看后台任务进度。
- **argv 模式**——当客户端 terminal 能力就绪后，考虑让 `AcpShellBackend` 直接传 `command` + `args` 而非走 `sh -c`，从协议层规避 shell 注入。
- **env 透传**——当前 `bash` 工具不暴露 `env` 字段（[`tools-bash.md` §1]）。如果客户端 terminal 支持 env 隔离，可以扩展 `ShellBackend::create` 加 `env` 参数。
- **`session/load` 与 terminal**——恢复 session 时如果有未释放的 terminal，怎么处理？v0 不做 session/load 持久化（[`acp-bridge.md` §7]），延后。

[`tools-bash.md` §1]: ../internal/tools-bash.md#1-工具名片
[`tools-bash.md` §4.2]: ../internal/tools-bash.md#42-输出捕获策略
[`acp-bridge.md` §7]: ./acp-bridge.md#7-v0-不做的事明确划线

## 9. 落地步骤

1. **`crates/agent/src/shell.rs`**（新文件）：[`ShellBackend`] trait + `ShellError` + `ShellOutput` + `TerminalId` + `TerminalExitStatus` + `NoopShellBackend`（测试占位，与 `NoopFsBackend` 同款）
2. **`crates/agent/src/lib.rs`**：`pub mod shell;`
3. **`crates/agent/src/tool.rs`**：`ToolContext` 加 `pub shell: Arc<dyn ShellBackend>`，`ToolContext::new` 签名扩展（破坏式变更——会牵动现有所有 `ToolContext::new` 调用点，包括测试 fixture）
4. **`crates/agent/src/session/default.rs`**：`AgentCore::create_session` / `load_session` 增加 `shell: Arc<dyn ShellBackend>` 参数；`DefaultSession` 持有并在构造 `ToolContext` 时透传
5. **`crates/tools/src/shell.rs`**（新文件）：[`LocalShellBackend`] 实现（含 reader task）
6. **`crates/tools/src/bash.rs`**：改造 `execute`——去掉 `Command::new("sh")`，改为 `ctx.shell.create(...)` + `wait_for_exit` + `output` + `release`。`BashTool` struct 字段不变
7. **`crates/acp/src/shell.rs`**（新文件）：[`AcpShellBackend`] 实现（镜像 `acp/src/fs.rs`）
8. **`crates/acp/src/serve.rs`**：
   - `ServeState` 加 `shell_mode: RwLock<ShellMode>`
   - `on_initialize` 调 `decide_shell_mode(&req.client_capabilities)` 写入
   - `on_session_new` / `on_session_load` 按 `ShellMode` 构造 `Arc<dyn ShellBackend>`，传给 `create_session` / `load_session`
   - 提取 `ServeState::shell_backend(...)` 方法（与 `fs_backend` 同款）
9. **`crates/acp/tests/shell_delegation.rs`**：跑 §7 测试矩阵
10. **文档**：更新 `docs/internal/tools-bash.md` §8 的演进口子（把"ACP terminal 委托"从 v0 不做到已落地），更新 `TODO.MD`

## 10. 与现有文档的协同更新

落地时同步更新：

- **`docs/internal/tools-bash.md`**：§8 "v0 不做"中"ACP terminal 委托"项改为已完成；§4 加注说明后端可替换
- **`docs/inbound/acp-bridge.md`**：§2 表格加 `terminal/*` 方法行，指向本文
- **`docs/architecture.md`**：crate 依赖图加 `defect-acp → ShellBackend` 边
- **`TODO.MD`**：shell 委托项完成后翻到「已完成」
