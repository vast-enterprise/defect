# ACP 文件系统委托设计

ACP 的 `fs/read_text_file` / `fs/write_text_file` 是**反向请求**（agent → client）：让 agent 把文件 IO 委托给客户端去做，而不是直接打本地盘。在 zed / vscode 这类有 workspace UI 的客户端里，这条委托链让"agent 改文件"和"用户编辑器看到 unsaved buffer"对齐。

本文沉淀这套委托的形状：[`FileSystemCapabilities`] 协商、[`AcpFsBackend`] 的实现、与 [`docs/internal/tools-fs.md`] 的对接路径。**与 fs 内置工具一次性 v0 落地**——不留"先本地、再委托"的尾巴，否则装配代码会在两次落地之间残留分叉。

设计原则：

1. **fs 委托是后端选择，工具实现不变**——三个 fs 工具（`read_file` / `write_file` / `edit_file`）只看 [`FsBackend`] trait，由 [`defect-acp`] 在 session 创建时决定塞 [`LocalFsBackend`] 还是 [`AcpFsBackend`]，工具层完全不感知。
2. **能力协商是单向硬契约**——客户端在 `initialize` 里说自己支持哪些 fs 操作；agent 严格按这个清单选后端，不"试探性发请求看会不会失败"。
3. **保守降级**——客户端没声明 `read_text_file` 但声明了 `write_text_file`（罕见但 ACP 允许），v0 直接退回 [`LocalFsBackend`] 整组（不混用）。混用需求出现再细化。
4. **工具层的工作区边界仍由 agent 自己守**——即便客户端可能 enforce 一遍，agent 不依赖客户端的边界检查（防止 LLM 让客户端做奇怪的事）。
5. **范围限定为 ACP 已支持的 fs 方法**——schema 0.13.2 仅有 `fs/read_text_file` / `fs/write_text_file` 两个反向请求，**没有** `fs/delete_*` / `fs/move_*` / `fs/create_dir`。所以 fs 工具家族也不引入这些操作（[`tools-fs.md` §1.2](../internal/tools-fs.md#12-不进-v0-的工具与原因)），LLM 走 [`bash`](../internal/tools-bash.md) (`rm` / `mv` / `mkdir`)。

[`FileSystemCapabilities`]: https://docs.rs/agent-client-protocol-schema/0.13/agent_client_protocol_schema/v1/struct.FileSystemCapabilities.html
[`docs/internal/tools-fs.md`]: ../internal/tools-fs.md
[`FsBackend`]: ../internal/tools-fs.md#2-fsbackend-抽象
[`LocalFsBackend`]: ../internal/tools-fs.md#21-localfsbackend
[`AcpFsBackend`]: #3-acpfsbackend
[`defect-acp`]: ./acp-bridge.md

## 1. 能力协商

### 1.1 ACP 的 `ClientCapabilities.fs`

ACP `InitializeRequest` 携带 [`ClientCapabilities`]，其中 `fs` 字段是 [`FileSystemCapabilities`]：

```rust
pub struct FileSystemCapabilities {
    pub read_text_file: bool,
    pub write_text_file: bool,
    pub meta: Option<Meta>,
}
```

两个 bool 是**独立**的——客户端可能只支持读不支持写（例如只读视图）。语义：

| `read_text_file` | `write_text_file` | 客户端意图 |
| --- | --- | --- |
| `true` | `true` | 完全接管（zed / vscode 典型） |
| `true` | `false` | 只读视图（log viewer 类） |
| `false` | `true` | （ACP 允许但少见——用户期望 agent 写入但不显示已读？）|
| `false` | `false` | 不接管，agent 用本地盘 |

[`ClientCapabilities`]: https://docs.rs/agent-client-protocol-schema/0.13/agent_client_protocol_schema/v1/struct.ClientCapabilities.html

### 1.2 决策表

```
┌──────────────────────────────────────┬──────────────────────────────┐
│ ClientCapabilities.fs                │ defect-acp 装配的 FsBackend  │
├──────────────────────────────────────┼──────────────────────────────┤
│ { read: true,  write: true  }        │ AcpFsBackend                 │
│ { read: true,  write: false }        │ LocalFsBackend（降级）       │
│ { read: false, write: true  }        │ LocalFsBackend（降级）       │
│ { read: false, write: false }        │ LocalFsBackend               │
│ 字段缺失 / fs = None                  │ LocalFsBackend               │
└──────────────────────────────────────┴──────────────────────────────┘
```

**不做混合后端**——只要有任意一项 false，整组退回本地。理由：

1. 混合后端语义复杂（"读走客户端、写走本地"会让客户端的 unsaved buffer 与磁盘出现 staleness）；
2. 真实场景里客户端要么全接、要么全不接，"半接"是 corner case；
3. 即便客户端真的需要"只读不写"，让它显式声明就好，本设计不预投资。

演进项见 §6。

### 1.3 实现位置

```rust
// crates/acp/src/serve.rs::initialize handler
async fn initialize(req: InitializeRequest, ...) -> Result<...> {
    let fs_mode = decide_fs_mode(req.client_capabilities.as_ref());
    // 把 fs_mode 存到 connection 级状态——session/new 时读出来。
    state.set_fs_mode(fs_mode);
    Ok(InitializeResponse::new(req.protocol_version)
        .agent_capabilities(agent_capabilities()))
}

fn decide_fs_mode(client_caps: Option<&ClientCapabilities>) -> FsMode {
    match client_caps.and_then(|c| c.fs.as_ref()) {
        Some(fs) if fs.read_text_file && fs.write_text_file => FsMode::Delegated,
        _ => FsMode::Local,
    }
}

enum FsMode { Local, Delegated }
```

`agent_capabilities()` 不变——agent 自己声明的 capabilities（`AgentCapabilities`）与客户端声明的 fs 能力是**两个不同方向**的协商，agent 这边只声明"我会做哪些 client→agent 方法"。

### 1.4 connection 状态：`fs_mode` 存哪？

ACP 一次 stdio 连接对应一个 `connection_id`；`session/new` 创建的 session 都挂在这条连接下。`FsMode` 是 connection 级（initialize 里协商一次，所有 session 共用），不是 session 级。

存放选项：

- **A：`AgentCore` 上挂 `Mutex<HashMap<ConnectionId, FsMode>>`**——简单粗暴，但污染了 `defect-agent` 的接口（它不该知道连接级状态）。
- **B：`defect-acp` 内部的 `Arc<RwLock<...>>`**——`fs_mode` 仅 `defect-acp` 内部用，符合分层。
- **C：把 `FsBackend` 直接附在 `session/new` 调用里**——`AgentCore::create_session` 接受 `Arc<dyn FsBackend>` 参数，由 `defect-acp` 在 handler 里构造对应后端。

**选 C**：让 `AgentCore` 的接口显式表达"session 用什么 fs 后端"，而 `defect-acp` 内部只需要在 connection 级缓存协商结果。`AgentCore::create_session` 加两个参数：

```rust
pub trait AgentCore {
    fn create_session(
        &self,
        id: SessionId,                       // 新增：见 §3.2 的时序问题
        cwd: PathBuf,
        mcp_servers: Vec<McpServer>,
        fs: Arc<dyn FsBackend>,              // 新增
    ) -> BoxFuture<'_, Result<Arc<dyn Session>, AgentError>>;
    ...
}
```

`defect-acp::serve` 在 `session/new` handler 里：

```rust
let session_id = SessionId::new(uuid_like());
let fs: Arc<dyn FsBackend> = match connection_state.fs_mode() {
    FsMode::Delegated => Arc::new(AcpFsBackend::new(
        cx.clone(),
        session_id.clone(),
        req.cwd.clone(),
    )),
    FsMode::Local => Arc::new(LocalFsBackend::new(req.cwd.clone())),
};
let session = agent.create_session(
    session_id,
    req.cwd,
    req.mcp_servers,
    fs,
).await?;
```

权衡：`AgentCore` 接口表面变宽，但语义明朗——session 持有的 fs 后端是显式注入的。

## 2. ACP 反向请求：形状回顾

### 2.1 `fs/read_text_file`

ACP wire 形态（schema 0.13.2）：

```rust
pub struct ReadTextFileRequest {
    pub session_id: SessionId,
    pub path: PathBuf,           // ACP 规定必须是绝对路径
    pub line: Option<u32>,       // 1-based 起始行
    pub limit: Option<u32>,      // 最大行数
    pub meta: Option<Meta>,
}

pub struct ReadTextFileResponse {
    pub content: String,
    pub meta: Option<Meta>,
}
```

- **路径必须绝对**——见 schema 0.13.2 `ReadTextFileRequest::path` 注释 `Absolute path to the file to read`。
- `line` / `limit` 可选；都缺省时返回全文。
- 客户端可能返回错误（路径越界 / 权限拒绝 / 不存在）；错误用标准 JSON-RPC `Error` 字段，没有专用 ErrorCode。

### 2.2 `fs/write_text_file`

```rust
pub struct WriteTextFileRequest {
    pub session_id: SessionId,
    pub path: PathBuf,
    pub content: String,
    pub meta: Option<Meta>,
}

pub struct WriteTextFileResponse {
    pub meta: Option<Meta>,
}
```

- **全量覆盖语义**——与 [`write_file`](../internal/tools-fs.md#4-write_file-工具) 工具一致。
- 客户端决定要不要 mkdir-p、要不要 atomic-replace、行末符如何处理；agent 不感知（[`AcpFsBackend`] 因此不重复 [`tools-fs.md` §6](../internal/tools-fs.md#6-localfsbackend-的正确性细节) 的本地侧细节）。

## 3. `AcpFsBackend`

### 3.1 形状

```rust
// crates/acp/src/fs.rs（新文件）

use std::path::{Path, PathBuf};

use agent_client_protocol::schema::{
    ReadTextFileRequest, SessionId, WriteTextFileRequest,
};
use agent_client_protocol::{Client, ConnectionTo};
use defect_agent::error::BoxError;
use defect_agent::fs::{FsBackend, FsError, resolve_workspace_path};
use futures::future::BoxFuture;

pub struct AcpFsBackend {
    cx: ConnectionTo<Client>,
    /// session id 在 backend 构造时已知（见 §3.2 的时序方案）。
    session_id: SessionId,
    /// 工作区 root，用于 agent 自身的边界校验（§4）。
    workspace_root: PathBuf,
}

impl AcpFsBackend {
    pub fn new(cx: ConnectionTo<Client>, session_id: SessionId, workspace_root: PathBuf) -> Self {
        Self { cx, session_id, workspace_root }
    }
}

impl FsBackend for AcpFsBackend {
    fn read_text(
        &self,
        path: PathBuf,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> BoxFuture<'_, Result<String, FsError>> {
        Box::pin(async move {
            let abs = resolve_workspace_path(&self.workspace_root, &path)?;
            let mut req = ReadTextFileRequest::new(self.session_id.clone(), abs);
            if let Some(l) = line { req = req.line(l); }
            if let Some(k) = limit { req = req.limit(k); }
            let resp = self
                .cx
                .send_request(req)
                .block_task()
                .await
                .map_err(map_wire_error)?;
            Ok(resp.content)
        })
    }

    fn write_text(
        &self,
        path: PathBuf,
        content: String,
    ) -> BoxFuture<'_, Result<(), FsError>> {
        Box::pin(async move {
            let abs = resolve_workspace_path(&self.workspace_root, &path)?;
            let req = WriteTextFileRequest::new(self.session_id.clone(), abs, content);
            self
                .cx
                .send_request(req)
                .block_task()
                .await
                .map_err(map_wire_error)?;
            Ok(())
        })
    }
}
```

- **`BoxFuture<'_, ...>` 而非 `async fn`**——本仓库不引 `async-trait` 宏，与 `LlmProvider::complete` / `AgentCore::create_session` 同形态。
- **`ConnectionTo<Client>` 是 `Clone`**——它是 `Arc<...>` 的 newtype（`agent-client-protocol` 已实现 `Clone`），所以 `AcpFsBackend` 持一份就行，不再 `Arc` 套娃。
- **`block_task`** 是 ACP SDK 的 await 模型；与 [`spawn_permission_request`](./acp-bridge.md#4-请权限的双向流程) 同源。
- **`map_wire_error`**：见 §3.4。

### 3.2 session_id 时序

`session/new` handler 的执行顺序：

```
1. handler 收到 NewSessionRequest
2. (此时还没有 SessionId)
3. 构造 AcpFsBackend → 需要 SessionId
4. agent.create_session(id, cwd, mcp, fs) → 内部把 session 装进 DashMap
5. response 返回 SessionId 给客户端
6. (之后客户端开始用这个 SessionId 调 session/prompt 等)
```

第 3 步要构造 `AcpFsBackend` 时 `SessionId` 还没生成。**解法**：让 `defect-acp` 在 handler 里先生成 `SessionId`，然后传给 `AgentCore::create_session`。`uuid_like()` 函数从 `defect-agent::session::default` 移到 `defect-acp` 内部；`AgentError` 增加 `DuplicateSessionId` variant，由 `DefaultAgentCore` 的 `DashMap::insert` 检测重复时返回——理论上单调 + 时间戳 nonce 不会冲突，这个 variant 主要是安全网。

理由：`SessionId` 的"唯一性"裁定本来就是协议层的事（防跨连接冲突），让 `defect-agent` 接受外部 id 比 `defect-acp` 反过来 lazy-fill 干净。

### 3.3 取消语义

ACP 的 `session/cancel` 主要管 turn 级别；`fs/read_text_file` / `fs/write_text_file` **不是**规范明确支持取消的方法。v0 行为：

- 工具层在 `select!` 里 race `cancel.cancelled()` vs `backend.read_text(...).await` / `backend.write_text(...).await`。
- 取消触发时直接 abort 反向请求 future（drop）。客户端实现可能仍把请求跑完，没办法。

代价：客户端可能继续把"写文件"跑完后才看到 future 被 drop——导致 agent 已 cancel 但文件被写。**这不算腐蚀**——客户端写入的内容是最后一次正确的 `WriteTextFileRequest::content`，不会半新半旧。只是 cancel 语义不严格。演进项见 §6。

### 3.4 错误映射

```rust
fn map_wire_error(err: agent_client_protocol::Error) -> FsError {
    use agent_client_protocol::schema::ErrorCode;
    match ErrorCode::from(err.code) {
        ErrorCode::ResourceNotFound => {
            // 客户端能给的 path 不一定回得来；message 透传给 LLM 排障。
            FsError::Backend(BoxError::new(WireFailure(err)))
        }
        _ => FsError::Backend(BoxError::new(WireFailure(err))),
    }
}

#[derive(Debug, thiserror::Error)]
#[error("ACP fs request failed: code {code} message {message}")]
struct WireFailure {
    code: i32,
    message: String,
}
```

- ACP 没硬性规定 `fs/*` 的错误码语义——客户端实现可能各异。v0 务实做法：**全部走 `FsError::Backend`**，把 wire `code` / `message` 透传到 `BoxError` source。客户端社区收敛后再细化（按 code 分支映射 `NotFound` / `NotPermitted`）。
- `WireFailure` 是个内部 newtype，给 `BoxError` 包裹用——`agent_client_protocol::Error` 不实现 `std::error::Error`。

## 4. 工作区边界（agent 自我约束）

即使委托给客户端，agent 也要做边界校验。理由：

- LLM 让 agent 调 `read_file("/etc/passwd")`，agent 直接转发给客户端——客户端可能也会拒（vscode workspace boundary），但 agent 不该依赖客户端兜底。
- 客户端的边界 enforce 不一定与 [`ToolContext::cwd`] 一致（用户在 vscode 里打开了多文件夹 workspace）；agent 仍以 cwd 为权威。

实现：[`AcpFsBackend`] 与 [`LocalFsBackend`] 调用同一份 `defect_agent::fs::resolve_workspace_path`（见 [`tools-fs.md` §6.3](../internal/tools-fs.md#63-路径校验)）。差异：

- [`LocalFsBackend`] 校验完后用本地路径 IO；
- [`AcpFsBackend`] 校验完后把**绝对路径**塞进 ACP 请求（满足 §2.1 / §2.2 的"必须绝对"要求）。

[`ToolContext::cwd`]: ../internal/tool-trait.md#6-toolcontext

## 5. e2e 测试

放在 `crates/acp/tests/fs_delegation.rs`。结构与现有 acp e2e 一致（`Channel` transport + 假 Client）。

| # | 场景 | 验证 |
| --- | --- | --- |
| 1 | client_capabilities.fs = { read: true, write: true } → 跑 `read_file` 工具 | 收到一条 `fs/read_text_file` 反向请求；agent 正确把 client 返回的 content 装进 ToolEvent::Completed |
| 2 | 同上 → 跑 `write_file` 工具 | 收到一条 `fs/write_text_file` 反向请求；client 返回成功；ToolCallFinished |
| 3 | 同上 → 跑 `edit_file` 工具 | 看到先 read 再 write 两条反向请求，顺序正确 |
| 4 | client_capabilities.fs = { read: true, write: false } → 跑 `read_file` | **不**发反向请求（退回 LocalFsBackend）；从假 client 视角看不到 fs/* |
| 5 | client_capabilities 不带 fs → 同上 | 退回本地 |
| 6 | 委托模式下 client 返回错误 | ToolCallFinished status = Failed；content 含 wire `message`；raw_output 有诊断字段 |
| 7 | 委托模式下 turn 中途 cancel | 反向请求被 drop；不阻塞 cancel；TurnEnded = Cancelled |
| 8 | path 越界（agent 自己拦） | **不**发反向请求；ToolCallFinished status = Failed |

#1–#3 是基线；#4 是降级回归；#7 是取消时序的 hang trap；#8 是边界校验回归（防止 LLM 让客户端读 /etc）。

## 6. 后续演进（不是 v0 必做的项）

- **混合后端（read 委托、write 本地 等）**：v0 整组降级。客户端社区出现真实需求时再细化决策表。
- **fs/* 反向请求的取消**：v0 硬切——drop future，客户端可能跑完。等 ACP 规范给 fs 反向请求加 cancel notification（参考 `session/cancel` 形态），或我们引入"per-request cancel token" 机制。
- **byte-level / 二进制 fs**：ACP 目前只有 `read_text_file` / `write_text_file`；图片 / PDF / notebook 需要 ACP 增加 `read_resource` / `write_resource` 类方法。等协议演进。
- **delete / move / mkdir 反向方法**：ACP 0.13.2 没有这些方法。v0 让 LLM 走 [`bash`](../internal/tools-bash.md)（`rm` / `mv` / `mkdir`）。如果 ACP 后续增加（`fs/delete_file` / `fs/move_file` / `fs/create_directory`），那时 [`FsBackend`] trait 同步加方法，[`AcpFsBackend`] 走对应反向请求，[`LocalFsBackend`] 走 `tokio::fs::remove_file` 等，并引入对应工具。
- **路径校验下沉到客户端**：v0 agent 做完整边界校验。等用户配置出现"workspace 跨多 root"时让客户端做权威 enforce、agent 仅做最小校验（`SecurityPolicy::AllowAcpClient`）。
- **fs/* 与 sandbox policy 的二次确认**：v0 即便走委托，policy 仍按 [`tools-fs.md` §7](../internal/tools-fs.md#7-与-sandbox-policy-的协作) 触发 Ask（让用户在 agent 侧也确认一次）。客户端的 unsaved-buffer UI 体验出现"双重确认"问题再优化（让 policy 在 delegated 模式下默认 AllowAlways，把决定权完全交给客户端 UI）。
- **错误码细化**：v0 全走 `Backend`，wire `message` 透传。等 ACP 客户端实现收敛 deny / quota / read-only 等错误码后再扩。
- **`session/load` 与 fs**：恢复 session 时如果连接是新的（fs_mode 可能与之前不同），怎么处理？v0 不做 session/load（见 [`acp-bridge.md` §7](./acp-bridge.md#7-v0-不做的事明确划线)），延后到持久化设计。

## 7. 与现有文档的协同更新

落地时同步更新：

- **`docs/inbound/acp-bridge.md`**：
  - §2 表格里 `fs/read_text_file` / `fs/write_text_file` 已指向本文。
  - §7 `fs/*` 那项已指向本文（不再说"声明 fs_capabilities = false"）。
  - §1.2 / §5 提及 `AgentCore::create_session` 签名变化（接受外部 `SessionId` 与 `Arc<dyn FsBackend>`）——本文落地时一并改。
- **`docs/internal/tools-fs.md`** §2.2 / §10 已经预留 [`AcpFsBackend`] 的前向引用，本文是其落地。
- **`docs/internal/tool-trait.md`** §6 提到 `ToolContext` 未来字段含"ACP 反向通道"——`fs` 字段是这个口子的第一份具体兑现。
- **`TODO.MD`**：fs 工具与本节同时落地后，两行一起翻到「已完成」。

## 8. 落地节奏

按 [`tools-fs.md` §10](../internal/tools-fs.md#10-落地节奏-与-acp-fsmd-同步) 的整体顺序，本节（步骤 3 / 5）涉及：

1. **`crates/agent/src/fs/`**（与 [`tools-fs.md`] 共用）：把 `resolve_workspace_path` helper 暴露成 `pub`，[`AcpFsBackend`] 共用。
2. **`crates/acp/src/fs.rs`**（新文件）：[`AcpFsBackend`] 实现（本文 §3）+ `WireFailure` newtype。
3. **`crates/acp/src/serve.rs`**：
   - `initialize` handler 读 `req.client_capabilities.fs`，存到连接级状态（`Arc<RwLock<FsMode>>` 或挂在 builder 闭包捕获的状态里）。
   - `session/new` handler 按 `FsMode` 构造 `Arc<dyn FsBackend>`，传给 `AgentCore::create_session`。
   - 把 `uuid_like()` 从 `defect-agent` 移过来。
4. **`crates/acp/tests/fs_delegation.rs`**：跑 §5 测试矩阵。
5. **更新 `TODO.MD`** 与 §7 列出的关联文档。
