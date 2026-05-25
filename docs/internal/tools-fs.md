# `fs` 内置工具设计（read / write / edit）

`fs` 是文件系统工具家族。与 [`bash`](./tools-bash.md) 同列，但语义更窄：每条工具只对**一个文件**做一件事，有明确的读写区分，路径全部受 [`ToolContext::cwd`] 约束。

本文沉淀三个工具（`read_file` / `write_file` / `edit_file`）的形状、与 ACP 的对位、底层 [`FsBackend`] 抽象、与 [`docs/inbound/acp-fs.md`](../inbound/acp-fs.md) 的协同。v0 已落地基础形态；v1 在三处补强：`edit_file` 并发写冲突检测、`read_file` 大文件窗口流式读、`write_file` describe 阶段精确 diff（详见 §11.1）。

设计原则按依赖顺序：

1. **以 ACP 为导向**——产出的字段直接对位 [`ToolCallUpdateFields`] / [`ToolCallContent`]，不另造内部结构。
2. **后端可替换、且 v0 同时落地两个**——读写不直接打 `std::fs`，而是经过 [`FsBackend`] trait。v0 同时实现 [`LocalFsBackend`]（直接打盘）与 [`AcpFsBackend`]（走 ACP `fs/read_text_file` / `fs/write_text_file` 反向请求委托给客户端）。决策权在 [`defect-acp`] 装配时——按客户端协商出的 [`FileSystemCapabilities`] 选定，工具实现完全不感知。
3. **工作区边界由工具自己守**——[`LocalFsBackend`] 内置 canonicalize + `starts_with(cwd)` 校验；委托模式下也要做（防止 LLM 让客户端读 `/etc/passwd`）。
4. **不留坑**——不为 v0 简化而做"会写坏数据 / 留半截文件"的实现。具体两条体现：行末符在 [`LocalFsBackend`] 写回时按文件原行末符规范化；覆盖写一律 `tmp + rename` 原子替换。详见 §6。
5. **删除 / 移动 / 创建目录不进 fs 工具家族**——ACP 0.13 schema 仅有 `fs/read_text_file` / `fs/write_text_file` 两个反向方法，**没有** `fs/delete_*` / `fs/move_*` / `fs/create_dir`。这意味着我们在委托模式下没法委托这些操作。v0 的明确选择：**不引入 `delete_file` / `move_file` / `mkdir` 工具**——LLM 想做就走 [`bash`](./tools-bash.md) (`rm` / `mv` / `mkdir -p`)。理由见 §1.2。

[`ToolCallUpdateFields`]: https://docs.rs/agent-client-protocol-schema/0.13/agent_client_protocol_schema/v1/struct.ToolCallUpdateFields.html
[`ToolCallContent`]: https://docs.rs/agent-client-protocol-schema/0.13/agent_client_protocol_schema/v1/enum.ToolCallContent.html
[`ToolContext::cwd`]: ./tool-trait.md#6-toolcontext
[`Tool`]: ./tool-trait.md
[`FileSystemCapabilities`]: https://docs.rs/agent-client-protocol-schema/0.13/agent_client_protocol_schema/v1/struct.FileSystemCapabilities.html
[`defect-acp`]: ../inbound/acp-bridge.md
[`FsBackend`]: #2-fsbackend-抽象
[`LocalFsBackend`]: #21-localfsbackend
[`AcpFsBackend`]: ../inbound/acp-fs.md#3-acpfsbackend

## 1. 工具家族总览

### 1.1 三个工具

| 工具 | safety_hint | ACP `kind` | 语义 |
| --- | --- | --- | --- |
| `read_file` | `ReadOnly` | `Read` | 读 UTF-8 文本（可选 offset/limit 行窗口） |
| `write_file` | `Mutating` | `Edit` | 全量覆盖写（新建或覆盖） |
| `edit_file` | `Mutating` | `Edit` | 精确字符串替换（不存在 / 多次匹配 → 报错） |

三个工具共用 [`FsBackend`] 抽象，差异仅在参数形状与 patch 算法。

### 1.2 不进 v0 的工具与原因

| 操作 | ACP 反向方法 | v0 选择 | 理由 |
| --- | --- | --- | --- |
| 删除文件 | （无） | 不做 `delete_file` 工具 | ACP schema 没有 `fs/delete_*`；委托模式下无路可走。LLM 用 `bash("rm path")`，policy 兜底（`Destructive`）。 |
| 移动 / 重命名 | （无） | 不做 `move_file` 工具 | 同上。`bash("mv a b")`。 |
| 创建目录 | （无） | 不做 `mkdir` 工具 | 同上。`bash("mkdir -p path")`。 |
| 读 / 写二进制 | （无） | `read_file` / `write_file` 拒绝二进制（fail loud） | ACP `fs/*` 只有 text 形态。多模态等 ACP 协议演进。 |

这是**故意不引入**——而非"v1 再说"。两条原则：

- 不在工具表面声明"v0 不工作"的能力（避免 LLM 看 schema 觉得有，调用时炸）。
- 不为这些操作单独造一份 `LocalFsBackend`-only 的 fast path——委托模式没有对位反向方法时整组拒绝引入工具，比"local 模式能跑、acp 模式炸"更诚实。

如果未来 ACP 增加这些方法（或我们认为 LLM 走 `bash rm` 体验太差），再回来加，**到时候** [`FsBackend`] trait 同步加 `delete_text_file` / `move_text_file` / `create_directory`，[`AcpFsBackend`] 走对应反向请求，[`LocalFsBackend`] 走 `tokio::fs::remove_file` 等。这是干净的延展；与现有设计无冲突。

## 2. `FsBackend` 抽象

```rust
use std::path::{Path, PathBuf};

use futures::future::BoxFuture;

pub trait FsBackend: Send + Sync {
    /// 读取整个文件的 UTF-8 文本。
    ///
    /// `line` / `limit` 与 ACP `ReadTextFileRequest` 同语义：
    /// - `line = Some(n)` 表示从第 n 行（1-based）开始读
    /// - `limit = Some(k)` 表示最多读 k 行
    /// - 两者皆 None 表示读全文
    ///
    /// # Errors
    /// - [`FsError::NotFound`] —— 文件不存在
    /// - [`FsError::NotPermitted`] —— 路径越界 / 权限不足 / 二进制
    /// - [`FsError::TooLarge`] —— 文件超 [`MAX_READ_BYTES`]
    /// - [`FsError::Backend`] —— 底层 IO / RPC 失败
    fn read_text(
        &self,
        path: PathBuf,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> BoxFuture<'_, Result<String, FsError>>;

    /// 全量覆盖写一个 UTF-8 文本文件。
    ///
    /// 父目录必须已存在；后端不静默 mkdir-p（与 §1.2 的"删/移/建目录走
    /// `bash`"一致——目录由 LLM 显式管，工具家族不做）。
    ///
    /// 行末符 / 原子性的责任划分见 §6。
    ///
    /// # Errors
    /// 同 [`FsBackend::read_text`]。
    fn write_text(
        &self,
        path: PathBuf,
        content: String,
    ) -> BoxFuture<'_, Result<(), FsError>>;

    /// 取一份"内容指纹"（v1 引入）。`edit_file` 在 read → modify → write
    /// 的窗口中用它检测并发外部修改：read 后取一份 baseline，write 前再
    /// 取一份；不一致即报 [`FsError::Conflict`]。
    ///
    /// 默认实现走 `read_text` 全文读 + 内容哈希——任何 [`FsBackend`] 实现
    /// 都能开箱即用（[`AcpFsBackend`] 走默认即可）。本地后端可重写为
    /// mtime + size 的更便宜判定。
    fn fingerprint(
        &self,
        path: PathBuf,
    ) -> BoxFuture<'_, Result<Fingerprint, FsError>> {
        Box::pin(async move {
            let text = self.read_text(path, None, None).await?;
            Ok(Fingerprint::of(&text))
        })
    }
}

/// 文件内容指纹。`(bytes, hash)` 的双字段比较：长度 + 哈希双重校验，把
/// 单 `u64` 哈希的碰撞概率压到可忽略。仅用于进程内一次性对比，不持久化、
/// 不跨进程，所以 std `DefaultHasher` 的"未指定但稳定"语义足够。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fingerprint {
    pub bytes: u64,
    pub hash: u64,
}

#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum FsError {
    #[error("file not found: {0}")]
    NotFound(PathBuf),
    #[error("operation not permitted: {0}")]
    NotPermitted(String),
    #[error("file too large: {bytes} bytes > {limit}")]
    TooLarge { bytes: u64, limit: u64 },
    /// `edit_file` 的 baseline / pre-write fingerprint 对不上，说明读到 LLM
    /// 决定如何 edit、再到工具回写之间，有外部进程 / 另一个 agent 改了同一
    /// 文件。映射为 [`ToolError::Execution`]，让 LLM 重读后重试。
    #[error("file modified concurrently: {0}")]
    Conflict(PathBuf),
    #[error("backend failure: {0}")]
    Backend(#[source] BoxError),
}
```

设计点：

- **`BoxFuture<'_, ...>` 而非 `async fn`**——本仓库不引 `async-trait` 宏；与 [`LlmProvider::complete`](./llm-trait.md#2-trait-主签名) / [`AgentCore::create_session`](../inbound/acp-bridge.md#5-session-的接口需求) 同形态。实现者写 `Box::pin(async move { ... })`。
- **入参用 owned `PathBuf` / `String` 而非 `&Path` / `&str`**——参考 `LlmProvider::complete(req: CompletionRequest, ...)` 的取舍，把 future 的生命周期收敛到 `&'_ self`，避免显式生命周期参数。一次 fs 调用多 clone 一个 path 不计代价。
- **`read_text` / `write_text` 是后端唯一的两个动词**——`edit` 由工具层组合：先 `read_text` 拿原文、做字符串替换、再 `write_text` 写回。后端无需理解 patch 语义；委托模式下两次反向请求显式可见。
- **不暴露 `metadata` / `exists`**——v0 用不上。文件不存在通过 `read_text` / `write_text` 的错误返回。
- **`FsError::NotPermitted` 携带字符串而非 enum**——v0 还不知道未来要分多少种"拒绝"理由（policy / sandbox / 客户端 deny），先用字符串占位，演进时再升枚举。

[`MAX_READ_BYTES`]: #41-名片

### 2.1 `LocalFsBackend`

```rust
pub struct LocalFsBackend {
    workspace_root: PathBuf,
}
```

- 由 [`defect-tools`] 装配时构造。`workspace_root` 来自 session 的 cwd——但 [`FsBackend`] 是 session 级注入（§3），所以 root 在 session 创建时就定下，不跟着每次调用变。
- `read_text` 用 `tokio::fs::read_to_string`，读完后做二进制检测（前 8 KiB 内有 `\0` 或非可打印字节比例 > 30%）→ 返回 `NotPermitted("binary file")`。
- `write_text` 走原子写（§6.2）+ 行末符规范化（§6.1）。

### 2.2 `AcpFsBackend`（前向引用）

实现细节在 [`docs/inbound/acp-fs.md`](../inbound/acp-fs.md) §3。本文不重复。本节只声明工具层不感知后端选择——拿到的就是 `Arc<dyn FsBackend>`。两个后端都是 v0 必出的——见 [acp-fs.md §1](../inbound/acp-fs.md#1-能力协商) 的决策表。

### 2.3 注入路径

[`ToolContext`] 加一个字段：

```rust
#[non_exhaustive]
pub struct ToolContext<'a> {
    pub cwd: &'a Path,
    pub cancel: CancellationToken,
    /// session 级 fs 后端。fs 工具家族通过它读写文件。
    pub fs: &'a dyn FsBackend,
}
```

注意是 `&'a dyn FsBackend` 而非 `Option<...>`——v0 装配保证总有后端（默认 [`LocalFsBackend`]，按 ACP 协商升级到 [`AcpFsBackend`]）。fs 工具假定它存在；不需要"开发期防御"分支。

[`ToolContext::new`] 签名相应改为接受三个参数（cwd / cancel / fs）；`#[non_exhaustive]` 已经让外部 crate 只能走构造函数，所以这是 source-compatible 演进——所有调用点（目前只有 `crates/agent/src/session/turn.rs::drive_tool_stream`）一并改。测试构造可以用一个 `NoopFsBackend` 占位（或直接 `LocalFsBackend::new(tempdir)`）。

## 3. `read_file` 工具

### 3.1 名片

```rust
ToolSchema {
    name: "read_file".to_string(),
    description: "Read a UTF-8 text file from the workspace. \
                  Optionally read a window starting at `offset` (1-based line) for `limit` lines. \
                  Returns the file content with 1-based line numbers prepended. \
                  Refuses binary files and files larger than 10 MiB.".to_string(),
    input_schema: json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path or path relative to the session cwd. \
                                Must resolve inside the workspace root."
            },
            "offset": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional 1-based start line (inclusive). Defaults to 1."
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5000,
                "description": "Optional max number of lines to read. Defaults to 2000."
            }
        },
        "required": ["path"]
    }),
}
```

字段取舍：

- **参数名 `offset` / `limit`** 与 opencode 一致，与 ACP `ReadTextFileRequest` 的 `line` / `limit` 同语义；后端调用时映射 `offset` → `line`。选 `offset` 而非 `line`：让 LLM 的心智更接近"分页 / 窗口"。
- **没有 `read_image` / `read_pdf`**——v0 拒绝二进制；多模态文件等 ACP 协议演进。
- **`limit` 默认 2000，硬上限 5000**——参考 opencode 的 `DEFAULT_READ_LIMIT = 2000`，让 LLM 单次读不至于把 context 灌满。
- **`MAX_READ_BYTES = 10 MiB`**——给真实代码仓库留余量（vendored / generated file 经常 1–2 MB），同时拦掉 `cat /var/log/...` 之类爆炸读。超过 → `FsError::TooLarge` （fail loud）。
- **窗口读绕过整文件大小上限**（v1）——`offset` / `limit` 任一非空时，[`LocalFsBackend`] 走流式 `BufReader::read_until(b'\n', ...)`，逐行跳过 / 累积，不预先要求整文件 ≤ `MAX_READ_BYTES`。窗口本身仍受 `MAX_READ_BYTES` 限制（窗口累积到 10 MiB 报 `TooLarge`），防止 LLM 用超大 `limit` 绕过。整文件读（无 offset/limit）保留原 fail-loud 语义。

### 3.2 `safety_hint`

```rust
fn safety_hint(&self, _args: &Value) -> SafetyClass {
    SafetyClass::ReadOnly
}
```

- 与 [`bash`](./tools-bash.md#2-安全等级safety_hint) 不同，`read_file` 一律 `ReadOnly`。配合 [`ReadOnlyPolicy`] 让"只读模式"用户能跑这个工具，但 `write_file` / `edit_file` 一律 deny。
- **不**因 path 在 `.git/` / `.env` 等敏感位置就升级——sandbox policy 才是最终守门员，这里保持简单一致。

[`ReadOnlyPolicy`]: ./sandbox-policy.md#5-v0-内置-policy

### 3.3 `describe`

```rust
ToolCallUpdateFields {
    title: Some(format!("Read {}", display_relative(path))),
    kind:  Some(ToolKind::Read),
    locations: Some(vec![ToolCallLocation { path: abs_path.clone(), line: offset }]),
    content:   None,
    raw_input: None,
    raw_output: None,
    status:    None,
}
```

- `title` 显示 workspace 相对路径（`display_relative`）让 UI 紧凑；`abs_path` 留给 `locations` 用。
- `line` 用 `offset`——客户端 follow-along 跳转到具体行。

### 3.4 `execute`

```text
       ToolEvent::Progress(fields = describe.fields)        // 立即推一条
                          │
                          ▼
       backend.read_text(canon_path, offset, limit)
                          │
            ┌─────────────┼──────────────┐
            ▼             ▼              ▼
         Ok(text)      cancel        FsError::*
            │             │              │
            ▼             ▼              ▼
   ToolEvent::Completed   Failed     ToolEvent::Failed(
   (content: Text(formatted),        ToolError::{
    raw_output: { lines, bytes,        InvalidArgs | Execution
                  truncated }))      })
```

- **不流式**——单帧 Progress（描述）+ 单帧 Completed（结果）。文件读完是一次性结果，没必要按行流。
- **content 格式**：`<file>\n   1| line1\n   2| line2\n...</file>` 的标准 `cat -n` 形态（opencode 同款）。便于 LLM 引用具体行号。超 `limit` 行后追加 `\n[file truncated; showing N of M lines starting at offset O]`。
- **二进制 / 太大** → `FsError::NotPermitted` / `TooLarge` → `ToolError::Execution`（不是 `InvalidArgs`，因为参数本身合法、是文件本身有问题）。
- **取消**：`backend.read_text` 是 async 的，工具层在 `select!` 里 race `cancel.cancelled()`。

### 3.5 raw_output

```rust
struct ReadFileOutput {
    bytes: u64,
    lines_returned: u32,
    lines_total: u32,
    truncated: bool,
}
```

给客户端 UI 展示"读了 N/M 行"。

## 4. `write_file` 工具

### 4.1 名片

```rust
ToolSchema {
    name: "write_file".to_string(),
    description: "Write a UTF-8 text file. \
                  Overwrites the file if it exists; creates it if it does not. \
                  Requires the parent directory to already exist. \
                  Path must be inside the workspace root.".to_string(),
    input_schema: json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path or path relative to the session cwd."
            },
            "content": {
                "type": "string",
                "description": "Full UTF-8 text content. Replaces the file entirely."
            }
        },
        "required": ["path", "content"]
    }),
}
```

字段取舍：

- **没有 `mode`（append / overwrite / create-only）**——v0 一律全量覆盖。append 让 LLM 状态难追；create-only 让 LLM 重试时碰墙。LLM 想 append 自己 `read_file` 拿原文 + 新内容，再 `write_file`。
- **没有 `mkdir`**——父目录不存在就报错，让 LLM 显式 `bash("mkdir -p ...")`。理由见 §1.2。
- **`MAX_WRITE_BYTES = 10 MiB`**——与 read 对称。

### 4.2 `safety_hint`

```rust
fn safety_hint(&self, _args: &Value) -> SafetyClass {
    SafetyClass::Mutating
}
```

`Mutating` 而非 `Destructive`：写文件可逆（git 能恢复 / 客户端能 undo）。`Destructive` 留给 `bash` 等真正不可逆的操作。配合 [`AskWritesPolicy`] 触发 Ask；用户 `AllowAlways` 后该 session 内不再问。

### 4.3 `describe`

```rust
ToolCallUpdateFields {
    title: Some(format!("Write {}", display_relative(path))),
    kind:  Some(ToolKind::Edit),  // 即便是新建，ACP 没有专门的 Create kind
    locations: Some(vec![ToolCallLocation { path: abs_path, line: None }]),
    content:   Some(vec![ToolCallContent::Diff(diff_block(None, &new_content))]),
    raw_input: None,
    raw_output: None,
    status:    None,
}
```

- **describe 时就附带 diff 预览**——给 [`RequestPermissionRequest`] 的 wire payload 一份"用户决定授权前能看到的 diff"。
- **describe 是 async 且持有 `&dyn FsBackend`**（v1）——`Tool::describe` 签名为 `fn describe<'a>(&'a self, args: &'a Value, ctx: ToolContext<'a>) -> BoxFuture<'a, ToolCallDescription>`，可以读旧内容。`write_file` 在 describe 阶段调 `ctx.fs.read_text(path, None, None).await.ok()` 拿旧内容，能给 `(Some(old), new)` 的精确 diff。文件不存在 / 越权时 `read_text` 返回 `Err`，转 `None`，diff 退化为 `(None, new)`——对新建场景这也是正确的最小输入。授权 wire payload 即可拿到精确 diff，UX 一并提升。

### 4.4 `execute`

```text
   Progress(fields = describe.fields)               // 立即一帧（已含精确 diff）
              │
              ▼
   backend.write_text(path, content)
              │
        ┌─────┴──────┐
        ▼            ▼
       Ok          FsError::*
        │            │
        ▼            ▼
   Completed(    Failed(ToolError::*)
    raw_output)
```

- **describe 阶段已经画好精确 diff**（§4.3，v1）——execute 不再做"第二次 Progress 带新 diff"的补偿；Progress 复用 describe 的 fields 即可。
- **行末符 / 原子写**：[`LocalFsBackend`] 内部按 §6 处理；[`AcpFsBackend`] 把决定权完全交给客户端。

### 4.5 raw_output

```rust
struct WriteFileOutput {
    bytes_written: u64,
    created: bool,         // true = 新建；false = 覆盖
    parent_existed: bool,
}
```

## 5. `edit_file` 工具

### 5.1 名片

```rust
ToolSchema {
    name: "edit_file".to_string(),
    description: "Replace a string in a UTF-8 text file. \
                  Performs an exact string replacement; \
                  fails if `old_string` is not found, or if it appears multiple times \
                  unless `replace_all` is true. \
                  Path must be inside the workspace root.".to_string(),
    input_schema: json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path or path relative to the session cwd."
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to replace. Must match a unique substring \
                                unless `replace_all` is true. Empty string is rejected."
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text. Must differ from old_string."
            },
            "replace_all": {
                "type": "boolean",
                "description": "When true, replace every occurrence; when false (default), \
                                require old_string to appear exactly once.",
                "default": false
            }
        },
        "required": ["path", "old_string", "new_string"]
    }),
}
```

字段取舍：

- **不实现 opencode 的 9-replacer 链**（Simple / LineTrimmed / BlockAnchor / Whitespace / Indentation / Escape / TrimmedBoundary / ContextAware / MultiOccurrence）。v0 选**严格语义**——精确字符串匹配，找不到 / 多次匹配（无 `replace_all`）就报错。理由：
  1. fuzzy 匹配带来的 surprise 不可控（"这条 LLM 想改的 import 误命中了同名变量声明"），出问题难复现——属于 latent bug。
  2. v0 让 LLM 学到"先 `read_file` 看原文 → 选独特上下文 → 调 `edit_file`"的工作流，比无脑 fuzzy 更稳。
  3. 9-replacer 整套 ~600 行 TS，照搬到 Rust 是 v0 不该背的成本。
  fuzzy 是"feature gap"而非"latent bug"——v0 用户写 `old_string` 错一个空格会立即拿到 `NotFound` 错误（fail loud），不会静默改错地方。演进项见 §10。
- **`old_string` / `new_string` 必须不同**——空 diff 是 LLM 笔误。
- **`old_string` 不许为空**——空字符串语义模糊（替换全文？什么都不替换？）。
- **`replace_all=false` + 多次匹配 → `InvalidArgs`**——让 LLM 看到 "found 3 matches; add unique context or set replace_all" 改 args 重试。

### 5.2 `safety_hint`

```rust
fn safety_hint(&self, _args: &Value) -> SafetyClass {
    SafetyClass::Mutating
}
```

与 `write_file` 同。

### 5.3 `describe`

```rust
ToolCallUpdateFields {
    title: Some(format!("Edit {}", display_relative(path))),
    kind:  Some(ToolKind::Edit),
    locations: Some(vec![ToolCallLocation { path: abs_path, line: None }]),
    // describe 阶段不读盘，所以 diff content 留给 execute 期填
    content:   None,
    raw_input: None,
    raw_output: None,
    status:    None,
}
```

### 5.4 `execute`

```text
   Progress(fields = describe.fields)
              │
              ▼
   backend.read_text(path)              // 必读
              │
        ┌─────┴──────┐
        ▼            ▼
       Ok        FsError::* → Failed
        │
        ▼
   backend.fingerprint(path)            // baseline，best-effort
   → baseline_fp: Option<Fingerprint>
              │
              ▼
   apply_edit(old, old_string, new_string, replace_all)
              │
        ┌─────┼──────────┐
        ▼     ▼          ▼
     Ok(new) NotFound  Ambiguous(N)
        │     │          │
        │     ▼          ▼
        │   Failed(InvalidArgs("old_string not found"))
        │   Failed(InvalidArgs("old_string matched N times; add context or set replace_all"))
        ▼
   Progress(content = Diff(old, new))
              │
              ▼
   if let Some(b) = baseline_fp:
       backend.fingerprint(path)        // 二次取，与 baseline 对比
       Ok(cur) if cur != b  → Failed(Execution(Conflict))
       _                    → 继续
              │
              ▼
   backend.write_text(path, new)
              │
        ┌─────┴──────┐
        ▼            ▼
       Ok        FsError::* → Failed
        │
        ▼
   Completed(raw_output)
```

- **`apply_edit` 是纯字符串操作**：
  - `replace_all = false`：扫描所有 match 位置，`> 1` 报 `Ambiguous(count)`，`== 0` 报 `NotFound`，`== 1` 替换。
  - `replace_all = true`：`String::replace`，统计替换次数；`== 0` 报 `NotFound`。
- **行末符**：边界处理放在 [`LocalFsBackend`] 而非 `apply_edit` 里——`apply_edit` 看到的就是 `read_text` 返回的字符串原貌。详见 §6.1。
- **content 终态用 `Diff(old, new)`**——同 `write_file`。
- **并发写冲突检测**（v1）——baseline / 二次 fingerprint 之间被外部进程改了文件 → `Failed(Execution(FsError::Conflict))`；LLM 重试时会先 `read_file` 拿到新内容、重做 edit。fingerprint 取/对比都是 best-effort：取 baseline 失败（NotPermitted / Backend）就放弃检测、不阻塞主路径——前置 read 已成功，"取指纹失败但写没失败"对用户来说仍是"edit 成功"，比 fail loud 更不打扰。两次 fingerprint 都走同一个 backend，所以 [`LocalFsBackend`] 的 mtime+size 与默认实现的内容哈希互不污染。

### 5.5 raw_output

```rust
struct EditFileOutput {
    matches_replaced: u32,
    bytes_before: u64,
    bytes_after: u64,
}
```

## 6. [`LocalFsBackend`] 的正确性细节

本节是 §1 设计原则 4 "不留坑" 的兑现——v0 必须做对的两件事。委托模式下责任在客户端，[`AcpFsBackend`] 不重复做。

### 6.1 行末符规范化（v0 必做）

**问题**：`tokio::fs::read_to_string` 不转换行末符。CRLF 文件读出来是 `"a\r\nb\r\n"`。LLM 给的 `new_string` 通常是 LF（`"a\nb"`）。如果直接 `String::replace` 把 `old_string` 替换成 `new_string` 后写回，文件会变成 CRLF / LF 混用——下游工具（git diff / 编辑器）显示异常，且这是**用户察觉不到**的腐蚀（文件还能打开、内容还对）。

**v0 行为**（[`LocalFsBackend::write_text`] 内部）：

```rust
fn detect_line_ending(text: &str) -> LineEnding {
    let crlf = text.matches("\r\n").count();
    let total_lf = text.matches('\n').count();
    let lone_lf = total_lf.saturating_sub(crlf);
    if crlf > lone_lf { LineEnding::Crlf } else { LineEnding::Lf }
}

fn normalize(content: &str, target: LineEnding) -> Cow<'_, str> {
    match target {
        LineEnding::Lf => {
            if content.contains("\r\n") {
                Cow::Owned(content.replace("\r\n", "\n"))
            } else {
                Cow::Borrowed(content)
            }
        }
        LineEnding::Crlf => {
            // 先归一到 LF，再统一替换为 CRLF——避免 "\r\n\n" 类输入二次拼接成 "\r\r\n"。
            let lf = content.replace("\r\n", "\n");
            Cow::Owned(lf.replace('\n', "\r\n"))
        }
    }
}
```

写入流程：

1. 如果文件已存在：`read_to_string` 旧内容 → `detect_line_ending` → 把要写入的 `content` 用 `normalize` 规范化到原行末符；
2. 如果文件不存在：直接写入 LLM 给的 `content`，不做改动（不在 v0 推断"该用什么行末符"的策略）。

**为什么不在 `read_text` 时就归一到 LF**：会让 LLM 看到的内容与磁盘真实字节不一致；后续 `edit_file` 的 `old_string` 如果含字面 `\r\n`（例如 LLM 是从 git diff 复制粘贴的）反而匹配不上。保留原貌、只在写回时规范化，是最小惊讶。

**回归测试**：CRLF 文件 + LF `new_string` 的 edit 必须写回纯 CRLF；mixed 文件 + edit 必须写回主流行末符（这种场景算"修复腐蚀"，从行为上可接受）。

### 6.2 原子写（v0 必做）

**问题**：`tokio::fs::write(path, content)` 是 `open(O_WRONLY|O_CREAT|O_TRUNC) → write → close`。中途崩溃 / 进程被杀 / 磁盘满会留半截文件——比"没写成"更糟，因为下次 `read_file` 看到的不是合法 JSON / 不是合法源码。这是**用户察觉不到**的腐蚀。

**v0 行为**（[`LocalFsBackend::write_text`] 内部）：

```rust
async fn atomic_write(workspace_root: &Path, path: &Path, content: &str) -> io::Result<()> {
    let parent = path.parent().ok_or_else(|| io::Error::other("path has no parent"))?;
    let file_name = path.file_name()
        .ok_or_else(|| io::Error::other("path has no file component"))?;
    // 临时文件名带 pid + 单调计数，落在同一父目录（确保 rename 跨设备不出问题）。
    let nonce = TMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let tmp_path = parent.join(format!(
        ".{}.defect-{pid}-{nonce}.tmp",
        file_name.to_string_lossy()
    ));
    // RAII：err 路径上自动 remove tmp，避免残留。
    let cleanup = TmpCleanup { path: tmp_path.clone() };
    tokio::fs::write(&tmp_path, content).await?;
    tokio::fs::rename(&tmp_path, path).await?;
    cleanup.disarm();
    Ok(())
}
```

要点：

- **临时文件落在同一父目录**——跨设备 `rename` 可能 `EXDEV` 报错；同 parent 保证 rename 是 inode 级原子操作。
- **隐藏文件前缀 `.<name>.defect-<pid>-<nonce>.tmp`**——避免与同目录用户文件冲突；`pid` 让多 agent 实例并行也不冲突；`nonce` 单调原子计数器避免同进程多次写同一路径打架。
- **err 路径清理 tmp**——用一个 `TmpCleanup` guard，drop 时若未 disarm 则尝试 `tokio::fs::remove_file`。失败 silently（最坏留个 .tmp 文件，比留半截目标文件好）。
- **不 `fsync`**——v0 不保证断电幂等（那是 ext4 / journal 的事）；`rename` 已经给"半截"问题足够防护。如果将来引入崩溃恢复测试再加。

**回归测试**：模拟 `write_text` 中途 panic（在 `tokio::fs::write` 之后、`rename` 之前 inject 一个错），验证目标文件未被创建 / 未被损坏；tmp 文件可能残留也可能被清理（不强求）。

### 6.3 路径校验

```rust
fn resolve_workspace_path(workspace_root: &Path, requested: &Path) -> Result<PathBuf, FsError> {
    let target = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        workspace_root.join(requested)
    };
    // canonicalize 父目录而非目标本身——目标在 write 场景可能尚未存在。
    let parent = target.parent().ok_or_else(|| {
        FsError::NotPermitted(format!("path has no parent: {}", target.display()))
    })?;
    let parent_canon = std::fs::canonicalize(parent)
        .map_err(|e| FsError::Backend(BoxError::new(e)))?;
    let root_canon = std::fs::canonicalize(workspace_root)
        .unwrap_or_else(|_| workspace_root.to_path_buf());
    if !parent_canon.starts_with(&root_canon) {
        return Err(FsError::NotPermitted(format!(
            "path {} escapes workspace root {}",
            target.display(), root_canon.display()
        )));
    }
    let file_name = target.file_name().ok_or_else(|| {
        FsError::NotPermitted(format!("path has no file component: {}", target.display()))
    })?;
    Ok(parent_canon.join(file_name))
}
```

要点：

- **canonicalize 父目录**——目标文件 `write` 场景可能尚未存在，对其 canonicalize 会失败。父目录必然存在（§4.1 要求），canon 后再拼上文件名。
- **symlink 越狱**：`workspace/dir/link → /etc` 时 `parent_canon = /etc`，`starts_with(workspace_root)` 失败。
- **跨平台**：Windows 上 `\\?\` 前缀的 `starts_with` 仍正确。

委托模式下，[`AcpFsBackend`] 也调用同一份 `resolve_workspace_path`——见 [`acp-fs.md` §4](../inbound/acp-fs.md#4-工作区边界agent-自我约束)。

## 7. 与 sandbox policy 的协作

| 工具 | safety_hint | ReadOnly policy | AskWrites policy | Open policy |
| --- | --- | --- | --- | --- |
| `read_file` | `ReadOnly` | Allow | Allow | Allow |
| `write_file` | `Mutating` | Deny（"read-only mode"） | Ask | Allow |
| `edit_file` | `Mutating` | Deny | Ask | Allow |

[`AskWritesPolicy`] 内部维护 `(tool_name, file_path)` 粒度的"AllowAlways" 记录：用户对 `edit_file src/foo.rs` 选 AllowAlways 后，再次 edit 同一文件不再 Ask；但 edit 别的文件仍要 Ask。具体记录粒度由 [`docs/internal/sandbox-policy.md`](./sandbox-policy.md) 决定，本文不重复。

[`PolicyDecision`]: ./sandbox-policy.md#2-决策模型
[`AskWritesPolicy`]: ./sandbox-policy.md#5-v0-内置-policy

## 8. 错误分类映射

| 来源 | ToolError variant | 主循环处理 |
| --- | --- | --- |
| args JSON 反序列化失败 | `InvalidArgs(serde_err)` | 喂回 LLM 改 args |
| `path` 越界 / 无 parent / `old_string` 为空 / `old_string == new_string` | `InvalidArgs(io_err / msg)` | 喂回 LLM |
| `edit_file`: `old_string` not found / 多次匹配（无 `replace_all`） | `InvalidArgs(msg)` | 喂回 LLM 让它改 args / 加上下文 |
| 文件不存在（`read` / `edit`） | `Execution(FsError::NotFound)` | 上 wire 失败、不算 invalid args（路径合法、是文件状态问题） |
| 文件超大 / 二进制 | `Execution(FsError::TooLarge / NotPermitted)` | 同上 |
| `edit_file`：baseline / pre-write fingerprint 对不上 | `Execution(FsError::Conflict)` | 上 wire 失败；LLM 通常会重读后重试 |
| 客户端 deny / `AcpFsBackend` 反向请求失败 | `Execution(FsError::Backend)` | 同上 |
| `ctx.cancel` 触发 | `Canceled` | 不计 retry |

为什么 "edit 找不到 old_string" 走 `InvalidArgs` 而非 `Execution`：让主循环把错误信息塞回 LLM，LLM 大概率能据此改 args 重试（"换段更独特的 old_string"）。`Execution` 暗示工具本身跑出了问题、不该让 LLM 重试。

## 9. 测试矩阵

每条都写成 `#[tokio::test]`，放在 `crates/tools/src/fs/tests.rs`。

| # | 工具 | 后端 | 场景 | 验证 |
| --- | --- | --- | --- | --- |
| 1 | read | local | 读现有 UTF-8 文件，无 offset/limit | content 含 `1| ` 行号；raw_output.lines_total = 实际行数 |
| 2 | read | local | offset=3, limit=2 | 只返回第 3-4 行；raw_output.lines_returned = 2 |
| 3 | read | local | 文件 12 MB | event = `Failed(Execution(TooLarge))` |
| 4 | read | local | 二进制文件（含 `\0`） | event = `Failed(Execution(NotPermitted("binary")))` |
| 5 | read | local | path 越界（`../../etc/passwd`） | event = `Failed(Execution(NotPermitted))` |
| 6 | read | local | path 是 symlink 指向 workspace 外 | 同 #5 |
| 7 | read | local | 取消（read 长跑时 cancel） | event = `Failed(Canceled)` |
| 8 | write | local | 新建文件 | raw_output.created = true；文件内容匹配；终态 content = Diff("", new) |
| 9 | write | local | 覆盖现有 LF 文件（content 也是 LF） | 行末符保持 LF；raw_output.created = false |
| 10 | write | local | **覆盖现有 CRLF 文件，content 给 LF** | 写回**全部 CRLF**（行末符回归）；§6.1 |
| 11 | write | local | 父目录不存在 | event = `Failed(Execution(NotFound))`；不创建 tmp 残留 |
| 12 | write | local | path 越界 | event = `Failed(Execution(NotPermitted))` |
| 13 | write | local | **写入中途模拟 panic（rename 前）** | 目标文件未被覆盖；§6.2 |
| 14 | edit | local | 唯一匹配 | raw_output.matches_replaced = 1；diff 正确 |
| 15 | edit | local | 多次匹配 + replace_all=false | event = `Failed(InvalidArgs("matched N times"))` |
| 16 | edit | local | 多次匹配 + replace_all=true | raw_output.matches_replaced = N |
| 17 | edit | local | 找不到 old_string | event = `Failed(InvalidArgs("not found"))` |
| 18 | edit | local | old_string == new_string | event = `Failed(InvalidArgs("must differ"))` |
| 19 | edit | local | old_string 为空字符串 | event = `Failed(InvalidArgs("must not be empty"))` |
| 20 | edit | local | **CRLF 文件 + LF `new_string`** | 写回纯 CRLF；§6.1 |
| 21 | read/write/edit | acp | 在 fake ACP client 上跑 #1 / #8 / #14 | 行为与 local 等价（从 wire 看到 `fs/read_text_file` / `fs/write_text_file` 反向请求） |
| 22 | 真实 e2e | local | deepseek "write a hello.txt with 'hello'" → `write_file` → 验证文件落盘 | TurnEnded = `EndTurn`；至少一次 ToolCallStarted/Finished |
| 23 | edit | local | **baseline 后外部修改文件，再写** | event = `Failed(Execution(Conflict))`；§5.4 v1 |
| 24 | edit | local | baseline 与二次 fingerprint 一致 | 正常完成；matches_replaced 正确 |
| 25 | read | local | **超过 `MAX_READ_BYTES` 的文件 + offset/limit 窗口** | 窗口内的行内容正确；不报 `TooLarge`；§3.1 v1 |
| 26 | read | local | 窗口本身累积 > `MAX_READ_BYTES` | event = `Failed(Execution(TooLarge))`（防绕过） |
| 27 | write | local | **现有文件 describe 阶段** | `content[0] = Diff(Some(old), new)`，diff 在授权前已是精确形态；§4.3 v1 |
| 28 | write | local | 新建文件 describe 阶段 | `content[0] = Diff(None, new)`（旧内容缺席不影响 diff 渲染） |

#5–#6（路径越界 / symlink）是回归基线。#10 / #13 / #20 是 §6 "不留坑"原则的回归基线（行末符腐蚀 / 半截文件）。#23 / #25 / #27 是 v1 三件事的回归基线（并发冲突 / 大文件分页 / describe 精确 diff）。#21 由 [`docs/inbound/acp-fs.md`](../inbound/acp-fs.md) §5 的 e2e 覆盖，本文测试矩阵列出但实现在 `defect-acp` crate 的 tests/ 下。

## 10. 落地节奏（与 [`acp-fs.md`](../inbound/acp-fs.md) 同步）

fs 工具与 ACP 委托是同一次落地——分三步前进：

1. **`crates/agent/`** —— 引入 `FsBackend` trait + `FsError`：
   - `crates/agent/src/fs/mod.rs`：trait + `FsError` + `resolve_workspace_path` helper（[`AcpFsBackend`] 共用）。
   - `crates/agent/src/tool.rs`：[`ToolContext`] 加 `fs: &dyn FsBackend` 字段；`ToolContext::new` 签名调整。
   - `crates/agent/src/session/`：`AgentCore::create_session` 签名加 `id: SessionId, fs: Arc<dyn FsBackend>`（[`acp-fs.md` §3.2](../inbound/acp-fs.md#32-session_id-时序问题)）；[`DefaultSession`] 持有 `fs`；`TurnRunner` / `drive_tool_stream` 把 `fs` 注入到 `ToolContext`。
2. **`crates/tools/src/fs/`**（新模块）—— [`LocalFsBackend`] + 三个工具：
   - `local_backend.rs`：[`LocalFsBackend`]（含 §6 行末符 + 原子写）。
   - `read.rs` / `write.rs` / `edit.rs`：三个 [`Tool`] 实现。
   - `tests.rs`：§9 #1–#20 / #22。
   - `lib.rs`：`pub mod fs; pub use fs::{LocalFsBackend, ReadFileTool, WriteFileTool, EditFileTool};`。
3. **`crates/acp/src/fs.rs`** —— [`AcpFsBackend`]（按 [`acp-fs.md` §3](../inbound/acp-fs.md#3-acpfsbackend)）；`crates/acp/src/serve.rs` 的 `initialize` / `session/new` handler 装配；`crates/acp/tests/fs_delegation.rs` 跑 §9 #21。
4. **`crates/cli/`** —— 默认装配把 [`LocalFsBackend`] 塞进 [`DefaultAgentCoreBuilder`]；e2e example 注册三个工具，verify §9 #22。
5. **更新 `TODO.MD`**：fs 工具 / ACP 文件系统委托两行同时翻到「已完成」。

## 11. 演进口子

下列每条都是诚实的"feature gap"——当前行为是 fail-loud 或 best-effort，**不会**静默走错路径。

- **fuzzy edit（多策略匹配）**：v0/v1 都是严格匹配，`old_string` 错一个空格 → `InvalidArgs("not found")` 立即可见。opencode `packages/opencode/src/tool/edit.ts` 的 9-replacer 链给出 fallback 路径；v2 可考虑整段移植，`edit_file` 增加 `match_strategy: "exact" | "lenient"`（默认仍 `exact`）。fuzzy 匹配的 surprise 不可控（同名变量误命中），v1 用户用 fail-loud 反馈学到"先 read 看原文再 edit"的 workflow，比无脑 fuzzy 稳。
- **二进制 / 多模态文件**：v0/v1 拒绝并报 `NotPermitted("binary")`。需要 ACP 协议扩展 `read_resource` 类方法（或类似），不是我们单方能做的事。
- **diff 算法升级**：当前 `Diff(old, new)` 让 ACP 客户端自己算渲染，agent 不发 patch hunk。claw-code `make_patch` / opencode `createTwoFilesPatch` 给客户端 UI 更细的 hunk 信息。后续可引入 `ToolCallContent::Diff` 的 hunk 模式（如 ACP 已支持）。
- **LSP / formatter 联动**：当前不触发任何后处理。opencode 写完后跑 LSP diagnostics 把 error 喂给 LLM。后续通过 ACP 反向请求或本地 LSP 客户端实现。
- **删除 / 移动 / 创建目录**：见 §1.2——不是"暂不做"，是"ACP 没对位反向方法、所以根本不进 fs 工具家族"。等 ACP 协议演进后再加。

### 11.1 v0 → v1 已落地

- **大文件分页 / streaming read**（v1）—— `read_file` 在 `offset` / `limit` 任一非空时走 line-by-line 流式跳过 + 累积，绕过整文件 `MAX_READ_BYTES = 10 MiB` 上限（窗口本身仍受限）。见 §3.1 v1 行 + 测试 #25 / #26。
- **并发写冲突检测**（v1）—— `edit_file` 在 read 后取 baseline fingerprint，write 前再取一次；不一致 → `Execution(Conflict)`。fingerprint 取/对比 best-effort，不阻塞主路径。见 §5.4 v1 + 测试 #23 / #24。
- **`write_file` describe 阶段精确 diff**（v1）—— `Tool::describe` 改为 async 并持有 `ToolContext`；`write_file` 在 describe 期 `ctx.fs.read_text(path).await.ok()` 拿旧内容，授权 wire payload 即含精确 diff。见 §4.3 v1 + 测试 #27 / #28。
