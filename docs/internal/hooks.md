# Hook 系统设计

`Hook` 是 `defect-agent` 在主循环关键时刻让外部代码 / Builtin 逻辑 / 显式 LLM 调用介入的扩展点。本文沉淀事件枚举、Handler trait、配置形态、信任模型，与 [`event-model.md`](./event-model.md) / [`turn-loop.md`](./turn-loop.md) / [`sandbox-policy.md`](./sandbox-policy.md) 的边界划分。

设计前提：
- 三家参考实现里 claude-code / codex 都把 shell 当成 hook 的唯一执行器；defect 是纯 Rust 单二进制 ACP server，跑在 headless 容器 / Zed 嵌入式 / 未来 WASM 等没有 shell 的环境里**也得能用**——shell 必须是"可选的执行器之一"而不是入口。
- 事件粒度 v0 落 5 件套（`SessionStart` / `UserPromptSubmit` / `PreToolUse` / `PostToolUse` / `PostToolUseFailure`），其余事件**enum 打桩**但主循环不 emit；演进时直接接入。
- 5 件套全部为 **Sync 拦截点**——主循环必须等所有匹配 handler 跑完才能继续；超时 / 错误按 §3.5 处理。

## 1. 定位与术语

| 概念 | 含义 |
| --- | --- |
| **HookEvent** | 主循环在某个时刻把"现在发生了 X"打包给 hook 引擎的载荷 |
| **HookHandler** | 处理 HookEvent 的一个具体执行器实例（Builtin / Command / Prompt） |
| **HookOutcome** | Handler 返回给主循环的结果——可同时携带 block / patch / append 三类副作用 |
| **Hook engine** | `defect-agent` 内置的派发器：路由 event 到匹配的 handlers、串行 pipeline 执行、合并 outcome、实施超时与信任检查 |

Hook **不是**：
- **第二条事件总线**——观察类 hook 直接订阅 [`AgentEvent`](./event-model.md) 流，不再独立 fan-out。
- **权限决策器**——sandbox policy 仍然是工具放行权威；hook 在 `PreToolUse` 上只能投"建议票"。多个 hook 投票时主循环取最严（任一 `block` 即整体拒绝）。
- **Tool**——`Tool` trait 暴露给 LLM 调用；hook 由主循环触发，模型完全不感知。

### 1.1 Sync vs Async：两条接入路径

事件按主循环对它的等待方式分两类：

| 类别 | 主循环行为 | 事件 | 允许的 outcome |
| --- | --- | --- | --- |
| **Sync 拦截** | emit 后**阻塞 await** 所有匹配 handler，按 outcome 决定走向 | `SessionStart` / `UserPromptSubmit` / `PreToolUse` / `PostToolUse` / `PostToolUseFailure` | 见 §3.3 各事件列 |
| **Async 观察** | 不阻塞主循环；handler 通过订阅 `AgentEvent` 流"事后"处理 | `SessionEnd` / `TurnStart` / `TurnEnd` / `PreLlmCall` / `PostLlmCall` / `PreCompact` / `PostCompact` / `PermissionAsk` | 仅 `Pass`（log/metrics 副作用自负） |

> 一个事件**只能**属于一类。`PostToolUse` 在 v0 是 Sync——主循环要让 hook 有机会在 `tool_result` 写进 history 之前追加注释（见 §7.1）；它不再走 AgentEvent 订阅。

> Async 观察的 handler 严格只读：返回任何非 `Pass` 的 outcome（`block` / `patch` / 非空 `append`）由引擎报错丢弃并打 warning——这条不变量保证 observer 不会因为"我加了一个 hook 想 block 一下"把主循环搞死。

## 2. HookEvent 枚举

完整打桩（`#[non_exhaustive]`）；v0 主循环只 emit 标 ✓ 的 5 件，其他变体编译期就在但运行时不触发。

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HookEvent<'a> {
    // ── Sync 拦截（v0 实际接入） ──
    SessionStart      { source: SessionSource, cwd: &'a Path },                    // ✓
    UserPromptSubmit  { content: &'a [ContentBlock] },                             // ✓
    PreToolUse        { id: &'a ToolCallId, name: &'a str,
                        args: &'a Value, safety: SafetyClass },                    // ✓
    PostToolUse       { id: &'a ToolCallId, name: &'a str, fields: &'a ToolCallUpdateFields }, // ✓
    PostToolUseFailure{ id: &'a ToolCallId, name: &'a str, error: &'a str },       // ✓

    // ── Async 观察（v0 enum 打桩，订阅 AgentEvent 即可使用） ──
    SessionEnd        { reason: AcpStopReason },
    TurnStart         { prompt: &'a [ContentBlock] },
    TurnEnd           { reason: AcpStopReason, usage: &'a Usage },
    PreLlmCall        { model: &'a str, attempt: u32 },
    PostLlmCall       { model: &'a str, attempt: u32, usage: &'a Usage, error: Option<&'a str> },
    PreCompact        { tokens_before: u64 },
    PostCompact       { tokens_before: u64, tokens_after: u64 },
    PermissionAsk     { id: &'a ToolCallId, request: &'a RequestPermissionRequest },
}

#[non_exhaustive]
pub enum SessionSource {
    /// 全新创建的 session
    New,
    /// resume 既有 session
    Resume { session_id: SessionId },
}
```

### 2.1 命名对位

|  defect HookEvent | claude-code | codex | opencode |
| --- | --- | --- | --- |
| `SessionStart` | — | `SessionStart` | — |
| `UserPromptSubmit` | — | `UserPromptSubmit` | `chat.message` |
| `PreToolUse` | `PreToolUse` | `PreToolUse` | `tool.execute.before` |
| `PostToolUse` | `PostToolUse` | `PostToolUse` | `tool.execute.after` |
| `PostToolUseFailure` | `PostToolUseFailure` | (合并进 PostToolUse) | (合并进 after) |
| `PreCompact` / `PostCompact` | — | `PreCompact` / `PostCompact` | `experimental.session.compacting` |
| `PermissionAsk` | (走 PreToolUse 的 permissionDecision) | `PermissionRequest` | `permission.ask` |

### 2.2 为什么字段是借用而不是拥有

`HookEvent` 是主循环 emit 入口的形状；engine 内部派发到 handler 时按需 clone 到 owned 形态（pipeline 还要在 handler 之间传递可变 state——见 §3.4）。这样：

- 主循环 emit 不付 clone 代价
- enum 体积稳定（无大字段沉到栈上）
- handler 拿到的是 owned 副本，不被借用生命周期约束

具体 owned 形态见 §4——`Command` handler 序列化成 JSON 喂 stdin、`Prompt` handler 转成 `ContentBlock` 喂 LLM、`Builtin` handler 直接看 owned 引用。

### 2.3 为什么 `PostToolUseFailure` 单独一件

claude-code 把成功 / 失败拆开两个事件；codex / opencode 合在一起靠字段区分。defect 跟 claude 一致：

- 失败路径上 `fields` 通常没有有意义的字段——只有 `error` 文本，独立载荷更准确
- 用户配置里"失败时跑某个审计脚本"是常见用例，独立事件让 matcher 简单（`pre_tool_use` / `post_tool_use_failure` 直接分桶，不用每个 handler 自己判断 `is_error`）
- enum variant 廉价；合并不省什么

## 3. HookHandler trait

```rust
pub trait HookHandler: Send + Sync {
    /// handler 自我宣告期望的语义类别——决定它能挂在哪一类事件上。
    fn capability(&self) -> HookCapability;

    /// 执行 hook。返回 outcome 给引擎；具体效果见 §3.3。
    fn handle(&self, ev: &HookEvent<'_>, ctx: HookCtx<'_>)
        -> BoxFuture<'_, Result<HookOutcome, HookError>>;
}

pub enum HookCapability {
    /// 仅适用于 Async 观察事件——只能日志/审计，引擎对返回非空 outcome 报错。
    Observe,
    /// 完整能力——可 block / patch / append。仅适用于 Sync 拦截事件。
    Intercept,
}

pub struct HookCtx<'a> {
    pub session_id: &'a SessionId,
    pub cwd: &'a Path,
    pub cancel: CancellationToken,
}
```

> 异步 trait 用 `BoxFuture` 是 [No async_trait] 约定；workspace 不引入 `async-trait` crate。

[No async_trait]: ../../CLAUDE.MD

### 3.1 HookOutcome：可组合结构

```rust
#[derive(Default)]
#[non_exhaustive]
pub struct HookOutcome {
    /// 早退理由。Some 时引擎不再调用后续 handler；其他字段忽略。
    pub block: Option<String>,

    /// 修改 in-flight 数据。具体补丁形态由事件决定。
    pub patch: Option<HookPatch>,

    /// 追加 system context / tool_result 注释（具体落点见 §3.3）。
    pub append: Vec<ContentBlock>,
}

#[non_exhaustive]
pub enum HookPatch {
    /// 用于 PreToolUse：替换工具参数。
    ToolArgs(Value),
    /// 用于 UserPromptSubmit：在用户原文前后追加内容（不允许完全替换，见 §3.6）。
    UserPrompt { prepend: Vec<ContentBlock>, append: Vec<ContentBlock> },
}
```

`Pass` = 全字段默认值（`block = None` / `patch = None` / `append = []`）。

> `HookOutcome` 是结构体而不是 enum，是为了让一个 handler 一次返回"修改 args 同时追加 system context"——这种组合在 stdout JSON schema（§4.2.2）和 builtin handler 实现里都自然出现，enum 形态强行三选一会逼用户写多个 handler 跳两次 pipeline。

### 3.2 各事件支持的 outcome 类型

| 字段 | `SessionStart` | `UserPromptSubmit` | `PreToolUse` | `PostToolUse` | `PostToolUseFailure` |
| --- | --- | --- | --- | --- | --- |
| `block` | ✗（有值则丢弃 + warning） | ✓（拒绝 turn） | ✓（拒绝该工具） | ✗（丢弃） | ✗（丢弃） |
| `patch = ToolArgs` | ✗ | ✗ | ✓ | ✗ | ✗ |
| `patch = UserPrompt` | ✗ | ✓ | ✗ | ✗ | ✗ |
| `append` | ✓（系统 prompt） | ✓（系统 prompt） | ✗（无意义） | ✓（拼到 tool_result.content） | ✓（拼到 tool_result.content） |

要点：

- **SessionStart 不允许 Block**：克隆陌生仓库时 hook 还没经过信任审查，能 block session 就是攻击面。SessionStart 等同于 "Sync 但 outcome 限定 Pass / Append"——主循环必须等它跑完（要拼 append 进 system prompt），但 handler 拒不了 session 启动。
- **Post* 不允许 Block**：工具已经跑完了，事后 block 没有可恢复语义。`block` 字段被引擎丢弃并 warn。
- **`append` 在 Post* 上的落点**：拼到该工具调用产出的 `tool_result.content` 末尾（见 §7.1 主循环改动）。下一轮 LLM 会把 hook 的注释当作 tool 输出的一部分看到——典型用途是"工具失败时让 hook 注入修复建议"。

### 3.3 不允许的字段被设置时如何处理

引擎按"宁可丢弃也不报错"的原则降级：

- `block` 在不允许 block 的事件上出现 → **丢弃**该字段，pipeline 继续，warn 一行
- `patch` 类型与事件不匹配（如 `PreToolUse` 上返回 `UserPrompt patch`） → **丢弃** patch，warn
- `append` 在 `PreToolUse` 上出现 → **丢弃** append，warn（PreToolUse 没有自然落点）

不向上抛错的原因：handler 实现可能跨多个事件复用同一段返回逻辑；严格检查会让 handler 写起来很烦。warning 足够覆盖配置 bug 的可观测性。

### 3.4 Pipeline：handler 串行 + 状态累积

匹配同一事件的 handler 按 **TOML 声明顺序** 串行调用。每个 handler 看到的事件 = **前序 handler 应用 patch 之后**的事件——这是 pipeline 而不是覆盖。

#### 3.4.1 内部状态

引擎维护一个 owned `EventState`：

```rust
struct EventState {
    event: OwnedHookEvent,         // 跟随 patch 演化的事件副本
    accumulated: HookOutcome,      // 累积的 patch / append；不含 block
}
```

每次调用 handler 之前，`EventState::event` 反映上一个 handler 的 patch 已应用；调用后按下表合并：

| 字段 | 合并规则 |
| --- | --- |
| `block` | 任一 handler 设置 `block = Some(_)` → engine 立即终止 pipeline，返回 `block` 为该值 |
| `patch = ToolArgs(v)` | 直接覆盖 `EventState::event` 中的 args，并把 `accumulated.patch = Some(ToolArgs(v))`（最终 outcome 反映"最后状态"） |
| `patch = UserPrompt { prepend, append }` | `EventState::event` 的内容：`[old_prepend, new_prepend, original_user_text, old_append, new_append]`；`accumulated.patch` 同步更新为合并结果 |
| `append` | `EventState::accumulated.append.extend(handler.append)`，按调用顺序拼接 |

#### 3.4.2 例子：PreToolUse pipeline

```
TOML 顺序：
  [[hooks.pre_tool_use]] handler = redact-secrets   (Builtin: 把 args 里 password 字段替换成 "***")
  [[hooks.pre_tool_use]] handler = audit-bash       (Command: 看 args 是不是黑名单命令)

主循环 emit PreToolUse { args: { command: "echo p=mysecret" } }
  ↓
[redact-secrets] 看到 args = { command: "echo p=mysecret" }
                 返回 { patch: ToolArgs({ command: "echo p=***" }) }
  ↓
EventState.event 更新为 args = { command: "echo p=***" }
  ↓
[audit-bash]    看到 args = { command: "echo p=***" }   ← 这里看到的是 redact 后的
                 返回 Pass
  ↓
最终 outcome = { patch: Some(ToolArgs({ command: "echo p=***" })), append: [] }
```

如果 `audit-bash` 反过来设置 `block = Some(...)`，engine 就**不再调用之后的 handler**，返回 block。

#### 3.4.3 为什么 pipeline 而不是覆盖

覆盖语义下"hook A 改路径 → hook B 复查改后路径"不成立——hook B 看到的还是原路径。pipeline 才能让"路径归一化 → 敏感词检查 → 审计"这种链式配置工作。

代价：handler 副作用顺序敏感，TOML 写错顺序行为会变。这是接受的——所有 pipeline 系统（middleware、interceptor）都有这个性质，文档显式约定即可。

### 3.5 HookError 与降级

```rust
#[non_exhaustive]
pub enum HookError {
    Timeout,
    HandlerFailed(BoxError),
    /// handler 信任未通过 / 未注册等配置层错误
    Configuration(String),
}
```

引擎对 `HookError` 的处理因事件类别而异：

| 事件类别 | 处理 |
| --- | --- |
| **Sync 拦截 + 允许 block 的事件**（`UserPromptSubmit` / `PreToolUse`） | 错误等价于 `block = Some(error.to_string())`——保守语义："hook 出问题就别让工具跑" |
| **Sync 拦截 + 不允许 block 的事件**（`SessionStart` / `PostToolUse[Failure]`） | 降级为 warning 日志，pipeline 继续；hook 失败永远不阻塞 session 启动 / tool_result 落盘 |
| **Async 观察事件** | 降级为 warning 日志，主循环继续 |

panic：所有事件类别一律捕获并转为 `HandlerFailed`，按上表降级处理。

### 3.6 UserPrompt 修改语义

`HookPatch::UserPrompt` 不允许"完全替换"——只能 `prepend` / `append`。理由：

- 完全替换等于 hook 偷换用户输入，给 LLM 看到的不是用户实际输入了什么；安全 / 审计上劣质。
- 真正的 use case（注入项目上下文 / 注入 skill 触发词）都是"在用户原文前后加东西"。
- `append` 字段（顶层 outcome）走 system prompt（不污染 user message），`patch = UserPrompt` 走 user message（让模型把 prepend/append 当成用户的话）——两条路都留给 hook 选。

## 4. Handler 实现

v0 内置三种 handler：`Builtin` / `Command` / `Prompt`。

### 4.1 Builtin

`crate::hooks::builtin::*` 注册给 hook engine 的 in-process Rust handler。最低成本、零外部依赖。

```rust
pub fn registry() -> Vec<(&'static str, Arc<dyn HookHandler>)> {
    vec![
        ("tracing-audit",   Arc::new(TracingAuditHook)),
        ("redact-secrets",  Arc::new(RedactSecretsHook)),
        // 后续 skill 加载、preload 项目元信息也走这条路（见 §9）
    ]
}
```

用户在 TOML 里以 `name` 引用：

```toml
[[hooks.post_tool_use]]
handler = { type = "builtin", name = "tracing-audit" }
```

未注册的 builtin name 在配置加载期 fail-fast——别让用户在 turn 跑到一半才发现拼错。

### 4.2 Command

执行外部命令。**默认走 argv，不经任何 shell**；shell 是显式 opt-in。

```rust
pub enum CommandSpec {
    /// 直接 spawn，不经 shell。
    Argv {
        argv: Vec<String>,
        argv_windows: Option<Vec<String>>,
        cwd: Option<PathBuf>,
        env: BTreeMap<String, String>,
        timeout_sec: Option<u64>,
    },
    /// 显式 shell。`shell` 字段必须存在，引擎不再"自动选 sh"。
    Shell {
        shell: ShellKind,
        command: String,
        cwd: Option<PathBuf>,
        env: BTreeMap<String, String>,
        timeout_sec: Option<u64>,
    },
}

#[non_exhaustive]
pub enum ShellKind {
    Sh,
    Bash,
    Pwsh,
    Cmd,
    Custom { program: String, args: Vec<String> /* 不含 command 本身 */ },
}
```

> **平台前提**：`Command` handler 不依赖 shell，但仍然依赖宿主具备**进程执行能力**（典型 OS / 容器）。真正没有 spawn 子进程能力的运行时（如纯 WASM）下整个 `Command` handler 类型由 `hooks-command` cargo feature flag 关闭，由 `Builtin` 与 `Prompt` 兜底。

#### 4.2.1 IO 协议

| 通道 | 协议 |
| --- | --- |
| stdin | `HookEvent` 的 owned 形态序列化为 JSON，单行 |
| stdout | 解析为 JSON 对象，按 §4.2.2 字段决定 outcome |
| stderr | 透传到主循环 tracing（debug 级），不影响 outcome |
| exit code | 0 = 按 stdout 决定；非 0 = `HookError::HandlerFailed`（按 §3.5 表降级） |

环境变量按事件注入：

| 事件类别 | 注入变量 |
| --- | --- |
| 全部事件 | `DEFECT_HOOK_EVENT`（事件名 snake_case）/ `DEFECT_SESSION_ID` / `DEFECT_CWD` |
| `PreToolUse` / `PostToolUse` / `PostToolUseFailure` | 加 `DEFECT_TOOL_NAME` / `DEFECT_TOOL_INPUT`（args 的 JSON 字符串） |
| `PostToolUseFailure` | 再加 `DEFECT_TOOL_ERROR`（错误文本） |
| `UserPromptSubmit` | 加 `DEFECT_USER_PROMPT`（concat 所有 text block 后的纯文本） |

stdin / stdout 与 env 同时存在是为了让脚本作者按习惯选——简单审计读 env 写 stderr 即可，复杂修改用 stdin 拿完整 JSON、stdout 回结构化结果。

#### 4.2.2 stdout JSON schema

```json
{
  "block": "string (optional, set means block with this reason)",
  "patch": { "tool_args": {...} } | { "user_prompt": { "prepend": [...], "append": [...] } },
  "append": [ContentBlock, ...]
}
```

字段缺省 = `Pass`。**不解析为 JSON 不算错**——空 stdout / 非 JSON stdout 都视为 `Pass`，便于轻量审计脚本（只 echo 日志）。但只要 stdout 含合法 JSON 就严格按 schema 解析；schema 不匹配 = `HookError::HandlerFailed`。

`block` / `patch` / `append` 三字段可同时出现——一次返回里既改 args 又追加 system context 是合法组合（与 §3.1 `HookOutcome` 1:1 对位）。

#### 4.2.3 取消 / 超时

- engine 用 `tokio::process::Command` + `kill_on_drop(true)` spawn
- timeout：`tokio::time::timeout(spec.timeout_sec.unwrap_or(default))` —— 默认见 §8
- `ctx.cancel.cancelled()` 触发：drop child（kill），返回 `HookError::Timeout`（业务上的"未完成"，与超时等价）

#### 4.2.4 与 claude-code / codex 的差异

| 维度 | claude-code | codex | defect |
| --- | --- | --- | --- |
| 默认执行器 | `sh -lc` / `cmd /C` | 同 | **直接 spawn argv** |
| 显式 shell | 无 | 无 | `{ shell = "sh", command = "..." }` |
| 跨平台 | `command_windows` 字段 | 同 | `argv_windows` 字段 |
| 字符串自动分词 | 是 | 是 | **不提供**（避免 shlex 在 Windows 上的争议） |

迁移指南：claude-code 用户原本写 `["./hook.sh"]` 在 defect 里直接以 `argv = ["./hook.sh"]` 跑（前提是脚本有 `#!/bin/sh` shebang）；想保留 `sh -lc <command>` 的语义就改写成 `{ shell = "sh", command = "..." }`。

### 4.3 Prompt

把 hook event 喂给一次 LLM 调用。典型用途：冷启动加载项目元信息成 system prompt、按 prompt 内容动态选 skill（见 §9）。

```rust
pub struct PromptHandlerSpec {
    /// 用哪个 model（缺省走 session 默认 model）
    pub model: Option<String>,
    /// 固定 system prompt 模板
    pub system: String,
    /// 把 HookEvent 渲染成 user message 的策略
    pub render: PromptRender,
    pub timeout_sec: Option<u64>,
}

pub enum PromptRender {
    /// 直接喂 JSON 序列化结果
    Json,
    /// 用 handlebars 模板从 event 字段取值
    Template { template: String },
}
```

执行流程：

1. engine 把 `HookEvent` 按 `render` 渲染成 `Vec<ContentBlock>`
2. 用 `LlmProvider::complete` 跑一次（**不入 history、不计 turn_request_count**）
3. LLM 输出文本作为 `HookOutcome { append: vec![text_block], .. }` 返回
4. 失败 / 超时按 §3.5 表降级（`SessionStart` 上失败 = warning，不阻塞启动）

#### 4.3.1 关键约束

- **不进 turn 主循环的 LLM 调用计数**：避免一个 SessionStart hook 把用户的 `max_turn_requests` 消耗一次。
- **冷启动可降级**：`SessionStart` 触发 Prompt handler 时若 LLM provider 还没握手就绪（典型：CLI 刚起来 provider 注册顺序），handler 必须返回 `Pass` 而不是 Error——session 启动不能被阻塞。具体由 `PromptHandler` impl 内部判断 `provider.is_ready()`。
- **不允许 Prompt handler 套 Prompt handler**：handler 内部触发的 LLM 调用不再 emit hook 事件（避免无限递归）。引擎用 task-local flag 实现：进入 handler 调用前 set `in_hook = true`，退出后清。

#### 4.3.2 为什么不让 Prompt handler 接入 PreToolUse

技术上可以；但 `PreToolUse` 每次工具调用都触发 = 每次 tool 都跑一次 LLM = turn 时延翻倍 + token 浪费。**v0 不在文档示例里展示这种用法**，但 trait 不禁止——用户配错 schema 引擎不会拦着，仅在 tracing 日志里打 warning。

## 5. 配置

### 5.1 文件位置

复用 [`docs/architecture.md`](../architecture.md) §2.1 的配置层级：

```
default < user < project < project-local < CLI
```

具体：

- 用户：`$XDG_CONFIG_HOME/defect/config.toml`
- 项目共享：`<repo>/.defect/config.toml`
- 项目本地：`<repo>/.defect/config.local.toml`（**默认禁用 hook**，见 §6）

### 5.2 TOML 形态

```toml
# 工具拦截
[[hooks.pre_tool_use]]
match.tool = "bash"
match.safety = ["destructive"]          # 数组形式；按 SafetyClass 过滤，可写多个
handler = { type = "command",
            argv = ["./scripts/audit.sh"],
            argv_windows = ["pwsh", "-File", "./scripts/audit.ps1"],
            timeout_sec = 10 }

[[hooks.pre_tool_use]]
match.tool = "edit"
handler = { type = "command",
            shell = "bash",
            command = "shellcheck \"$DEFECT_TOOL_INPUT\"" }   # 见 §4.2.1 env 表

# Builtin
[[hooks.post_tool_use]]
handler = { type = "builtin", name = "tracing-audit" }

# Prompt：冷启动注入项目元信息
[[hooks.session_start]]
handler = { type = "prompt",
            system = "你是项目摘要助手...",
            render = { type = "template", template = "项目位于 {{cwd}}" },
            timeout_sec = 5 }

# UserPromptSubmit 上的 skill 触发器（见 §9）
[[hooks.user_prompt_submit]]
handler = { type = "builtin", name = "skill-router" }
```

### 5.3 matcher 字段

```rust
#[serde(default)]
pub struct HookMatcher {
    /// 按工具名精确匹配（仅 *ToolUse* 事件）
    pub tool: Option<String>,
    /// 按工具名 glob 匹配（仅 *ToolUse* 事件）
    pub tool_glob: Option<String>,
    /// 按 SafetyClass 过滤（仅 PreToolUse）；数组语义为"任一匹配即命中"
    pub safety: Option<Vec<SafetyClass>>,
}
```

字段全空 = 匹配该事件的所有触发。matcher 不放正则——glob 已经够；regex 让 hook 匹配开销不可控（codex 用 regex，但 codex 的事件量远低于 defect 在长跑 turn 里 PreToolUse 的频率）。

### 5.4 配置合并

claude-code 已知 issue：下游层级的 hook 数组**完全替换**上游层级的，导致项目本地 toml 可以静默移除安全 hook。defect 的合并规则是 **append + 去重**：

- 同一个事件下的 hook 数组从上到下 append（保留 TOML 声明顺序——pipeline 语义见 §3.4）
- 完全相同的 (matcher, handler) 元组去重一次
- **不**让 project-local 取消 user / project 的 hook

要"取消"上游 hook，必须用 `[[hooks.disable]]` 显式声明：

```toml
[[hooks.disable]]
event = "pre_tool_use"
handler = { type = "builtin", name = "tracing-audit" }
```

这是个噪声大的语法，但好处是明面：disable 是审计可见的一行，而不是悄悄替换数组。

## 6. 信任模型

`.defect/config.local.toml`（项目本地、默认不入 git）的 hook **默认禁用**——克隆陌生仓库不会让攻击者控制的 hook 直接跑。

```rust
pub struct HookTrust {
    /// 该 hook 配置的稳定哈希（涉及字段：matcher + handler）
    pub hash: String,
    /// 用户显式 trust 的 hash 列表（持久化在 user-level config）
    pub trusted: HashSet<String>,
}
```

CLI：

```bash
defect hooks list           # 列出所有 hook 与信任状态（trusted / untrusted）
defect hooks trust <hash>   # 显式信任，写入 user-level config
defect hooks revoke <hash>  # 撤销
```

未信任的 hook 在 engine 加载时**直接丢弃**（warn 一行），不参与任何派发。这条规则只针对 `project-local` 层；`user` / `project` 层的 hook 默认信任（用户已经主动放进自己 home / 已经主动 commit 进 repo，相当于隐式信任）。

> 与 codex 的 `trusted_hash` 等价；与 claude-code 的"全开 + issue #106 已知缺陷"明确区别开。

## 7. 与 AgentEvent / Turn / Sandbox 的关系

### 7.1 主循环新增的 await 点（Sync 拦截事件）

5 件套都是主循环的同步阻塞点。伪代码：

```rust
// session 创建：DefaultAgentCore::create_session 内
fn create_session(&self, ...) {
    let outcome = self.hooks.fire(HookEvent::SessionStart { source, cwd }, ctx).await;
    // outcome.block 永远是 None（被引擎丢弃）；只取 append
    let preload_blocks = outcome.append;
    let session = DefaultSession::new(..., preload_blocks);  // 拼到 system prompt 后缀
}

// turn 主循环：DefaultSession::run_turn 内
fn run_turn(&self, prompt: Vec<ContentBlock>) {
    // ① UserPromptSubmit
    let outcome = self.hooks.fire(HookEvent::UserPromptSubmit { content: &prompt }, ctx).await;
    if let Some(reason) = outcome.block { return Ok(StopReason::Refusal /* with reason */); }
    let prompt = apply_user_prompt_patch(prompt, outcome.patch);
    let prompt = prepend_system_blocks(prompt, outcome.append);  // append 走 system prompt
    self.history.append(Message::user(prompt));

    loop {
        // ... drain LLM stream → outcome.tool_uses ...

        for tu in &mut outcome.tool_uses {
            // ② PreToolUse
            let pre = self.hooks.fire(HookEvent::PreToolUse {
                id: &tu.id, name: &tu.name, args: &tu.args, safety: tool.safety_hint(&tu.args)
            }, ctx).await;
            if let Some(reason) = pre.block { mark_denied(tu, reason); continue; }
            if let Some(HookPatch::ToolArgs(new_args)) = pre.patch { tu.args = new_args; }
            // pre.append 在 PreToolUse 上无落点，被引擎丢弃 + warn
        }

        // ... policy.classify / 工具执行 → results ...

        for result in &mut results {
            // ③ PostToolUse / PostToolUseFailure
            let ev = if result.is_error {
                HookEvent::PostToolUseFailure { id: &result.id, name: &result.name, error: &result.error }
            } else {
                HookEvent::PostToolUse { id: &result.id, name: &result.name, fields: &result.fields }
            };
            let post = self.hooks.fire(ev, ctx).await;
            // post.block 被丢弃；append 拼到 tool_result.content 末尾
            result.append_content(post.append);
        }

        self.history.append(assistant_message(...));
        self.history.append(tool_result_message(results));  // 注释已经在上一步合进去了
    }
}
```

`fire` 由 `TurnRunner` / `DefaultSession` 持有的 `&dyn HookEngine` 实现，签名见 [§8 HookEngine trait](#8-hookengine-trait)。

### 7.2 与 AgentEvent（Async 观察事件）

观察类 hook（`SessionEnd` / `TurnStart` / `TurnEnd` / `PreLlmCall` / `PostLlmCall` / `PreCompact` / `PostCompact` / `PermissionAsk`）的事件来源**直接订阅 [AgentEvent 流](./event-model.md)**——不在主循环里专门 emit 第二条事件。具体路径：

```text
主循环 ──► AgentEvent ──┬─► defect-acp / defect-storage / tracing
                       │
                       └─► hook engine（订阅者之一）
                              │
                              ▼
                        派发到 observe-only handler
```

观察类 handler 永远不能改变主循环行为（`HookCapability::Observe` 强约束）：返回非空 outcome 由引擎丢弃 + warn，主循环不感知。

> 一个 handler 不能既挂 Sync 又挂 Async 事件——`capability()` 二选一，配置加载时按事件类别校验。

### 7.3 与 sandbox policy

sandbox policy 与 hook 是**两个独立的检查点**：

```text
LLM tool_use ──► PreToolUse hook ──► policy.classify ──► tool.execute
                  ↑ 任一 block 即 Denied（早退）
```

hook **先于** policy。理由：
- 让 hook 能在 policy 看到之前替换 args（场景：路径 normalize、敏感字段脱敏）
- 让 hook 能短路掉永远不该跑的工具调用（场景：黑名单脚本，policy 都不需要算）

policy 仍然是工具放行的最终权威——hook 只能 `block` 或 `patch`，不能"批准" policy 想 deny 的工具。"hook 投票 allow 但 policy deny" → 走 policy 决策（denied）。

### 7.4 与 PermissionAsk

`PermissionAsk` 当前是 Async 观察事件、v0 不接入。**未来如果要让 hook 介入权限决策**，有两种路线：

1. 把 `PermissionAsk` 升级成 Sync 拦截事件——主循环在 policy `Ask` 决定要请权限、ACP `RequestPermissionRequest` 还没发出去之间触发；handler 可 `block`（替主循环回答 deny）/ `append`（注入 reasoning hint）。
2. 把它继续保留为 Async，让 hook 只做审计。

选哪条路线由后续真实需求驱动；v0 仅承诺 enum 已经留好这个 variant。

## 8. HookEngine trait

```rust
pub trait HookEngine: Send + Sync {
    /// Sync 拦截事件入口：emit 一个事件，等所有匹配 handler 跑完返回合并 outcome。
    fn fire<'a>(&'a self, ev: HookEvent<'a>, ctx: HookCtx<'a>)
        -> BoxFuture<'a, HookOutcome>;

    /// Async 观察入口：从 AgentEvent 投影出 HookEvent 并派发，不阻塞调用方。
    /// fan-out task 内部调用，主循环不直接用。
    fn observe<'a>(&'a self, ev: HookEvent<'a>, ctx: HookCtx<'a>);
}
```

实现要点：

- 默认实现 `DefaultHookEngine` 持有 `BTreeMap<HookEventKind, Vec<(Matcher, Arc<dyn HookHandler>)>>`
- `fire` 内部按 matcher 过滤 + 串行 await（block 早退；pipeline 状态见 §3.4）
- 默认超时（拦截类）：5 秒；handler spec 自带 `timeout_sec` 时取 spec 值
- `observe` 不超时（observe-only handler 阻塞订阅链上自己；按 [`event-model.md`](./event-model.md) §5 的 backpressure 语义"不丢事件、可阻"）
- 派发期捕获 panic，转 `HookError::HandlerFailed`，按 §3.5 降级

注入位置：

- `DefaultAgentCoreBuilder::hook_engine(Arc<dyn HookEngine>)` —— 默认 `DefaultHookEngine::from_config(config)`
- `DefaultSession` / `TurnRunner` 通过 `&'a dyn HookEngine` 借用，与 `policy` / `tools` / `provider` 同形

## 9. 演进口子：Skill 加载 / 冷启动 preload

这两个用例既是 hook 系统的**首批用户**，也是验证 enum / handler 设计是否够用的试金石。本节给出预期接入形态——具体落地见后续 `docs/internal/skills.md`（待写）。

### 9.1 冷启动项目预载

需求：session 启动时把项目摘要（README 摘要 / `AGENTS.md` / git 状态 / cwd 文件树）塞进 system prompt，让 LLM 第一轮就有项目上下文。

接入：`SessionStart` hook 链——

```toml
[[hooks.session_start]]
handler = { type = "builtin", name = "preload-project-readme" }   # 读 README/AGENTS.md，零 LLM 调用

[[hooks.session_start]]
handler = { type = "builtin", name = "preload-git-status" }        # git status / git log -3

[[hooks.session_start]]
handler = { type = "prompt",                                       # 可选：让 LLM 总结
            system = "你是项目摘要器，把以下文件压成 200 字...",
            render = { type = "template",
                       template = "{{readme_excerpt}}\n{{git_status}}" },
            timeout_sec = 8 }
```

每个 handler 返回 `HookOutcome { append: vec![text_block], .. }`，引擎按 §3.4 顺序拼接到 session 的 system prompt 后缀。

`SessionStart` 的语义按 §3.2 / §3.5：

- handler 是 Sync 阻塞——主循环必须等 append 拼完才能让 session 进入可接受 prompt 的状态
- 但 outcome 中的 `block` 字段被引擎**丢弃**——session 启动不会被 hook 拒绝
- handler 失败按 §3.5 降级 warning——preload 失败永远不阻塞 session 启动

### 9.2 Skill 动态加载

skill = 一组以 markdown 形式定义的"工具用法 / 规约 / 提示词片段"，按条件加载到 system prompt（参考 `TODO.MD` 远期目标 + Anthropic 的 skill 概念）。

三种触发时机：

| 时机 | hook 事件 | handler 类型 | 备注 |
| --- | --- | --- | --- |
| 启动时全量注入 always-on skill | `SessionStart` | Builtin（读 `.defect/skills/*.md` 中 `always: true` 的） | 同 §9.1 |
| 每轮按 prompt 内容动态匹配 | `UserPromptSubmit` | Builtin（按 frontmatter glob 匹配 prompt 文本） | 命中即在 outcome.append 里返回 |
| LLM 自己按需加载 | — | Tool（不是 hook） | 走 `Tool` trait 的 `skill` 工具，不在本文范围 |

Builtin handler 形态（`crate::hooks::builtin::skill_router`）：

```rust
struct SkillRouterHook { skills: Vec<SkillDescriptor> }

impl HookHandler for SkillRouterHook {
    fn capability(&self) -> HookCapability { HookCapability::Intercept }  // 挂 Sync 事件需要 Intercept

    fn handle(&self, ev: &HookEvent<'_>, _: HookCtx<'_>) -> BoxFuture<'_, ...> {
        Box::pin(async move {
            let HookEvent::UserPromptSubmit { content, .. } = ev else {
                return Ok(HookOutcome::default());  // Pass
            };
            let prompt_text = collect_text(content);
            let matched: Vec<_> = self.skills.iter()
                .filter(|s| s.matches(&prompt_text))
                .map(|s| ContentBlock::text(s.body.clone()))
                .collect();
            Ok(HookOutcome { append: matched, ..Default::default() })
        })
    }
}
```

注意：

- skill router 用 `append`（系统 prompt 注入），不 `patch = UserPrompt`——保留用户原文不被改写，符合 §3.6。
- `SkillDescriptor::matches` 的具体匹配实现（glob frontmatter / regex / LLM 二次判定）是 skill 子系统设计 (`docs/internal/skills.md`) 的范畴；hook 这一层**只承诺这个挂载点**。
- 如果未来 skill 路由本身需要 LLM 二次判定（"prompt 里要加载哪些 skill"），把这个 builtin 换成 `Prompt` handler 即可——hook 接入形态不变，只换 handler type。

### 9.3 不在本文范围

- skill 文件格式（frontmatter / markdown / YAML 哪种）
- skill 加载时机的细节（lazy load / 启动期全扫描）
- preload 的 token 预算（README 多大就截）
- 运行期 reload skill / hook（监听 `.defect/skills` 目录变化）

留给 `docs/internal/skills.md` 与 `docs/internal/config.md` 的演进版本细化。**hook 系统只承诺三件事**：
1. `SessionStart` / `UserPromptSubmit` 事件存在且按 §2 形状
2. Handler 可以返回 `append: Vec<ContentBlock>` 把内容注进 system prompt
3. 多个 handler 按声明顺序 pipeline 串行（§3.4）

## 10. 落地节奏

按下列顺序，不另开 ticket：

1. 新建 `crates/agent/src/hooks.rs`（trait + enum + DefaultHookEngine 骨架）
2. `defect-config` 新增 `[hooks]` section 的解析（§5）
3. `DefaultSession::create_session` 接入 SessionStart（§7.1）
4. `TurnRunner` 接入 UserPromptSubmit / PreToolUse / PostToolUse / PostToolUseFailure 四个 await 点
5. Builtin handler：`tracing-audit`、`redact-secrets` 各一个
6. Command handler：`Argv` 形态先；`Shell` 形态做 `ShellKind::Sh`
7. Prompt handler：依赖 `LlmProvider`，落地放最后
8. CLI：`defect hooks list / trust / revoke`
9. e2e：mock provider + scripted hook 配置走 5 件套各一遍

测试策略：复用 [`docs/testing/e2e.md`](../testing/e2e.md) 的 mock 框架；hook engine 单测用 mock handler 验合并规则与 pipeline 顺序。
