# 错误类型分层设计

`defect` 跨多个 crate（`defect-agent` / `defect-llm` / `defect-tools` /
`defect-sandbox` / `defect-acp` / ...），每层都会失败、每层失败的语义都不一样。
本文沉淀**错误类型怎么切层、怎么跨层传递、谁负责给重试建议**这三件事，让后续
新写一个 provider / 一个工具 / 一个 sandbox 后端时，错误形状不再每次重新发明。

> 本文是规范文档，不是实现说明。当前代码里的 [`ProviderError`] / [`ToolError`] /
> [`AgentError`] / [`TurnError`] / [`AcpError`] 已经按本文规则成形，后续工作
> 把缺位的 `StorageError` / `ConfigError` / `HttpError` / `McpError` 按同一
> 套路补上。

## 1. 设计原则

围绕四条铁律：

1. **每个 crate 的公共错误是一个 `#[non_exhaustive] enum`，走 `thiserror`**。
   不要用 `anyhow::Error` / `Box<dyn Error>` 作为 crate 公开的错误类型——它们
   把语义糊掉，调用方匹配时只能写 `format!("{e}")`。`anyhow` 只在 binary
   crate（`defect-cli`）的 `main` 与 examples 里用。
2. **跨层传递走显式转换，不要一路 `From<E1> for E2`**。每层错误是它自己的
   语义；上层接到下层的错误后要决定"这个对我意味着什么"再选 variant。例如
   `ProviderError` 落到 `TurnError` 时，主循环要先决策"这是不是已经把重试
   预算用尽了"，而不是无脑 `From<ProviderError> for TurnError`。
   - 例外：纯透传（无语义变化）的"包装一层"场景，可以用 `#[from]`
     （[`TurnError::Provider`] 即是此例）。下游错误**已经带分类信息**，上层
     只是把它和别的失败合并在一个 enum 里时才允许。
3. **类型擦除的 source 走 [`BoxError`]，不要在公共 API 里写裸
   `Box<dyn std::error::Error + Send + Sync>`**。`BoxError` 是 `defect-agent`
   提供的 newtype（`crates/agent/src/error.rs`），唯一用途是承担"我的某个
   variant 需要透传任意 std error 来源"——既不暴露具体类型、又比 `Box<dyn ...>`
   可读。所有 `#[source] BoxError` 字段都属于这个用法。
4. **重试 / 用户应答建议必须挂在错误自身上，不能让调用方反查**。`ProviderError`
   的 [`RetryHint`] 是范本：错误产生方比调用方更知道"这个错该怎么处理"。
   `ToolError` v0 没有 retry hint 是因为 policy 才是真正的决策方
   （[`tool-trait.md`](./tool-trait.md) §7），不是疏漏。

## 2. 三个切分轴

层级、生命周期、显式 vs 兜底——同一个 error enum 同时在这三个轴上回答问题。

### 2.1 层级（按 crate 切）

```text
                 ┌──────────────────────────────────┐
                 │ defect-cli (anyhow 边界)         │
                 └─────────────────┬────────────────┘
                                   │
        ┌──────────────────────────┴──────────────────────────┐
        │                                                     │
        ▼                                                     ▼
┌──────────────────┐                                ┌──────────────────┐
│ defect-acp       │                                │ defect-agent     │
│  AcpError        │◀──── 投影 / 错误兜底 ─────────│  AgentError      │
│  (transport)     │                                │  TurnError       │
└──────────────────┘                                │  ToolError       │
                                                    │  ProviderError   │
                                                    │   (re-export)    │
                                                    └────────┬─────────┘
                                                             │
                            ┌────────────────┬───────────────┼────────────────┬───────────────┐
                            ▼                ▼               ▼                ▼               ▼
                    ┌──────────────┐ ┌────────────────┐ ┌────────────┐ ┌──────────────┐ ┌──────────────┐
                    │ defect-llm   │ │ defect-tools   │ │ defect-    │ │ defect-mcp   │ │ defect-      │
                    │  ProviderErr │ │  ToolError     │ │ sandbox    │ │  McpError    │ │ storage      │
                    │  (从 agent   │ │  (从 agent     │ │  (v0 不抛  │ │  (P1)        │ │  StorageErr  │
                    │   re-export) │ │   re-export)   │ │   错；§4)  │ │              │ │  (P1)        │
                    └──────────────┘ └────────────────┘ └────────────┘ └──────────────┘ └──────────────┘
```

定位：

- `defect-agent` 是 trait 与 enum 的**主源**：`ProviderError` / `ToolError` /
  `AgentError` / `TurnError` 都定义在这里。`defect-llm` / `defect-tools`
  实现 trait 时直接 use 这两个 error。这样多个 provider / 工具实现共享一份
  错误形状，不必各家维护一份。
- `defect-acp` 只关心**传输层**错误。LLM / 工具 / 主循环的失败由 `AgentEvent`
  与 `TurnError` 表达，桥接层把它们投射成 ACP wire 即可，不需要再定义一个
  "服务端错误"枚举。
- `defect-sandbox` 当前只做策略决策（`SandboxPolicy` 是 infallible 的
  纯函数），所以不需要错误类型。引入 OS 级 sandbox 后再加 `SandboxError`
  ——见 §6.1。
- `defect-config` / `defect-storage` / `defect-mcp` / `defect-cli` HTTP 基础
  设施未就绪，等到对应 P1 任务动手时按本文 §3 的形状补 enum。

### 2.2 生命周期（按"被谁观察到"切）

错误产生后会被三类消费者观察：

| 消费者 | 关心什么 | 期望 |
|--------|---------|------|
| **主循环** | "这次 turn 还能不能继续？" "要不要重试？" "要不要把它送回 LLM？" | 强类型 enum + retry hint |
| **ACP 客户端** | 给用户看的字符串 + 是否触发 stop reason | `Display` + 投影到 `StopReason` / `ToolCallUpdate.status` |
| **运维 / tracing** | 排障线索（request_id / 子源 / span context） | `Debug` + `source()` chain + 错误产生处带的诊断字段 |

诊断字段（如 [`ProviderError::request_id`]）挂在**顶层 struct**而非各 variant，
保证不论 kind 是什么都能拿到。形状参考：

```rust
pub struct ProviderError {
    pub kind: ProviderErrorKind,         // 分类
    pub request_id: Option<String>,      // 诊断
}
```

新 enum 如果有跨 variant 的诊断字段（如 `tool_name` / `path` / `command`），
也按这个模式抽顶层 struct。当前各 enum 没有这种需求时维持纯 enum 即可，**不要**
预先抽 wrapper struct。

### 2.3 显式 vs 兜底

每个 enum 必有的两类 variant：

- **显式分类**：主循环 / 桥接层会 match 它做差异化处理。例如
  [`ProviderErrorKind::AuthExpired`] 触发 refresh、`RateLimit` 触发等待。
- **兜底**：[`Other`](#) / [`Internal`] / `Execution`。**只承载"暂时归不进
  显式分类的来源"**，不应是默认 variant。
  - **见到一类错误反复落入兜底，立刻提取为新 variant，**而不是让兜底慢慢吞掉
    一切。这条规则已经写在 `ProviderErrorKind` 的 doc 里，本文把它升格为
    全局规则。

`#[non_exhaustive]` 是为新增 variant 留口子，不是给"懒得分类"的借口。

## 3. 各层错误类型清单

下表是当前 + 规划中的所有公共错误类型；新建 enum 时对照本表确认形状一致。

| Crate | 类型 | 顶层结构 | 兜底 variant | retry hint | 状态 |
|-------|------|---------|------------|-----------|------|
| `defect-agent` | [`ProviderError`] | `struct { kind, request_id }` | `ProviderErrorKind::Other(BoxError)` | `RetryHint` | 已实现 |
| `defect-agent` | [`ToolError`] | enum (3 variants) | `Execution(BoxError)` | 由 policy 决定 | 已实现 |
| `defect-agent` | [`AgentError`] | enum | `Other(BoxError)` | 不重试 | 已实现 |
| `defect-agent` | [`TurnError`] | enum | `Internal(BoxError)` | 不重试 | 已实现 |
| `defect-acp` | [`AcpError`] | enum (1 variant) | — | 由传输层决定 | 已实现 |
| `defect-sandbox` | — | — | — | — | v0 不需要（§6.1） |
| `defect-config` | `ConfigError` | enum | `Source(BoxError)` | 不重试 | 待 P1 |
| `defect-storage` | `StorageError` | enum | `Io(BoxError)` | 不重试 | 待 P1 |
| `defect-mcp` | `McpError` | enum | `Transport(BoxError)` | RetryHint? | 待 P1 |
| 公共 HTTP 客户端 | `HttpError` | enum | `Other(BoxError)` | RetryHint | 待 P1 |

待补 enum 的最小形状要求：

- `#[non_exhaustive] #[derive(Debug, thiserror::Error)]`
- 至少一个显式 variant + 一个兜底 variant
- 兜底 variant 透传具体 source 时用 `#[source] BoxError`（如果该 crate 已经能
  访问到 `defect_agent::error::BoxError`）；否则 crate 内自己定义同样形状的
  `BoxError` newtype（不要直接暴露 `Box<dyn Error + Send + Sync>`）。
- 在 doc comment 里写清"这一层失败对调用方的语义"——见 [`TurnError`] doc 的
  "划线规则"段落作为范本。

## 4. BoxError 与跨层传递

`crates/agent/src/error.rs::BoxError` 是 newtype，不是 type alias。两个职责：

1. **签名上区分"任意 std error"与"我的错误本身"**——`BoxError` 自己也实现
   `Error`，而 `Box<dyn Error>` 在签名里读起来意图模糊。
2. **预留替换实现的口子**——后续要换成 `anyhow::Error`、加 backtrace 等只改一处。

构造方式只有两个：

```rust
let e = BoxError::new(io::Error::other("..."));        // 从 std error 包装
let e: BoxError = boxed_dyn_err.into();                 // 从已 boxed 形式迁移
```

**没有**为任意 `E: Error` 提供 `From<E>`：Rust 一致性规则下会与
`From<T> for T` 反射 impl 重叠（因为 `BoxError` 自身实现 `Error`）。调用方
显式用 `BoxError::new(...)` 包。这条不是限制，是**强制写出"我在做类型擦除
这件事"**——用了 `BoxError::new` 就等于声明"我下游的具体错误类型不再让上层
matchable，故意如此"。

跨层传递时的两种用法：

```rust
// 用法 A：把第三方错误塞进自家 enum 的 variant
#[error("transport error: {0}")]
Transport(#[source] BoxError),

let err = ProviderError::new(ProviderErrorKind::Transport(BoxError::new(reqwest_err)));

// 用法 B：自家 enum 透明转上层 enum（仅当语义无变化）
#[error(transparent)]
Provider(#[from] ProviderError),
```

**禁止**用法 C：在公共 enum 里直接写 `Box<dyn Error + Send + Sync>`
（=「跳过 BoxError 这层抽象」）。

## 5. 重试与建议的归属

谁产生错误，谁给建议：

| 错误来源 | 给建议的 trait/字段 | 谁消费 |
|---------|-------------------|-------|
| `ProviderError` | [`ProviderError::retry_hint() -> RetryHint`] | 主循环（决定是否重发请求） |
| `ToolError` | 由 [`SandboxPolicy`](./sandbox-policy.md) 决定 | 主循环（同 retry 预算） |
| `TurnError` | 不可重试（已是终态） | ACP 桥接（投成 JSON-RPC error） |
| `AgentError` | 不可重试 | CLI 启动阶段（直接 fail-fast） |
| 待补层 | 各自定义 | — |

`RetryHint` 是 `ProviderError` 自带的 enum（`No` / `Immediate` / `After(Duration)`
/ `Backoff` / `AfterAction(RetryAction)`），主循环根据 hint 决定动作；**不**让
主循环自己 match `ProviderErrorKind` 判断哪些 variant 可重试。

为什么 hint 也要做成强类型枚举？

- 主循环匹配 `RetryHint::After(d)` 比 `if matches!(kind, RateLimit { retry_after: Some(d) })`
  可读
- 后续要加"切模型重试" / "降级到 streaming=false 重试"等动作时，扩 `RetryAction`
  enum 即可，不动错误类型本身

`ToolError` 不挂 retry hint 的原因（再强调一遍）：工具的可重试性**依调用方
而非错误本身**。同一个 `Execution(io::Error::PermissionDenied)`：

- `fs.read` 不可重试（路径权限不会自己变）
- `bash` 可能可以（用户可能临时改了 chmod）

让 policy 而不是错误回答这个问题，符合 [`tool-trait.md`](./tool-trait.md) §4 的
"工具自己只回答想做什么"原则。

## 6. 演进口子

### 6.1 OS 级 sandbox 引入

[`sandbox-policy.md`](./sandbox-policy.md) §8 已经预留 `ToolSandbox` trait。
真正接 landlock / seatbelt / seccomp 时新增：

```rust
// crates/sandbox/src/error.rs（届时新建）

#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    /// OS 拒绝了我们的隔离配置（landlock 内核版本不支持等）。
    #[error("sandbox setup unsupported: {hint}")]
    Unsupported { hint: String },

    /// 隔离生效后，工具命中规则被拒。区分于工具自己的 `ToolError::Execution`：
    /// 这个表示**沙箱阻止了它**，不是工具本身失败。
    #[error("sandbox denied: {hint}")]
    Denied { hint: String },

    #[error("sandbox internal error: {0}")]
    Internal(#[source] BoxError),
}
```

`ToolContext::sandbox` 调用失败时，`Tool::execute` 把它包成
`ToolError::Execution(BoxError::new(sandbox_err))` 上抛。把 sandbox 故障
**当作工具失败**对待，主循环就不需要为 sandbox 单开一条事件路径。

### 6.2 storage / config / mcp / http

按 §3 表里给出的最小形状直接补。每个的 doc comment 必须写：

- 这一层失败的"语义后果"（fail-fast 还是降级运行？）
- 兜底 variant 收集了哪些 case，以及"哪种 case 应该提取出去"的标准

### 6.3 错误的 `Display` 与 i18n

v0 错误 `Display` 是英文短语 + 关键字段（`{request_id}` / `{model}` 等）。
**不**做 i18n、**不**让错误带模板。需要展示给最终用户的本地化字符串由
ACP 客户端 / CLI 自己根据 enum variant 自己映射。

## 7. CLI / examples 边界：anyhow 的合理用法

`defect-cli` 的 `main` 与 `examples/` 直接用 `anyhow::Result` / `anyhow::anyhow!`
是合理的：

- main 不被任何代码调用，错误最终落到 `eprintln!` 或 `tracing::error!`
- 启动失败的处理只有"打印 + 退出码"一种，不需要 match
- `anyhow!` 的 chain 形式比手搓 `Display` 链更清晰

但 **defect-cli 暴露给 acp::serve 的 trait 方法、provider 实现、tool 实现都
不允许返回 `anyhow::Error`**——它们是 lib 层。本文 §1 的第 1 条铁律就是
管这事的。

## 7.1 ACP wire 投影：message 字段必须有信息量

`AcpError::into_wire_error` 把内部错误投到 wire 的两条铁律：

1. **wire `message` 字段填内层 `Display`**——不要直接用
   `Wire::internal_error()`（其 message 永远是字面量 `"Internal error"`）。
   客户端 UI（acpx 等）默认只渲染 `message`，把诊断信息埋在 `data` 里
   会让用户只看见 `RUNTIME: Internal error` 这种无意义占位。
2. **`code` 选择避开 ACP 自定义码段的语义陷阱**：acpx 把 `-32001` /
   `-32002` 视为 "resource not found / NO_SESSION"，把 `-32000` 视为
   `auth_required`。Provider error / Internal error 即使想给个独立
   code 也不要落到这些值上——会让客户端误判成会话丢失或要登录。
   v0 简单粗暴：除 `TurnInProgress` 走 `InvalidRequest` 外，其它都
   `InternalError`，靠 message 文本携带信息让客户端的 text-rule
   命中（"rate limit" / "model not found" 等）。

## 8. 与 tracing 的衔接

错误形状要让 tracing instrumentation 不用"再造一份字段"：

- 顶层 struct 上的诊断字段（`request_id` 等）将由 `tracing.md` 规范的 span
  字段直接读取（详见 [`docs/outbound/tracing.md`](../outbound/tracing.md)）。
- 错误的 `Display` 进事件 message，`source()` chain 进 `error.cause` 字段。
- Rust 侧不要在 error 构造点 `tracing::error!` 自我打印——日志由调用方
  在边界（acp 桥接 / CLI main / 重试器）统一发，否则同一个错误会被多次打印。

## 9. 落地节奏

本文是规范，不是 PR 切单。当前代码已经**符合规范**，落地工作只在新写代码时
"按本文做"即可。具体未来动作：

1. 补 P1 待写 enum（`ConfigError` / `StorageError` / `McpError` / `HttpError`）
   时按 §3 形状一次写到位，对照本文 review。
2. 引入 OS sandbox 时新增 `SandboxError`（§6.1）。
3. 任何对现有 enum 的"加 variant" / "重排兜底"动作，都视为 minor 改动，
   不需要单独立项；只在 PR 描述里引一下本文规范。

[`ProviderError`]: ../../crates/agent/src/llm/error.rs
[`ProviderError::request_id`]: ../../crates/agent/src/llm/error.rs
[`ProviderError::retry_hint() -> RetryHint`]: ../../crates/agent/src/llm/error.rs
[`ProviderErrorKind::AuthExpired`]: ../../crates/agent/src/llm/error.rs
[`ProviderErrorKind::Other(BoxError)`]: ../../crates/agent/src/llm/error.rs
[`ToolError`]: ../../crates/agent/src/tool.rs
[`AgentError`]: ../../crates/agent/src/session.rs
[`TurnError`]: ../../crates/agent/src/session.rs
[`AcpError`]: ../../crates/acp/src/serve.rs
[`BoxError`]: ../../crates/agent/src/error.rs
[`RetryHint`]: ../../crates/agent/src/llm/error.rs
[`Internal`]: ../../crates/agent/src/session.rs
