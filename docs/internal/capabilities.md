# Capabilities 设计

> Capability 是「这个 session 拥有什么能力」的抽象。它不是工具——工具回答「**怎么执行**一次具体动作」，capability 回答「这个能力**从哪儿来**」。

`search` 是当前唯一落地的 capability；后续 `image_generation` / `code_execution` / `computer_use` 等同形态能力沿用本设计。本文沉淀 capability 的层级、与 [`Tool`] / [`LlmProvider`] / 配置的边界，以及 P1 的运行时装配规则。

[`Tool`]: ./tool-trait.md
[`LlmProvider`]: ./llm-trait.md

设计原则：

1. **能力是能力，工具是工具**——一次能力可以由 provider-hosted 实现，也可以由本地 [`Tool`] 实现；同一份能力**在任一 turn 内只有一个来源**。
2. **session 启动期一次性裁决**——`(provider, mode)` 在 session 生命周期内不变，turn loop 不重判。
3. **provider-hosted 能力不实现 [`Tool`] trait**——hosted search 由 provider 自己执行、agent 无法逐次拦截，强行套 [`Tool`] 会扭曲 stop reason / 审批 / 事件流语义。

---

## 1. 抽象层级

```
┌───────────────────────────────────────────────────────────┐
│  capability 层                                            │
│  ─ 回答："这个 session 有 search 能力吗？来源是什么？"     │
│  ─ 配置入口：[capabilities.search]                         │
│  ─ 类型：SearchCapabilityMode { Delegate, Local, Disabled }│
└───────────────────────────────────────────────────────────┘
                ↓ session 启动期裁决
┌───────────────────────────────────────────────────────────┐
│  装配层                                                   │
│  ─ ResolvedSessionCapabilities                            │
│  ─ ┌─ hosted: HostedCapabilities { search: bool }         │
│  └─ register_local_search: bool                           │
└───────────────────────────────────────────────────────────┘
                ↓
┌──────────────────────┐  ┌─────────────────────────────────┐
│  provider 层         │  │  工具层                          │
│  ─ 暴露 hosted       │  │  ─ 内置 search Tool（mode=Local）│
│    search 给 wire    │  │  ─ fetch / fs / bash / ...      │
└──────────────────────┘  └─────────────────────────────────┘
```

| 层 | 职责 | 不负责 |
|---|---|---|
| capability | 决定能力是否存在、来源是什么 | 怎么执行 |
| provider | 实现状态自报家门 + hosted lower 到 wire | 产品语义 / 来源选择 |
| 工具 | 本地真正执行的能力（含 mode=Local 时的 search） | 跨 session 持久状态 |

---

## 2. 前置假设：session 绑定单一 provider

本文及之后的能力来源、装配、shadow 规则都建立在一个前置假设上：

> **一个 session 在生命周期内只绑定一个 provider，不支持会话内切 provider**。

这与当前 codebase 一致：[`Session`](../../crates/agent/src/session.rs) 只暴露 `set_model`（同 provider 内换模型），不存在 `set_provider`。每轮 turn 的 provider 等于 session 启动时绑定的 provider。

因此本文中「当前轮的能力来源」与「当前 session 的能力来源」**指同一件事**。后续若要支持 session 内切 provider，需要先单独立项扩 `Session` 接口（重新协商 hosted capability、降级 history `ProviderActivity` 等），与本设计不冲突。

---

## 3. `SearchCapabilityMode`：三态

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchCapabilityMode {
    /// 委托给 provider-hosted search。provider 不支持时 session 启动失败。
    Delegate,
    /// 用 defect 本地 `search` tool。provider 是否支持 hosted 不影响。
    #[default]
    Local,
    /// 既不暴露 hosted，也不暴露本地 `search` tool。
    Disabled,
}
```

TOML 串：`"delegate"` / `"local"` / `"disabled"`。

语义对照：

| 模式 | hosted search | 本地 `search` tool | MCP 同名 `search` |
|---|---|---|---|
| `Delegate` | ✅ 暴露 | ❌ 不注册 | 走 `mcp.<server>.search`（§6） |
| `Local` | ❌ 不暴露 | ✅ 注册（裸名 `search`） | 走 `mcp.<server>.search` |
| `Disabled` | ❌ 不暴露 | ❌ 不注册 | 走 `mcp.<server>.search` |

为什么不做 `prefer_provider` / `require_provider` 这类 fallback 模式：

1. 「能力来源选择」与「失败回退策略」混在一起会让配置语义可预测性变差
2. 静默回退会改变行为可观测性
3. 三态先收敛实现与测试矩阵；P1 不做隐式回退

---

## 4. `HostedCapabilities`：provider 自报家门

[`LlmProvider`] trait 新增独立方法：

```rust
pub trait LlmProvider {
    // ... 现有方法 ...

    /// provider 自报家门：当前实现支持哪些 hosted capability。
    fn hosted_capabilities(&self) -> HostedCapabilities {
        HostedCapabilities::default()  // 默认全 false
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HostedCapabilities {
    pub search: bool,
}

impl HostedCapabilities {
    /// 跨 crate 构造入口（`#[non_exhaustive]` 后不能 struct literal）。
    pub const fn with_search(search: bool) -> Self { Self { search } }
}
```

与 [`Capabilities`](./llm-trait.md) 区分：

| 类型 | 含义 | 来源 |
|---|---|---|
| `Capabilities` | 模型属性（thinking / vision / tool_calls / ...） | 模型文档 |
| `HostedCapabilities` | adapter 实现状态（能否在 wire 上声明 hosted search） | adapter 自己声明 |

各 provider 当前实装：

| provider | `hosted_capabilities().search` | 说明 |
|---|---|---|
| Anthropic | `false`（P1） | 待接 `web_search_*` hosted tool |
| OpenAI | `false`（P1） | 待接 Responses API `web_search` |
| DeepSeek | `false` | 不支持 hosted search |
| Echo | `false` | 测试 stub |

> **P1 实装现状**：trait 接缝齐全，但所有 provider 都返回 `false`——hosted wire 编解码留待后续单独立项。详见 [P1 实装状态](#10-p1-实装状态)。

### 4.1 hosted tool 版本选择

Anthropic / OpenAI 的 hosted tool 自身有版本（`web_search_20250305` / `web_search_20260209` 等）。P1 处理：

- **hosted tool 版本由 adapter 内部硬编码取最新**
- agent 层不感知版本——`HostedCapabilities { search: bool }` 只回答 yes / no
- 暴露版本字段会让 agent 背上 provider-specific 知识，违反层级边界

未来若需要多版本切换，应当是 adapter 内部按 model id / capabilities 决定，不构成 trait breaking。

---

## 5. session 启动期裁决

### 5.1 装配时机

裁决在 [`AgentCore::create_session`](../../crates/agent/src/session/default.rs) 一次性完成：

```rust
let resolved = ResolvedSessionCapabilities::resolve(
    self.capabilities,
    self.provider.hosted_capabilities(),
    &self.provider.info().vendor,
)?;
```

```rust
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResolvedSessionCapabilities {
    /// 最终决定本 session 是否走 hosted search。
    /// `Delegate × supported` → `true`；其余情况 → `false`。
    pub hosted: HostedCapabilities,
    /// 最终决定本 session 是否要注册本地 `search` tool。
    /// 仅 `Local` 时为 `true`。
    pub register_local_search: bool,
}
```

session 启动期裁决一次的好处：

- 配置不满足能在最早时机报错，不让 turn loop 跑到一半才 fail
- 每轮 turn 装配 [`CompletionRequest`](./llm-trait.md) 不重新查 `hosted_capabilities()`，省一次开销
- session 内任意 turn 看到的 search 来源都一致，不会出现「上一轮 local 这一轮 hosted」

### 5.2 裁决表

| `mode` | provider 支持 hosted search | 装配结果 |
|---|---|---|
| `Delegate` | 支持 | `hosted.search = true`，本地 `search` tool 不注册 |
| `Delegate` | 不支持 | session 启动失败，返回 [`SessionInitError::CapabilityUnsatisfied`](#53-启动失败) |
| `Local` | 任意 | `register_local_search = true`，hosted 不暴露 |
| `Disabled` | 任意 | 两边都不暴露 |

### 5.3 启动失败

```rust
#[non_exhaustive]
#[derive(Debug)]
pub enum SessionInitError {
    CapabilityUnsatisfied {
        capability: &'static str,
        provider: String,
    },
}
```

`Delegate` 是用户的显式选择「我要 hosted」。provider 不支持时静默 fallback 到 local 会改变行为可观测性，静默不暴露 search 会让模型幻觉调用一个不存在的能力——两者都比 fail-fast 差。

错误信息内嵌 actionable hint：

```text
search capability is unsatisfied: provider `deepseek` does not support hosted search.

To fix this, choose one of:
  1. Override per-provider in your config:
       [providers.deepseek.capabilities.search]
       mode = "local"
  2. Change global default to `local` and keep hosted only for providers that support it:
       [capabilities.search]
       mode = "local"
       [providers.<hosted-supported>.capabilities.search]
       mode = "delegate"
```

hint 文本由 [`SessionInitError`](../../crates/agent/src/session.rs) 的 `Display` 实现渲染——结构化字段只有 `capability` / `provider`，避免字段污染。

---

## 6. shadow 规则与 MCP 命名空间

### 6.1 能力来源唯一

任一 turn 内 `search` 只有一个来源：

- `Delegate` 时：不向模型暴露本地 `search` tool
- `Local` 时：不暴露 provider-hosted search
- `Disabled` 时：两边都不暴露

否则模型会在两种能力间摇摆、transcript 语义会分裂、审批与可观测性混乱。

### 6.2 MCP 工具一律命名空间化

**所有** MCP 工具在本地 [`ToolRegistry`](./session.md) 里一律以 `mcp.<server>.<name>` 注册，不区分名字、不区分 capability mode、不区分本地工具 enabled。

```rust
// crates/mcp/src/lib.rs
#[must_use]
pub fn registered_mcp_tool_name(server: &str, upstream_name: &str) -> String {
    format!("mcp.{server}.{upstream_name}")
}
```

为什么不只在撞名时改名：

1. 注册名一眼能看出「这是 MCP 工具，来自哪个 server」——provenance 在工具表上是显式的
2. 后续给 defect 新增任何内置工具（`fetch` / `search` / 未来的 `grep` / `memory` 等）都不会触发 MCP 旁路或静默改名，避免「不同 session 下同一个 MCP 工具有两种名字」
3. 「能力来源唯一」原则不再依赖 MCP 命名空间的运行时判断——只有内置 / hosted 才能占用裸名，MCP 永远走前缀

「上游 wire 名」与「本地注册名」的边界：

| 名字 | 用途 | 取值 |
|---|---|---|
| 注册名 | agent 的 [`ToolRegistry`] 与暴露给 LLM 的 tools schema | `mcp.<server>.<tool>` |
| Wire 名 | agent 调 `call_tool` 发回 MCP server 时使用 | `<tool>`（原始 MCP server 暴露的名字） |

server 不知道也不在乎本地的 `mcp.<server>.` 前缀。

装配时仍然发 `ConfigWarning::McpToolRenamed { server, original, renamed }`，给用户一条明确告知：「这是你 MCP server `<server>` 的 `<original>` 工具，在 defect 里以 `<renamed>` 形式调用」。

---

## 7. transcript 与事件流

### 7.1 不追求统一执行模型

provider-hosted search 与 local search 的底层执行路径不同，不需要强行抹平：

- provider-hosted search 走 provider response item 路径
- local search 走本地 [`Tool`] 路径

### 7.2 上层语义统一为 ACP `ToolCallUpdate`

虽然底层分轨，hosted search 由 provider adapter 在解析流时翻译成 ACP `ToolCallUpdate { kind: ToolKind::Search, ... }`，复用 ACP 已有的 `Search` 工具种类。

| 字段 | 来源 |
|---|---|
| `tool_call_id` | provider 给的 hosted call id 直接采用（不复用本地 tool registry 的 id 分配器） |
| `kind` | `ToolKind::Search` |
| `content` | provider 返回的 sources / summary |

这样：

- ACP 客户端不需要认识「hosted」概念，所有 search 都长一样
- agent 主循环也不需要为 hosted 分配 tool_call_id
- transcript / UI 上统一显示 "Searching: <query>" → sources 列表

agent 主循环在收到 hosted search 的 ACP 推送后，**不**像本地工具那样需要等 `tool_call_update.status = Completed` 才进入下一轮——provider stream 自己会带回 result，turn loop 不阻塞在 hosted search 上。

### 7.3 `MessageContent::ProviderActivity`

hosted search 的调用与结果**进入 history**（否则跨轮不可见，模型会重复发起同样的搜索），但用专属 `MessageContent` variant，不复用 `ToolUse / ToolResult`：

```rust
#[non_exhaustive]
pub enum MessageContent {
    // ... 现有 variant ...

    /// provider-hosted 能力产生的活动。agent 不解释 payload；codec
    /// 在重发同 provider 时透传，切 provider 时由 codec 决定如何降级。
    ProviderActivity {
        provider_id: String,
        kind: ProviderActivityKind,
        /// provider-native payload（黑盒）。Anthropic web_search 整段
        /// `server_tool_use` + `web_search_tool_result` 块原样塞这里；
        /// OpenAI Responses 的 `web_search_call` 同样原样塞这里。
        payload: serde_json::Value,
    },
}

#[non_exhaustive]
pub enum ProviderActivityKind {
    Search,
}
```

设计决策：

| 决策 | 理由 |
|---|---|
| **payload 黑盒** | agent 主循环不读它，只在 history 里搬运。和当前 `Thinking { signature }` 的 anthropic-only payload 同款套路，但语义更宽 |
| **`provider_id` 进字段** | 切 provider 时 codec 能根据 `provider_id != self.id` 判断这条 activity 是「他人产生的」，决定丢弃还是降级为纯文本 summary（详见 §7.4） |
| **不持久化到磁盘** | `#[serde(skip)]` 或在持久化 codec 里显式丢弃。session resume 后如果模型再次触发 hosted search，会重新发起一次新调用，不依赖旧 payload |
| **不上抬到 ACP transcript** | 前端看到的是 §7.2 描述的 `ToolCallUpdate { kind: Search }`，看不到 `ProviderActivity`。后者纯粹是 history 内部状态 |

### 7.4 跨 provider 切换时的降级

session 切换 provider（例如用户中途改 model）时，history 里已有的 `ProviderActivity { provider_id: "anthropic", ... }` 不能原样喂给 OpenAI——payload schema 完全不同，wire 会报错。

降级规则：

1. 切 provider 后首次发请求前，agent codec 扫一遍 history
2. 所有 `provider_id != current_provider` 的 `ProviderActivity` 转成 `MessageContent::Text { text: synthesize_summary(activity) }`
3. `synthesize_summary` 由 provider adapter 自己实现：把 hosted result 的 sources 列表渲染成 markdown 文本，丢失 citation 块的结构化数据但保住语义

这条降级是**有损**的（citation 丢失），但避免了双轨 history 维护，符合「history 真相只有一份」的现状。session 内不切 provider 时（绝大多数场景）payload 完整保留。

> **当前状态**：`Session` trait 没有 `set_provider`，因此本节降级是为后续支持「会话内切 provider」预留的设计契约，P1 不会触发。

---

## 8. 与 [`Tool`] trait 的边界

### 8.1 hosted 不实现 [`Tool`] trait

```rust
// 不会出现：
impl Tool for HostedSearchAdapter { ... }    // ✗ 拒绝
```

否则会导致：

- 审批时序假装一致（hosted 不能在 call site 拦截）
- provider stop reason 被错误建模成本地 tool pause/resume
- 本地 tool 测试矩阵被 hosted 路径污染

也不在 P1 引入 `HostedTool` sibling trait——当前更重要的是先把架构边界拉清，而不是急着做 trait 并列抽象。

### 8.2 本地 `search` tool 仍然是普通 [`Tool`]

当 `mode = Local` 时：

- 注册一个本地 `search` tool
- 它就是普通 [`Tool`]：可以走审批、执行、事件流、错误恢复
- schema / 实现参数由 `[tools.search]` 段控制（详见 [`config.md`](./config.md)）

[`Tool`] trait 当前承载：`fetch` / `fs` / `bash` / mode=Local 时的 `search`。

---

## 9. `CompletionRequest` 上的启用信号

agent 装配每轮请求时把 session 启动期决定的 hosted 启用集合塞进请求：

```rust
pub struct CompletionRequest {
    // ... 现有字段 ...

    /// 本轮允许 provider 自行使用的 hosted capability 集合。
    /// provider adapter 据此把 hosted tool definition lower 到
    /// provider-native 请求格式。
    pub hosted_capabilities: HostedCapabilities,
}
```

provider adapter 责任：

- 看到 `hosted_capabilities.search = true` 时，在 wire 上声明 hosted search tool（具体名字 / 版本由 adapter 决定）
- 看到 `false` 时，wire 上不声明 hosted search——即便 provider 服务端支持，也不暴露给模型
- agent 不感知具体 wire 字段名

这样 hosted capability 的启用控制是**单向、显式**的：agent → provider；provider 不能擅自启用 hosted。

---

## 10. P1 实装状态

| 部件 | 状态 |
|---|---|
| [`SearchCapabilityMode`] / [`SessionCapabilitiesConfig`] | ✅ |
| [`ResolvedSessionCapabilities::resolve`] | ✅ |
| [`SessionInitError::CapabilityUnsatisfied`] + actionable hint | ✅ |
| [`HostedCapabilities`] + [`LlmProvider::hosted_capabilities`] 默认实现 | ✅ |
| [`CompletionRequest::hosted_capabilities`] 字段 | ✅ |
| [`MessageContent::ProviderActivity`] variant | ✅（`#[serde(skip)]`） |
| MCP 全量命名空间化 + [`ConfigWarning::McpToolRenamed`] | ✅ |
| Anthropic / OpenAI hosted search wire 编解码 | ❌（adapter 仍返回 `search: false`） |
| 本地 `search` tool 实现 | ❌（mode=Local 时 `[tools.search]` 已可解析，但 tool 本身未实装） |
| 切 provider 时 history 降级 | ❌（`Session` 尚无 `set_provider`） |

[`SearchCapabilityMode`]: ../../crates/agent/src/session/capabilities.rs
[`SessionCapabilitiesConfig`]: ../../crates/agent/src/session/capabilities.rs
[`ResolvedSessionCapabilities::resolve`]: ../../crates/agent/src/session/capabilities.rs
[`SessionInitError::CapabilityUnsatisfied`]: ../../crates/agent/src/session.rs
[`HostedCapabilities`]: ../../crates/agent/src/llm/capability.rs
[`LlmProvider::hosted_capabilities`]: ../../crates/agent/src/llm/provider.rs
[`CompletionRequest::hosted_capabilities`]: ../../crates/agent/src/llm/request.rs
[`MessageContent::ProviderActivity`]: ../../crates/agent/src/llm/request.rs
[`ConfigWarning::McpToolRenamed`]: ../../crates/config/src/types.rs
[`ToolRegistry`]: ./session.md

---

## 11. 测试矩阵

session 启动期裁决（[`crates/agent/src/session/capabilities.rs`](../../crates/agent/src/session/capabilities.rs) 单测 + [`crates/agent/tests/session_capabilities.rs`](../../crates/agent/tests/session_capabilities.rs) 集成测试）：

- `Delegate × supported` → `hosted.search = true`，`register_local_search = false`
- `Delegate × unsupported` → `SessionInitError::CapabilityUnsatisfied`
- `Local × *` → `register_local_search = true`，`hosted.search = false`
- `Disabled × *` → 两边都关
- 错误消息含 `[providers.<p>.capabilities.search]` 与 `[capabilities.search]` 两条 hint

MCP 命名空间（[`crates/mcp/src/test.rs`](../../crates/mcp/src/test.rs) + [`crates/cli/tests/mcp_stdio_smoke.rs`](../../crates/cli/tests/mcp_stdio_smoke.rs)）：

- `registered_mcp_tool_name` 三元组测试（普通工具 / `search` 撞名 / `fetch` 撞名）
- stdio / config-stdio / sse 三个端到端 smoke 用 `mcp.<server>.<tool>` 注册名调通

配置侧（[`crates/config/src/loader/test.rs`](../../crates/config/src/loader/test.rs)）：

- `[capabilities.search]` 三态解析
- provider 覆写 merge fallback
- 全局 `delegate` + provider 不支持 → 启动失败由 agent 层接住
- `[tools.search]` 在 mode != Local 时发 [`InactiveSection`](./config.md) warning

---

## 12. 拒绝的替代方案

### 12.1 把 hosted search 继续伪装成 [`Tool`]

拒绝原因：

- agent 无法真实介入执行
- 审批时序不成立
- stop reason 与事件流语义会被扭曲

### 12.2 让 hosted/local search 同时暴露给模型

拒绝原因：

- 能力来源竞争
- 行为不稳定
- 可观测性变差

### 12.3 把 `fetch` 也提升为 capability

拒绝原因：

- 当前没有足够收益
- 会把一个天然适合作为本地工具的能力过度抽象——详见 [`tools-fetch.md`](./tools-fetch.md)

---

## 13. 决议

1. `search` 是 capability，三态：`Delegate` / `Local` / `Disabled`
2. provider-hosted 与 local tool 分轨，**不**共用执行路径与 [`Tool`] trait
3. session 启动期一次性裁决，不满足直接 fail
4. MCP 工具一律 `mcp.<server>.<name>` 注册
5. hosted activity 进 history 但不持久化、不上抬 ACP transcript
6. fetch 走 [`Tool`] 而不是 capability——参见 [`tools-fetch.md`](./tools-fetch.md)
