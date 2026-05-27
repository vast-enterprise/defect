# `search` 能力与 `fetch` 工具重构提案

## 1. 结论

这份文档直接给出新的建模结论：

1. `search` 是**能力**，不是先验本地工具
2. `fetch` 是 **defect 原生工具**
3. provider-hosted 能力与本地工具**彻底分轨**
4. agent 只负责决定“暴露哪一种能力来源”，不把 provider-hosted search 伪装成本地 `Tool`

这意味着旧思路需要放弃：

- 不再尝试把 hosted `search` 塞进 `Tool` trait
- 不再尝试让 hosted/local `search` 共用一条执行时序
- 不再围绕 hosted `fetch` 设计主架构

## 2. 为什么要推倒重来

之前版本最大的问题不是字段不够，而是抽象层级错了。

provider-hosted search 有一个无法回避的事实：

> 它不是 agent 可以逐次拦截、逐次执行、逐次回填结果的本地工具调用。

一旦 provider 原生支持 `web_search`：

- agent 只能决定要不要把这个能力暴露给模型
- 不能在 call site 像本地 `Tool::execute()` 那样介入
- 也不能把它自然塞进现有 `ToolEvent` / `RequestPermission` 时序

所以 `search` 不应该先被建模成“一个工具”，而应该先被建模成“模型是否拥有某种外部检索能力”。

## 3. 新分层

### 3.1 能力层

能力层回答：

- 模型在当前 session 里能不能搜索外部信息

（前置假设见 §4.0：一个 session 绑定单一 provider，因此 session 级和 turn 级在 P1 内等价。）

当前只讨论：

- `search`

后续同类能力还有可能包括：

- `image_generation`
- `code_execution`
- `computer_use`

这些能力都可能来自 provider-hosted，而不是本地 `Tool`。

### 3.2 工具层

工具层只放 defect 本地真正执行的东西：

- `fs`
- `bash`
- `fetch`
- 以及 provider 不支持时用于补位的本地 `search` tool

工具层的特点是：

- agent 可以审批
- agent 可以执行
- agent 可以流式发进度
- agent 可以决定失败/重试语义

### 3.3 provider 层

provider 层只负责两件事：

1. 报告自己是否支持某个 hosted capability
2. 把被允许暴露的 hosted capability lower 到 provider-native 请求格式

provider 层**不拥有产品语义**，但它拥有 hosted capability 的 wire 编解码职责。

## 4. `search` 的新建模

### 4.0 前置假设：session 绑定单一 provider

本节及之后的「能力来源」「装配」「shadow」全部建立在一个前置假设上：

> **一个 session 在生命周期内只绑定一个 provider，不支持会话内切 provider**。

这与当前 codebase 状态一致：[`Session`](../../crates/agent/src/session.rs) 只暴露 `set_model`（同 provider 内换模型），不存在 `set_provider`。每轮 turn 的 provider 等于 session 启动时绑定的 provider，模型可以在同 provider 的候选 model 间切换。

因此本提案中的「当前轮的能力来源」与「当前 session 的能力来源」**指同一件事**——以下章节默认两种说法等价；后续若要支持 session 内切 provider，需要先单独立项扩 `Session` 接口（重新协商 hosted capability、降级 history `ProviderActivity` 等），与本提案不冲突。

`search` 不再默认等于一个叫 `search` 的本地工具。

`search` 的真实问题是：

> 这个 session 里，模型的搜索能力来自哪里？

答案只有三类：

1. provider-hosted
2. local tool
3. disabled

### 4.2 三态模型

建议直接写成：

```rust
#[non_exhaustive]
pub enum SearchCapabilityMode {
    Delegate,
    Local,
    Disabled,
}
```

TOML 串：`"delegate"` / `"local"` / `"disabled"`。

语义：

- `Delegate`
  - 当前 provider 支持 hosted search 时，暴露 provider-native search
  - 同时不向模型暴露本地 `search` tool（也 shadow MCP 同名 `search`，见 §7.3）
- `Local`
  - 不暴露 provider-hosted search
  - 向模型暴露 defect 本地 `search` tool
- `Disabled`
  - 两边都不暴露——provider-hosted、defect 内置本地 `search` tool 都不注册
  - **MCP 同名 `search` 也被重命名为 `mcp.<server>.search`**（见 §7.3），保证 session 内不存在裸 `search` 工具
  - 语义是「这个 session 不提供 search 这个名字下的能力」，不是「defect 自己不提供，但允许 MCP 占名」
  - 不影响其他能力或工具——例如本地 `fetch` 仍按 `tools.fetch` 配置正常注册

### 4.3 为什么不用 fallback 型模式

例如：

- `prefer_provider`
- `require_provider`

这些模式不是不能做，但 P1 不建议先上。

原因：

1. 它把“能力来源选择”和“失败回退策略”混在一起
2. provider 不支持时是否自动退回本地，会让配置可预期性变差
3. 先做三态更容易让实现和测试收敛

P1 建议明确选择来源，不做隐式回退。

## 5. `fetch` 的新建模

### 5.1 `fetch` 是 defect 原生工具

`fetch` 的目标是：

- 读取指定 URL
- 控制格式、超时、大小、重定向、解析

这本质上更像：

- `fs.read` 的网络版

而不是 provider-hosted capability。

所以 `fetch` 在 P1/P2 应直接建模为 defect 本地工具。

### 5.2 为什么不围绕 hosted fetch 建模

1. provider 支持不对称
2. 行为不可控
3. 审批模型更差
4. 和 `Tool` trait 的契合度远低于本地 HTTP fetch

因此即便 Anthropic 支持 `web_fetch`，也不应该让它定义 defect 的 `fetch` 主架构。

### 5.3 后续是否完全禁止 hosted fetch

不需要在文档里永久禁止。

但结论应写清：

> hosted fetch 不是当前设计中心，也不是 P1/P2 的必需项。

如果以后要支持，应单独新增设计，而不是反向污染当前工具抽象。

## 6. 运行时装配规则

这是这次重写最重要的实装结论。所有装配规则都建立在 §4.0 的「session 绑定单一 provider」假设上。

### 6.1 `search`

装配在 **session 启动时一次性完成**，而不是每轮 turn 单独决策。这条结论建立在 §4.0 的前置假设上——session 绑定单一 provider，turn 之间不会发生 provider 切换，因此 `(provider, mode)` 对 session 整个生命周期都不变，启动期裁决一次后续 turn 直接复用。

这样做的好处：

- 配置不满足能在最早时机报错，避免 turn loop 跑到一半才发现 provider 不支持
- 每轮 turn 装配 `CompletionRequest` 不重新查询 `hosted_capabilities()`，省一次开销
- session 内任意 turn 看到的 search 来源都一致，不会出现「上一轮是 local 这一轮是 hosted」的诡异行为

session 启动流程：

1. 取本次 session 绑定的 provider id
2. 读 `capabilities.search.mode`，应用 `providers.<provider>.capabilities.search.mode` 覆写得到本 session 的最终 mode
3. 调 `provider.hosted_capabilities()`（见 §10）拿到 provider 自报家门
4. 按 mode 与 provider 支持情况裁决：

| mode | provider 支持 hosted search | 装配结果 |
|------|----------------------------|----------|
| `Delegate` | 支持 | 在 session 上记 `hosted_search = true`，本地 `search` tool **不注册** |
| `Delegate` | 不支持 | **session 启动失败**，返回 `SessionInitError::CapabilityUnsatisfied { capability: "search", provider }` |
| `Local` | 任意 | `hosted_search = false`，本地 `search` tool 注册 |
| `Disabled` | 任意 | `hosted_search = false`，本地 `search` tool 不注册 |

session 启动失败的原因：`Delegate` 是用户的显式选择「我要 hosted」；provider 不支持时静默 fallback 到 local 会改变行为可观测性，静默不暴露 search 会让模型幻觉调用一个不存在的能力——两者都比 fail-fast 差。

每轮 turn 装配 `CompletionRequest` 时，只是把 session 上记好的 `hosted_search` 标记和 tool registry 透传给 provider，不再重新决策。

#### 6.1.1 常见配置 pitfall 与错误消息

最容易踩的坑是「全局 `delegate` + 切到不支持 hosted 的 provider」：

```toml
[capabilities.search]
mode = "delegate"
```

用户在 Anthropic 下能用，切到 DeepSeek 启动 session 时直接报 `CapabilityUnsatisfied`，体感是「为什么换个 provider 就崩了」。

为了让这个错误自解释，`SessionInitError::CapabilityUnsatisfied` 必须带 actionable hint，至少包含两条可选修复路径：

```text
search capability is unsatisfied: provider `deepseek` does not support hosted search.

To fix this, choose one of:
  1. Override per-provider in your config:
       [providers.deepseek.capabilities.search]
       mode = "local"
  2. Change global default to `local` and keep hosted only for providers that support it:
       [capabilities.search]
       mode = "local"
       [providers.anthropic.capabilities.search]
       mode = "delegate"
       [providers.openai.capabilities.search]
       mode = "delegate"
```

实现上 `SessionInitError::CapabilityUnsatisfied` 携带 `provider: String`，`Display` 时由 agent 层渲染上述 hint 模板（hint 文本本身不进结构化字段，避免字段污染）。

### 6.1.2 推荐的最佳实践配置形态

基于 §6.1.1 的 pitfall，推荐用户写成「全局 local 兜底 + 支持 hosted 的 provider 单独覆写为 delegate」：

```toml
[capabilities.search]
mode = "local"

[providers.anthropic.capabilities.search]
mode = "delegate"

[providers.openai.capabilities.search]
mode = "delegate"
```

这样：

- 默认安全——任何 provider 启动都能成功，最差降级为本地 search tool
- 支持 hosted 的 provider 自动用 hosted（更准、引用更全、不消耗本地工具配额）
- 新增 provider 时不会因为忘了写覆写而炸

### 6.2 `fetch`

`fetch` 不走 capability 装配逻辑。

它只走本地工具装配逻辑：

- 工具启用 -> 注册本地 `fetch`
- 工具禁用 -> 不注册

## 7. shadow 规则

你前面提到的 “shadow 掉本地的” 是对的，这里正式写成规则。

### 7.1 当 `search` 走 provider-hosted 时

- 不向模型暴露本地 `search` tool
- 不让模型同时看到 hosted search 和 local search 两条路径

原因：

1. 模型会在两种能力间摇摆
2. transcript 语义会分裂
3. 审批和可观测性会更混乱

### 7.2 当 `search` 走 local tool 时

- 不暴露 provider-hosted search

也就是说，`search` 在任一 turn 内都应只有一个来源。

### 7.3 MCP 工具一律命名空间化

更直接的规则：**所有** MCP 工具在本地一律以 `mcp.<server>.<name>` 注册，不区分名字、不区分 capability mode、不区分本地工具 enabled。

为什么不只在 `search` / `fetch` 撞名时改名：

1. 注册名一眼能看出「这是 MCP 工具，来自哪个 server」——provenance 在工具表上是显式的，不需要追溯
2. 后续给 defect 新增任何内置工具（`fetch`、`search`、未来的 `grep`、`memory` 等）都不会触发 MCP 旁路或静默改名，避免「不同 session 下同一个 MCP 工具有两种名字」
3. 「能力来源唯一」原则不再依赖 MCP 命名空间的运行时判断——只有内置 / hosted 才能占用裸名，MCP 永远走前缀

具体规则平坦：

- 任何 mode、任何 enabled 状态下，MCP server `<server>` 暴露的 `<tool>` 工具在本地 ToolRegistry 里以 `mcp.<server>.<tool>` 注册
- 内置 `search` tool（mode = Local 时注册）占用裸 `search`；MCP 的 `search` 走 `mcp.<server>.search`，不会与之竞争
- `Disabled` 时既没有内置 `search` 也没有 hosted search，但 MCP 仍然走前缀名——「session 不提供裸 `search`」这条约束自然满足
- 用户想调 MCP 的工具时直接用 `mcp.<server>.<tool>` 完整名字；不必关心是否撞名

「上游 wire 名」与「本地注册名」的边界：

- 注册名：`mcp.<server>.<tool>`——agent 的 `ToolRegistry` 与暴露给 LLM 的 tools schema 用这个
- Wire 名：`<tool>`（原始 MCP server 暴露的名字）——agent 调 `call_tool` 发回 MCP server 时用这个；server 不知道也不在乎本地前缀

具体实现见 [`crates/mcp/src/lib.rs`](../../crates/mcp/src/lib.rs) 的 `registered_mcp_tool_name`。

装配时仍然发 `ConfigWarning::McpToolRenamed { server, original, renamed }`，给用户一条「这是你 MCP server `<server>` 的 `<original>` 工具，在 defect 里以 `<renamed>` 形式调用」的明确告知。

## 8. transcript 与事件流

### 8.1 不再追求同一个底层执行模型

provider-hosted search 和 local search 的底层执行路径不同，这件事不需要强行抹平。

应当接受：

- provider-hosted search 走 provider response item 路径
- local search 走本地 `Tool` 路径

### 8.2 上层语义可以统一

虽然底层分轨，但在更高一层的 transcript / UI 上，仍然可以统一展示成：

- “发生了一次 search”
- “search 返回了哪些 sources / summary”

也就是说，统一的是**观测结果**，不是**执行机制**。

具体落地：hosted search 由 provider adapter 在解析流时翻译成 ACP `ToolCallUpdate { kind: ToolKind::Search, ... }`，复用 ACP 已有的 `Search` 工具种类。`tool_call_id` 由 provider 给的 hosted call id 直接采用（不复用本地 tool registry 的 id 分配器）。这样：

- ACP 客户端不需要认识「hosted」概念，所有 search 都长一样
- agent 主循环也不需要为 hosted 分配 tool_call_id
- transcript / UI 上统一显示 "Searching: <query>" → sources 列表

agent 主循环在收到 hosted search 的 ACP 推送后，**不**像本地工具那样需要等 `tool_call_update.status = Completed` 才进入下一轮——provider stream 自己会带回 result，turn loop 不阻塞在 hosted search 上。

### 8.3 不把 hosted search 接到 `Tool::execute`

这是本提案的硬性结论：

> provider-hosted search 不实现 `Tool` trait。

否则会导致：

- 审批时序假装一致
- provider stop reason 被错误建模成本地 tool pause/resume
- 本地 tool 测试矩阵被 hosted 路径污染

### 8.4 hosted search 在 history 里的形态

hosted search 的调用与结果**进入 history**（否则跨轮不可见，模型会重复发起同样的搜索），但用专属 `MessageContent` variant，不复用 `ToolUse / ToolResult`。

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

- **payload 黑盒**——agent 主循环不读它，只在 history 里搬运。和当前 `Thinking { signature }` 的 anthropic-only payload 同款套路，但语义更宽。
- **provider_id 进字段**——切换 provider 时 codec 能根据 `provider_id != self.id` 判断这条 activity 是「他人产生的」，决定丢弃还是降级为纯文本 summary（详见 §8.5）。
- **不持久化到磁盘**——`#[serde(skip)]` 或在持久化 codec 里显式丢弃。session resume 后如果模型再次触发 hosted search，会重新发起一次新调用，不依赖旧 payload。
- **不上抬到 ACP transcript**——前端看到的是 §8.2 描述的 `ToolCallUpdate { kind: Search }`，看不到 `ProviderActivity`。后者纯粹是 history 内部状态。

### 8.5 跨 provider 切换时的降级

session 切换 provider（例如用户中途改 model）时，history 里已有的 `ProviderActivity { provider_id: "anthropic", ... }` 不能原样喂给 OpenAI——payload schema 完全不同，wire 会报错。

降级规则：

- 切 provider 后首次发请求前，agent codec 扫一遍 history
- 所有 `provider_id != current_provider` 的 `ProviderActivity` 转成 `MessageContent::Text { text: synthesize_summary(activity) }`
- `synthesize_summary` 由 provider adapter 自己实现：把 hosted result 的 sources 列表渲染成 markdown 文本，丢失 citation 块的结构化数据但保住语义

这条降级是**有损**的（citation 丢失），但避免了双轨 history 维护，符合「history 真相只有一份」的现状。session 内不切 provider 时（绝大多数场景）payload 完整保留。

## 9. 对 `Tool` trait 的影响

### 9.1 `Tool` trait 不为 hosted search 扩签名

当前 [`Tool` trait](../internal/tool-trait.md) 仍然只服务于本地工具。

本提案明确：

- 不新增 `Hosted` 分支到 `Tool::execute`
- 不新增 `HostedTool` sibling trait 作为 P1 必需项

原因：

当前更重要的是先把架构边界拉清，而不是急着做 trait 并列抽象。

### 9.2 本地 `search` tool 仍然是普通工具

当 `SearchCapabilityMode = LocalTool` 时：

- 会注册一个本地 `search` tool
- 它就是普通 `Tool`
- 可以走审批、执行、事件流、错误恢复

所以 `Tool` trait 只需要承载：

- `fetch`
- `fs`
- `bash`
- local `search`

## 10. 对 provider 能力矩阵的影响

之前把 `web_search` 当成“工具层概念”是不够准确的。

新的结论应当是：

- `search` 本身不是 provider 通用能力字段
- provider 真正需要表达的是：是否支持 **hosted search capability**

但这类能力不建议直接塞进与 `thinking` / `vision` 平级的通用 `Capabilities` 里，原因：

1. 它不是纯模型属性，还受配置影响
2. 它和本地工具补位是联合决策
3. 更适合放在 provider adapter 的 hosted capability 声明层

### 10.1 `LlmProvider::hosted_capabilities()`

`LlmProvider` trait 新增独立方法：

```rust
pub trait LlmProvider {
    // ... 现有方法 ...

    /// provider 自报家门：当前实现支持哪些 hosted capability。
    /// 与运行时 `Capabilities`（模型属性）分离——前者是 adapter 自己
    /// 实现状态的体现，后者是模型本身的能力。
    fn hosted_capabilities(&self) -> HostedCapabilities;
}

#[non_exhaustive]
pub struct HostedCapabilities {
    pub search: bool,
}
```

各 provider 的实装选择：

- Anthropic：`search: true`（接 `web_search_20260209`）
- OpenAI：`search: true`（接 Responses API 的 `web_search`）
- DeepSeek / Echo：`search: false`

### 10.2 `CompletionRequest` 上的启用信号

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

这样 hosted capability 的启用控制是**单向、显式**的：agent → provider；provider 不能自己擅自启用 hosted。

### 10.3 hosted tool 版本选择

provider 的 hosted tool 自身有版本（如 Anthropic `web_search_20250305` / `web_search_20260209`，OpenAI Responses API 也会换 hosted tool 的 schema）。

P1 的处理是：**hosted tool 版本由 adapter 内部硬编码取最新**——agent 不感知版本字段，`HostedCapabilities { search: bool }` 只回答 yes/no。

理由：

- P1 没有需要在多版本间切换的真实需求；选最新即可
- 暴露版本字段会让 agent 层背上 provider-specific 知识，违反 §10.0 的 wire 编解码归属边界
- 真要做版本切换，应该是 adapter 自己根据 model id / capabilities 决定，不是配置层暴露

未来若 provider 之间版本演进节奏拉开（例如某个版本只对部分 model 可用），再抽 `HostedCapabilityCodec` trait，让 adapter 内部按需选择——这是 adapter 内部演进，不构成 trait breaking。

## 11. 对配置模型的影响

新的配置应当分成两棵树：

1. `capabilities.*`
2. `tools.*`

而不是都挤进 `tools.*.delegation`

### 11.1 `search`

`search` 放到能力树下：

```toml
[capabilities.search]
mode = "delegate"
```

provider 覆写：

```toml
[providers.anthropic.capabilities.search]
mode = "delegate"

[providers.openai.capabilities.search]
mode = "delegate"

[providers.deepseek.capabilities.search]
mode = "local"
```

### 11.2 `fetch`

`fetch` 继续放在工具树下：

```toml
[tools.fetch]
enabled = true
default_timeout_secs = 30
max_timeout_secs = 120
max_response_bytes = 5242880
default_format = "markdown"
```

## 12. 为什么 `search` 和 `fetch` 不再对称

这是一个正确的不对称。

### 12.1 `search`

- 本质是“让模型获得外部检索能力”
- 能力来源可能是 provider，也可能是本地工具
- 首先是能力问题，其次才是实现问题

### 12.2 `fetch`

- 本质是“执行一次可控的网络读取”
- 本地工具语义天然更强
- 首先是工具问题，不是能力协商问题

因此把两者彻底分开建模，是架构修正，不是特例补丁。

## 13. P1 落地建议

### 13.1 `search`

骨架：

- 定义 `SearchCapabilityMode`（§4.2）
- session 启动时按 §6.1 一次性裁决，写进 `SessionInitError::CapabilityUnsatisfied` 失败路径
- `Local` 时注册本地 `search` tool；`Delegate` 时不注册并 shadow MCP 同名（§7.3）
- `Disabled` 时完全不暴露 search

trait / 协议接缝：

- `LlmProvider::hosted_capabilities()` 新方法（§10.1），Anthropic / OpenAI / DeepSeek / Echo 各自实装
- `CompletionRequest::hosted_capabilities` 新字段（§10.2）
- `MessageContent::ProviderActivity { provider_id, kind, payload }` 新 variant（§8.4）；`#[serde(skip)]` 不持久化
- ACP 路径复用 `ToolKind::Search`，hosted search 由 provider adapter 直接产出 `ToolCallUpdate`（§8.2）
- 切 provider 时 codec 扫描 history 降级 `ProviderActivity`（§8.5）

### 13.2 `fetch`

- 直接实现 defect 原生 `fetch` tool
- 参考 codex / opencode 的经验补足：
  - timeout
  - response bytes cap
  - html -> markdown
  - content type 处理

### 13.3 测试矩阵

最少覆盖：

- `(mode, provider hosted_search)` 笛卡尔积：`Delegate × supported / Delegate × unsupported / Local × * / Disabled × *`
- session 启动校验：`Delegate + 不支持` 必返 `SessionInitError::CapabilityUnsatisfied`
- shadow 规则：`Delegate` 时本地 `search` tool 不出现在 `ToolRegistry::schemas()`
- MCP 命名冲突：三种 mode 下 MCP `search` 都重命名为 `mcp.<server>.search` 并发 warning；MCP `fetch` 无论 `tools.fetch.enabled` 取何值都重命名为 `mcp.<server>.fetch`
- history round-trip：hosted search 调用一次后，下一轮请求携带原 payload；切 provider 后 payload 降级为 summary 文本，history 不报 wire 错

### 13.4 文档同步

后续应拆成三篇正式文档：

1. `docs/internal/capabilities-search.md`
2. `docs/internal/tools-fetch.md`
3. `docs/internal/config-capabilities-and-tools.md`

并同步改动：

- `docs/internal/llm-trait.md`：补 `LlmProvider::hosted_capabilities()` 与 `CompletionRequest::hosted_capabilities`
- `docs/internal/tool-trait.md`：明确 hosted search 不实现 `Tool` trait，`Tool` 仍然只承载本地工具
- `docs/internal/session.md`（如已有）：补 session 启动期 capability 裁决

## 14. 拒绝的替代方案

### 14.1 把 hosted search 继续伪装成 `Tool`

拒绝原因：

- agent 无法真实介入执行
- 审批时序不成立
- stop reason 和事件流语义会被扭曲

### 14.2 让 hosted/local search 同时暴露给模型

拒绝原因：

- 能力来源竞争
- 行为不稳定
- 可观测性变差

### 14.3 把 `fetch` 也提升为 capability

拒绝原因：

- 当前没有足够收益
- 会把一个天然适合作为本地工具的能力过度抽象

## 15. 决议

采用本方案，后续所有实现与配置设计以此为准：

1. `search` 是 capability
2. `fetch` 是本地工具
3. provider-hosted 与 local tool 分轨
4. `search` 在任一 turn 内只有一个来源
