# `capabilities` / `tools` 配置重构提案

## 1. 结论

这份文档重写配置模型，核心结论只有两条：

1. `search` 配在 `capabilities.*`
2. `fetch`、`fs`、`bash` 这类本地执行项配在 `tools.*`

也就是说：

- provider-hosted search 的来源选择，是**能力配置**
- defect 原生 fetch 的启停和参数，是**工具配置**

旧方案里 `tools.search.delegation` 这类写法不再采用，因为它把“能力来源选择”和“本地工具配置”混在了一起。

## 2. 为什么要重写配置模型

之前配置方案最大的问题是试图用一套 `tools.*.delegation` 同时表达：

- 这个工具在本地是否存在
- 这个能力是否委托给 provider
- provider 不支持时是否 fallback

这会导致几个问题：

1. `search` 到底是工具还是能力，说不清
2. `fetch` 被错误地拉进 hosted delegation 语义
3. provider-hosted 与 local tool 的职责边界被配置层反向抹平

因此配置层必须先承认架构边界：

- 能力是能力
- 工具是工具

## 3. 顶层 schema

建议新的顶层结构是：

```toml
[capabilities]

[capabilities.search]

[tools]

[tools.fetch]
[tools.fs]
[tools.bash]
[tools.search]   # mode = local 时的本地 search tool 参数

[providers.<provider>.capabilities.search]
```

语义分工：

- `capabilities.*`
  - 决定模型在当前上下文拥有哪些高层能力
- `tools.*`
  - 决定 defect 本地工具是否启用、如何运行
- `providers.<p>.capabilities.*`
  - 覆写该 provider 下的能力来源
- 注意 P1 **不**支持 `providers.<p>.tools.*`——本地工具参数在所有 provider 下保持一致；真出现 per-provider 工具差异时再开口子，避免提前给一个无人使用的覆写位

## 4. `search` 的配置模型

### 4.1 `search` 是 capability

`search` 不再通过 `tools.search` 配置主语义。

主配置应写成：

```toml
[capabilities.search]
mode = "delegate"
```

### 4.2 枚举

建议直接用三态：

```rust
#[non_exhaustive]
pub enum SearchCapabilityMode {
    Delegate,
    Local,
    Disabled,
}
```

对应 TOML：

- `"delegate"`
- `"local"`
- `"disabled"`

### 4.3 语义

- `delegate`
  - 若当前 provider 支持 hosted search，则暴露 provider-hosted search
  - 不注册本地 `search` tool
  - 若 provider 不支持，则视为配置不满足
- `local`
  - 不暴露 provider-hosted search
  - 注册本地 `search` tool
- `disabled`
  - 不暴露 provider-hosted search
  - 不注册本地 `search` tool

### 4.4 为什么 `delegate` 不自动 fallback 到 `local`

P1 不建议让 `delegate` 自动回退。

原因：

1. 配置应可预测
2. 能力来源切换会改变行为和观测语义
3. fallback 应该由用户显式选择，而不是隐式发生

如果未来要支持 fallback，应新增独立模式，而不是让 `delegate` 偷偷变义。

## 5. provider 级 `search` 覆写

这是本次最关键的配置能力。

建议：

```toml
[capabilities.search]
mode = "local"

[providers.anthropic.capabilities.search]
mode = "delegate"

[providers.openai.capabilities.search]
mode = "delegate"

[providers.deepseek.capabilities.search]
mode = "local"
```

这样表达的是：

- 默认走本地 `search` tool
- Anthropic / OpenAI 下改成 provider-hosted
- DeepSeek 下继续本地

### 5.1 Rust 形状

建议新增：

```rust
pub struct CapabilitiesConfig {
    pub search: SearchCapabilityConfig,
}

pub struct SearchCapabilityConfig {
    pub mode: SearchCapabilityMode,
}
```

provider 覆写：

```rust
pub struct ProviderCapabilityOverrides {
    pub search: Option<SearchCapabilityConfig>,
}
```

然后放到各 provider config 下：

```rust
pub struct AnthropicConfigFile {
    pub base_url: Option<String>,
    pub default_model: Option<String>,
    pub models: Option<Vec<String>>,
    pub capabilities: Option<ProviderCapabilityOverrides>,
}
```

OpenAI / DeepSeek 同理。

## 6. `search` 的运行时决策

运行时顺序建议定死为：

1. 读取全局 `capabilities.search.mode`
2. 若存在 `providers.<provider>.capabilities.search.mode`，则覆写
3. 得到本轮最终 mode
4. 执行装配：
   - `delegate` -> 暴露 provider-hosted search，shadow 本地 `search`
   - `local` -> 注册本地 `search` tool，不暴露 provider-hosted
   - `disabled` -> 两边都不暴露

这条规则不应由 prompt 约束，而应由装配层硬决定。

### 6.1 推荐的最佳实践配置形态

`mode = "delegate"` 的失败语义在 [`search-capability-and-fetch-tool.md`](./search-capability-and-fetch-tool.md) §6.1.1 已写明（provider 不支持 hosted search 时 session 启动失败）。为避免「全局 delegate + 切到不支持 hosted 的 provider」导致的启动崩溃，推荐配置形态是：

```toml
[capabilities.search]
mode = "local"

[providers.anthropic.capabilities.search]
mode = "delegate"

[providers.openai.capabilities.search]
mode = "delegate"
```

即：

- 全局兜底为 `local`，新接入或不支持 hosted 的 provider 自然走本地工具
- 已知支持 hosted 的 provider 单独覆写为 `delegate`

这条建议同时写入 §11 的最小 TOML 草案与文档示例。

### 6.2 Inactive section warning

当某个 mode 让本来要生效的工具段失效时（例如 `mode = "delegate"` 时 `[tools.search]` 不进入 tool registry），`load_config` 应发结构化 warning：

```rust
pub enum ConfigWarning {
    // ... 既有 variant ...

    /// 配置文件里出现了某段，但在当前 mode 下不会生效。
    InactiveSection {
        path: PathBuf,
        section: String,           // 例如 "tools.search"
        reason: String,            // 例如 "capabilities.search.mode = \"delegate\""
    },
}
```

为什么不只用 `tracing::debug!`：

- tracing filter 是用户可调的（`RUST_LOG`），用户为了减噪很可能屏蔽
- 这条信息属于「配置语义」而不是「运行时观测」，应进入 `LoadedConfig.warnings` 让 `defect doctor` / `defect config show` 之类命令能稳定列出
- 与 `IgnoredProjectKey` / `UnknownKey` 等已有 warning 同级——它们也都是配置层信号

触发时机：load_config 完成后、装配前一次性扫描，按 `(mode, 段是否出现)` 笛卡尔积发：

| 场景 | 是否发 warning |
|------|----------------|
| `mode = "delegate"` 且配置里出现 `[tools.search]` | ✅ |
| `mode = "disabled"` 且配置里出现 `[tools.search]` | ✅ |
| `mode = "local"` 且配置里出现 `[tools.search]` | ❌（正常使用） |
| `mode = *` 且配置里没出现 `[tools.search]` | ❌（无段可言） |

未来其他 capability（image_generation / code_execution）落地时复用同一 warning variant，不再单独造轮子。

## 7. `fetch` 的配置模型

### 7.1 `fetch` 是本地工具

`fetch` 不进入 `capabilities.*`。

主配置写在：

```toml
[tools.fetch]
enabled = true
default_timeout_secs = 30
max_timeout_secs = 120
max_response_bytes = 5242880
default_format = "markdown"
html_to_markdown = true
follow_redirects = true
```

### 7.2 Rust 形状

```rust
pub struct FetchToolConfig {
    pub enabled: bool,
    pub default_timeout_secs: u32,
    pub max_timeout_secs: u32,
    pub max_response_bytes: u64,
    pub default_format: FetchFormat,
    pub html_to_markdown: bool,
    pub follow_redirects: bool,
}
```

### 7.3 provider 级 `fetch` 覆写

P1 **不支持**。

`fetch` 是全局本地工具，per-provider 启停或参数差异在 P1 没有真实需求——在 Anthropic 下不让 fetch、在 OpenAI 下让 fetch 的场景太罕见，提前给覆写位只会产生空配置段。

真出现 per-provider fetch 行为差异时再开口子（缺省值不变，不算 breaking）。同时也避免任何「per-provider 启用 hosted fetch」的暗示——hosted fetch 不是当前设计中心，参见 `search-capability-and-fetch-tool.md` §5。

## 8. 原生工具总启停

对本地工具仍然需要一个全局入口：

```rust
#[non_exhaustive]
pub struct ToolsConfig {
    pub defaults: ToolDefaultsConfig,
    pub bash: BashToolConfig,
    pub fs: FsToolConfig,
    pub fetch: FetchToolConfig,
    pub search: SearchToolConfig,
}
```

其中：

```rust
pub struct ToolDefaultsConfig {
    pub enabled: Option<Vec<String>>,
    pub disabled: Option<Vec<String>>,
}
```

对应 TOML：

```toml
[tools]
enabled = ["bash", "fs", "fetch"]
disabled = ["shell"]
```

`tools.enabled` / `tools.disabled` 列表里**只列真本地工具**，**不列 `"search"`**——本地 `search` tool 是否注册由 `capabilities.search.mode` 单一决定（`mode = "local"` 时注册，否则不注册）。这样 capability 层和 tool 层不会互相 override，避免 §3「能力是能力、工具是工具」的边界被启停列表反向打破。

`tools.search` 段（见 §9）仅在 mode = local 时生效——不是 mode = local 的额外开关，而是 mode = local 时本地实现的参数。

## 9. `tools.search` 段（mode = local 时的本地实现参数）

虽然 `search` 是 capability，但当 mode = `local` 时仍需要本地 `search` tool 实现，这套实现要有自己的参数段。TOML 路径放 `[tools.search]`——既然 capability 路径是 `[capabilities.search]`、tool 路径是 `[tools.search]`，namespace 已经分开，不会冲突，不需要 `_local` 之类的防御性命名。

```toml
[tools.search]
default_max_results = 8
default_recency_days = 30
backend_order = ["remote_api", "mcp"]
```

注意**不带 `enabled` 字段**——本地 `search` 是否注册完全由 `capabilities.search.mode` 决定，再加 `tools.search.enabled` 会与 capability 层重复。`tools.enabled` / `tools.disabled` 列表里也不列 `"search"`（见 §8）。

Rust：

```rust
pub struct SearchToolConfig {
    pub default_max_results: u32,
    pub default_recency_days: Option<u32>,
    pub backend_order: Vec<SearchLocalBackendKind>,
}
```

职责区分：

- `capabilities.search` → 有没有 search 能力、来源是什么
- `tools.search` → 如果来源是本地工具，它怎么工作

`tools.search` 在 `mode = "delegate"` / `mode = "disabled"` 时被读但不生效——保留段而不报错可以让用户在不同 mode 间切换时不丢配置。

## 10. 配置合并顺序

### 10.1 `search capability`

```text
内建默认
< [capabilities.search]
< [providers.<provider>.capabilities.search]
```

### 10.2 本地工具

```text
内建默认
< [tools]
< [tools.<tool>]
```

P1 **没有**`[providers.<p>.tools.<tool>]` 这一层——本地工具参数在所有 provider 下保持一致。理由见 §3 / §7.3：P1 没有真实的 per-provider 工具差异需求，提前给覆写位只会产生空配置段。

能力层和工具层分别合并，不互相覆盖。

## 11. 最小 TOML 草案

```toml
[capabilities.search]
mode = "local"

[tools]
enabled = ["bash", "fs", "fetch"]

[tools.fetch]
enabled = true
default_timeout_secs = 30
max_timeout_secs = 120
max_response_bytes = 5242880
default_format = "markdown"
html_to_markdown = true
follow_redirects = true

[tools.search]
default_max_results = 8
default_recency_days = 30
backend_order = ["remote_api", "mcp"]

[providers.anthropic.capabilities.search]
mode = "delegate"

[providers.openai.capabilities.search]
mode = "delegate"

[providers.deepseek.capabilities.search]
mode = "local"
```

要点：

- `tools.enabled` 列表不含 `"search"`
- 没有 `tools.search.enabled` / `tools.search.delegation` / `providers.<p>.tools.fetch` 等任何 P1 不支持的字段
- `tools.search` 段在所有 provider 下生效（DeepSeek 走 local 时直接读）；`providers.anthropic.capabilities.search.mode = "delegate"` 时该段在 Anthropic session 下不生效但保留

## 12. 为什么不再使用 `tools.search.delegation`

因为它会同时承载三层语义：

1. search 是否存在
2. search 是否本地执行
3. search 是否委托到 provider

这三件事不是同一抽象。

新的命名里：

- `capabilities.search.mode`
  - 只回答来源选择
- `tools.search`
  - 只回答 mode = local 时的本地实现参数

职责更干净。

## 13. 对 `ToolsConfig` / provider config 的建议修正

现有 [`crates/config/src/types.rs`](../../crates/config/src/types.rs) 需要扩两处：

### 13.1 增加能力配置

```rust
#[non_exhaustive]
pub struct EffectiveConfig {
    pub cli: CliConfig,
    pub turn: TurnConfig,
    pub base_prompt: BasePromptConfigFile,
    pub prompt: PromptConfigFile,
    pub capabilities: CapabilitiesConfig,
    pub providers: ProviderConfigs,
    pub tools: ToolsConfig,
    pub sandbox: SandboxConfig,
    pub tracing: TracingConfig,
    pub mcp: McpConfig,
    pub http: HttpClientConfig,
}

#[non_exhaustive]
pub struct CapabilitiesConfig {
    pub search: SearchCapabilityConfig,
}
```

`#[non_exhaustive]` 标注是为后续追加 `image_generation` / `code_execution` 等能力字段不构成 breaking。

### 13.2 扩展 provider 配置

各 provider config 增加：

```rust
pub capabilities: Option<ProviderCapabilityOverrides>,
```

P1 **不**增加 `tools: Option<ProviderToolOverrides>`——理由见 §3 / §7.3。真出现 per-provider 工具差异时再补，不算 breaking。

## 14. MCP 工具命名空间

**所有** MCP 工具在本地 `ToolRegistry` 里一律以 `mcp.<server>.<name>` 注册，不区分是否撞名、不区分 capability mode、不区分本地工具 enabled。

为什么不只在撞名时重命名：

1. 注册名一眼能看出「这是 MCP 工具，来自哪个 server」——provenance 在工具表上是显式的
2. 后续给 defect 新增任何内置工具（fetch、search、未来的 grep 等）都不会触发 MCP 旁路或静默改名，避免「不同 session 下同一个 MCP 工具有两种名字」
3. 配置层的 `capabilities.search.mode` 与 `tools.fetch.enabled` 都不再需要影响 MCP 命名，规则平坦化

「上游 wire 名」与「本地注册名」的边界：

- 注册名：`mcp.<server>.<name>`——agent 的 `ToolRegistry` 与 LLM 暴露的 `tools` schema 都用这个
- Wire 名：`<name>`（原始 MCP server 暴露的名字）——agent 调 `call_tool` 发回 MCP server 时用这个；server 不知道也不在乎本地的 `mcp.<server>.` 前缀

这条规则发生在 session 启动期 MCP 连接建立后、`ToolRegistry` 装配前；具体实现见 [`crates/mcp/src/lib.rs`](../../crates/mcp/src/lib.rs) 的 `registered_mcp_tool_name`。

`ConfigWarning::McpToolRenamed` 仍然保留——给用户一条「这是你 MCP server `<server>` 的 `<original>` 工具，在 defect 里以 `<renamed>` 形式调用」的明确告知，避免提示词里写裸名时困惑。

## 15. P1 落地建议

1. 先把 `capabilities.search.mode` 和 `tools.fetch` / `tools.search` 的 schema 定下来
2. 装配层实现 `search` 的三态选择，按 [`search-capability-and-fetch-tool.md`](./search-capability-and-fetch-tool.md) §6.1 在 session 启动期一次性裁决
3. 本地 `fetch` tool 按 `tools.fetch` 配置落地
4. provider 级实现 `providers.*.capabilities.search` 覆写
5. P1 **不**实现 `providers.*.tools.*`

## 16. 测试矩阵

最少覆盖：

- 全局 `capabilities.search.mode` 三态 × provider 覆写存在/不存在 = 6 种合并结果
- session 启动期 `mode = "delegate"` + provider 不支持 hosted search → `SessionInitError::CapabilityUnsatisfied`
- `tools.enabled` 含 `"search"` 时报 `ConfigWarning::UnknownToolEntry { name: "search" }`（防止用户误用旧 schema）
- `tools.search` 段在 `mode = "delegate"` 时被加载但不影响装配，发 `ConfigWarning::InactiveSection { section: "tools.search", reason: "capabilities.search.mode = \"delegate\"" }`
- `tools.search` 段在 `mode = "disabled"` 时被加载但不影响装配，发 `ConfigWarning::InactiveSection`
- `tools.search` 段在 `mode = "local"` 时**不**发 InactiveSection warning
- MCP server 暴露的**任意**工具（含但不限于 `search` / `fetch`）在所有 capability mode 下、所有本地工具 enabled 状态下都以 `mcp.<server>.<name>` 形式注册
- `providers.<p>.tools.fetch` 段出现时报 `ConfigWarning::UnknownKey`（P1 不支持）

## 15. 决议

采用本方案，后续配置设计以此为准：

1. `search` 走 capability 树
2. `fetch` 走 tool 树
3. provider-hosted 与 local tool 在配置层彻底分开
