# 配置加载与合并

`defect-config` 负责把用户配置、项目配置、环境变量与 CLI override 收敛成一份
启动时可直接消费的强类型配置对象。本文先定义 **P1 最小可落地方案**，后续再演进
 profile / managed config / 远程组织配置等能力。

---

## 1. 目标与远期目标

### 1.1 目标

P1 只解决四件事：

1. **稳定的层级顺序**：用户 / 项目 / 本地项目 / CLI 覆盖有明确 precedence。
2. **强类型输出**：CLI、agent、provider、tool、sandbox 都拿结构化配置，不直接读 env / TOML。
3. **安全边界清晰**：项目配置可改“仓库内协作行为”，不可改“凭证去向 / 出站目标”。
4. **可解释**：出问题时能回答“这个最终值来自哪一层”。

### 1.2 远期目标

以下能力**不是 P1 范围**，但属于明确的后续演进方向：

- managed config / 企业强制策略
- 远程组织配置
- profile / 多套用户配置切换
- runtime 热重载
- 配置写回 API

这些在 `code-reference` 里都存在先例；这里只是明确它们不进入 P1 首批落地范围，不代表后续不做。

---

## 2. 参考实现结论

本设计主要参考了 `code-reference` 中三套实现：

### 2.1 Codex

`codex-rs` 的价值主要有三点：

1. **配置按 layer stack 表达**，而不是“直接 merge 完就丢掉来源”。
2. **项目层有 denylist / trust 边界**。项目配置不能改模型出口、通知、某些远程端点等。
3. **CLI override 走 dotted-path patch layer**，而不是散落在各处手工覆盖。

对我们最有价值的是第 2 点。`defect` 也是 agent，本地仓库里的配置不应能静默把流量导向任意 endpoint。

### 2.2 OpenCode

`opencode` 的价值主要有三点：

1. **明确写出 precedence**，并强调“配置是 merge，不是 replace”。
2. **按声明来源解析相对路径**，避免 merge 后 `./foo` 失去语义。
3. **少数数组字段做特判**，不是一刀切“所有数组 append”。

对我们最有价值的是第 1、3 点。`defect` 也应该把每个字段的 merge 规则写死，不能让实现时凭感觉。

### 2.3 Claw Code

`claw-code` 的价值主要有两点：

1. **层次很简单**：user → project → local，便于落地和测试。
2. **加载结果保留 loaded entries**，CLI 能做 `/config`、`doctor` 这类诊断。

对我们最有价值的是“先做简单层次，再留演进口”，这和 `defect` 当前阶段最匹配。

### 2.4 对 `defect` 的取舍

最终取舍：

- 借鉴 Codex 的 **layer stack + 项目层限制**
- 借鉴 OpenCode 的 **merge 语义显式化 + 路径按来源解析**
- 借鉴 Claw 的 **先做最小闭环**

不照搬的部分：

- 不做 Codex 那套 managed / trust / profile 全家桶
- 不做 OpenCode 的远程 config / 目录型 config / JSONC
- 不照搬 Claw 的 JSON 形态与兼容文件名；但保留它的 project-local override 思路

---

## 3. 配置源与优先级

P1 只支持五类输入，按 **低到高** 的 precedence 排列：

1. **内建默认值**
2. **用户配置**：`$XDG_CONFIG_HOME/defect/config.toml`，否则 `~/.config/defect/config.toml`
3. **项目配置**：从 `cwd` 向上找到仓库根；仅加载 `<repo>/.defect/config.toml`
4. **本地项目覆盖**：`<repo>/.defect/config.local.toml`
5. **CLI override**

补充规则：

- `.env` **不算配置层**。它只用于补 provider 凭证和少量兼容型 env。
- `DEFECT_*` 这类产品层环境变量不单独作为配置层；它们通过 CLI 入口的 `clap` `env` 支持注入参数解析结果，再自然落入 CLI override 层。
- 当前不做多级父目录 `.defect/config.toml` 链式叠加；只认 repo root 下的 `config.toml` 与 `config.local.toml`。

这样做的原因：

- 比 Codex 的 `cwd/tree/repo` 多层项目配置更简单，P1 足够。
- 比当前散落在 `crates/cli/src/main.rs` 的 `--provider` / `DEFECT_PROVIDER` / `.env` 逻辑更统一。

---

## 4. 文件格式与路径

### 4.1 格式

P1 使用 **TOML**。

原因：

- Rust 生态成熟，类型映射直接。
- workspace 里已经大量使用 TOML。
- 与 Codex 的 Rust 实现一致，后续 dotted-path override 也容易做。

### 4.2 路径

用户配置：

```text
$XDG_CONFIG_HOME/defect/config.toml
~/.config/defect/config.toml
```

项目配置：

```text
<repo>/.defect/config.toml
<repo>/.defect/config.local.toml
```

规则：

- 项目层以 git root 为准；若不在 git repo 内，则不加载项目配置。
- `config.toml` 面向仓库共享；`config.local.toml` 面向机器本地覆盖，默认应加入 `.gitignore`。
- 所有相对路径都相对于**声明它的配置文件所在目录**解析。

---

## 5. 配置对象分层

`defect-config` 不直接把原始 TOML 暴露给调用方，而是输出两层结构：

### 5.1 原始层

```rust
pub struct ConfigLayerEntry {
    pub source: ConfigSource,
    pub path: PathBuf,
    pub raw_toml: Option<String>,
    pub value: toml::Value,
    pub disabled_reason: Option<String>,
}
```

用途：

- debug / doctor / 测试断言
- 回答“这个值来自哪”
- 后续支持严格模式 / ignored key warning

### 5.2 有效配置层

```rust
pub struct EffectiveConfig {
    pub cli: CliConfig,
    pub turn: TurnConfigFile,
    pub provider: ProviderConfigSet,
    pub tools: ToolsConfig,
    pub sandbox: SandboxConfigFile,
    pub tracing: TracingConfigFile,
    pub mcp: McpConfig,
}
```

其中：

- `CliConfig`：CLI 自己消费，比如默认 provider、model、resume 路径等
- `TurnConfigFile`：映射到 `defect-agent::session::TurnConfig`
- `ProviderConfigSet`：各 provider 的显式配置
- `CapabilitiesConfig`：能力层配置（`search` 三态 mode；image_generation / code_execution 等待加），详见 [`capabilities.md`](./capabilities.md)
- `ToolsConfig`：bash / fs / fetch 等本地工具默认行为；本地 `search` tool 的参数也在 `[tools.search]` 段（仅 `capabilities.search.mode = "local"` 时生效，详见 [`tools-fetch.md`](./tools-fetch.md) / [`capabilities.md`](./capabilities.md)）
- `SandboxConfigFile`：默认 sandbox / permission policy
- `TracingConfigFile`：日志等级、结构化 tracing 相关项
- `McpConfig`：默认启用的 MCP server 名单与具名 server 定义表

P1 不要求这些类型一次到位，但要求 schema 先按这个方向组织，避免后面再从大杂烩拆分。

---

## 6. 建议 schema

P1 推荐的顶层 TOML 结构：

```toml
[default]
provider = "echo"
model = "echo"

[base_prompt]
file = "~/.config/defect/prompts/base.md"
text = "global instruction overlay"

[prompt]
file = "AGENTS.md"
text = "user-level appended prompt"

[prompt.providers.deepseek]
text = "deepseek-specific guidance"

[prompt.models.deepseek-v4-pro]
text = "model-specific guidance"

[turn]
request_limit = 32
max_llm_retries = 2
compact_threshold_tokens = 120000   # hard 水位绝对值覆盖（否则 compact_ratio·window）
compact_ratio = 0.85                 # hard 水位：到此同步压缩兜底
background_compact_enabled = true    # 越 soft 水位异步起摘要压缩，不阻塞当轮
compact_soft_ratio = 0.7             # soft 水位
microcompact_enabled = true          # 越 micro 水位清旧的大 tool_result，不调 LLM
microcompact_ratio = 0.6             # micro 水位
max_hook_continues = 3   # before-turn-end hook 强制续命硬上限（防 hook 无限 Continue）

[providers.anthropic]
base_url = "https://api.anthropic.com"
default_model = "claude-sonnet-4-5"
models = ["claude-sonnet-4-5", "claude-sonnet-4-5-thinking"]

[providers.openai]
base_url = "https://api.openai.com/v1"
default_model = "gpt-4o-mini"
models = ["gpt-4o-mini", "gpt-4.1-mini"]
organization = "org_xxx"
project = "proj_xxx"

[providers.litellm]
base_url = "http://localhost:4000/v1"
default_model = "openai/gpt-4o-mini"
models = ["openai/gpt-4o-mini", "anthropic/claude-sonnet-4-5"]
api_key_env = "LITELLM_API_KEY"

[providers.siliconflow]
protocol = "openai-chat"
base_url = "https://api.siliconflow.cn/v1"
default_model = "deepseek-ai/DeepSeek-V3"
models = ["deepseek-ai/DeepSeek-V3"]
display_name = "SiliconFlow"
api_key_env = "SILICONFLOW_API_KEY"

[providers.siliconflow.headers]
x-provider-test = "enabled"

[tools.bash]
default_timeout_ms = 30000
max_timeout_ms = 600000

[tools.fs]
read_default_limit = 2000
read_max_limit = 20000

[tools.background]                   # 后台 subagent 进度视图（inspect_background_task 看到的"最近几个消息块"）
default_recent_blocks = 10           # inspect 不带 recent_blocks 时默认返回多少条最近消息块（=提交给 LLM 的 Message 块，非流式增量）
block_text_limit = 0                 # 单 block 自由正文(assistant/思考/工具结果)字符上限；0=只留摘要/元信息(默认，不灌子 turn 正文)；工具名不受此限

[sandbox]
mode = "ask-writes"

[tracing]
filter = "info,toac=warn"

[mcp]
enabled_servers = ["echo"]

[mcp.servers.echo]
transport = "stdio"
command = "mcp-echo"
args = ["--port", "9000"]

[mcp.servers.echo.env]
MCP_TEST_VALUE = "from-config"

[http]
total_timeout_ms = 600000        # 单次请求总超时，0 = 不限
transport_retries = 2            # transport 抖动重试上限
initial_backoff_ms = 200         # 重试初始 backoff
user_agent = "my-org-agent/1.0"  # 不设则用编译期默认

[http.proxy]
mode = "from-env"                # "from-env"（默认）| "disabled" | "explicit"
http_proxy = "http://127.0.0.1:10808"
https_proxy = "http://127.0.0.1:10808"
no_proxy = ["localhost", ".internal"]
```

当前语义补充：

- `default.model` 用于全局当前默认模型。
- `providers.<name>.default_model` 用于该 provider 的默认模型。
- `providers.<name>.models` 是该 provider 在配置层声明的候选白名单。
- ACP 暴露给前端的模型候选会先取 provider `list_models()`，再与配置白名单取交集。
- `base_prompt` 是**单层覆盖**语义。最终只选择最高优先级且显式声明的那一层；该层内部可同时提供 `file` 与 `text`，按 `file -> text` 拼接。
- `prompt` 是**逐层追加**语义。当前顺序为：`base_prompt -> Environment -> prompt.text -> prompt.file(默认 AGENTS.md) -> provider overlay -> model overlay -> session overlay`。
- `prompt.file` 默认值为 `AGENTS.md`。当使用默认值时，会从 repo root 到当前 `cwd` 逐级收集同名文件并顺序拼接，便于项目级 prompt 自然随目录层级叠加。
- **拼接格式**：每个片段套一级标题（`#`，如 `# Base Prompt` / `# Project Instructions (apps/web/AGENTS.md)`），片段之间以 markdown 水平分割线（`---`）相隔，让模型把每段当作独立文档理解。**编写 `base_prompt` / `AGENTS.md` 等 prompt 文档时，正文标题请从二级（`##`）起步**——一级标题（`#`）由注入层占用，从二级起步可自然嵌套其下，避免与片段边界混淆。
- **`# Environment` 段**：在 `base_prompt`（身份）之后、project 约定之前自动注入运行环境事实——平台与发行版版本、defect 版本、frontend 接入方式（ACP 时附带 fs / shell 是 `local` 直控还是 `delegated` 经客户端代理）、cwd、默认 shell。该段始终注入，不受 prompt 配置是否为空影响。
- `[http]` 顶段对应 `defect_http::HttpStackConfig`；`mode = "explicit"` 时需要同时给出 `http_proxy` / `https_proxy`，`no_proxy` 走 [GNU 风格](https://about.gitlab.com/blog/we-need-to-talk-no-proxy/) 域名后缀匹配。`mode = "from-env"`（默认）从 `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` 环境变量读取；`mode = "disabled"` 强制不走代理。

[mcp.servers.docs]
transport = "sse"
url = "http://127.0.0.1:8123/mcp"

[mcp.servers.docs.headers]
x-mcp-test = "enabled"
```

说明：

- `default.provider` / `default.model` 是全局默认选择。
- `default.provider` 可指向内置 provider（`echo` / `anthropic` / `openai` / `deepseek` / `litellm`），也可指向 `[providers.<name>]` 中声明的自定义 provider。
- `providers.<name>.default_model` 是该 provider 的默认模型；当当前 provider 命中时可作为回退值。
- `providers.<name>.models` 是该 provider 允许暴露的模型集合。
- `providers.<name>.protocol` 是协议编解码选择，不是一个独立 instance 概念。P1 自定义 provider 只支持 `openai-chat`；后续 Bedrock 这类 provider 会复用其它 protocol 但单独处理 transport / auth。
- `providers.<name>.base_url` / `api_key_env` / `headers` 描述 provider 运行时接入点。`api_key_env` 只声明环境变量名，实际凭证仍不进入 TOML。
- `mcp.servers` 是“可引用的 server 定义表”。
- `mcp.enabled_servers` 是入口默认启用的 MCP server 名单。
- `session/new.mcp_servers` 仍然保留，用于会话级追加；若与入口默认配置重名，则会话级配置覆盖默认配置。
- 凭证不进 TOML，仍走 env。

凭证 env 规则：

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `DEEPSEEK_API_KEY`

兼容 env 输入：

- `DEFECT_PROVIDER`
- `DEFECT_MODEL`

这两个在实现上应被视为 **CLI 层等价覆盖**，具体做法是直接使用 `clap` 的 `env` 支持，而不是在 `defect-config` 或 provider 内部再各自手工读一遍。

---

## 7. Merge 规则

这是实现时最重要的部分。P1 不允许“统一 deep merge 一把梭”。

### 7.1 标量

标量字段直接按高优先级覆盖低优先级：

- `string`
- `bool`
- `integer`
- `duration-like integer`

例如：

- `default.provider`
- `default.model`
- `sandbox.mode`
- `tracing.filter`
- `mcp.servers.echo.transport`
- `mcp.servers.docs.url`

### 7.2 表对象

表对象按 key 递归 merge。

例如：

- `[providers.openai]`
- `[tools.bash]`
- `[mcp.servers.echo]`
- `[mcp.servers.docs.headers]`

低层有、上层没有的 key 保留；冲突 key 由高层覆盖。

### 7.3 数组

P1 默认规则：**数组整体替换，不做拼接**。

原因：

- 最容易解释。
- 避免 OpenCode 那种“部分数组去重拼接、部分覆盖”的复杂度进入 P1。

例外只在真正需要时单独开白名单。目前 P1 不预设数组拼接字段。

已落地的典型例子：

- `mcp.enabled_servers`
- `mcp.servers.<name>.args`

### 7.4 dotted-path CLI override

CLI override 统一先构造成一层 TOML，再按普通 merge 流程叠上去。

例如：

```text
--config default.provider=\"anthropic\"
--config providers.anthropic.default_model=\"claude-sonnet-4-5\"
--config providers.anthropic.models='[\"claude-sonnet-4-5\",\"claude-sonnet-4-5-thinking\"]'
--config tools.bash.default_timeout_ms=10000
--config mcp.enabled_servers='[\"echo\",\"docs\"]'
```

这样实现比在 `main.rs` 里一个字段一个字段手工覆盖更稳，也更接近 Codex 的做法。

---

## 8. 项目配置的边界

**早期实现**对共享项目层 `config.toml` 做过一份 denylist sanitize：剥离
`default.provider` / `providers.*.{base_url,api_key_env,...}` / `tracing.otlp` /
`tracing.langfuse` / `http.proxy` 并发 `IgnoredProjectKey` warning，理由是「仓库内
checked-in 配置不应能静默把流量/凭据导向第三方」（借鉴 Codex 的 trust 边界）。

**现已移除（决策见下）**。`config.toml` 与 `config.local.toml`、用户配置一视同仁，
所有 key 原样生效，不再剥离、不再 warning。

为什么移除：

- 产品宗旨是**最小化**——不替用户审查仓库共享配置是否「可疑」。
- 这层防护防的是「clone 不可信仓库时被劫持」，但那本质是用户该不该信任、该不该
  在陌生仓库里跑 agent 的问题，不该由配置加载器替用户做静默裁剪。裁剪反而违背
  「可解释」目标（用户写了配置却被无声改写）。
- 留着就得维护一份「敏感字段」镜像清单，又是一处手维护真相源（参见 §11.1 删掉
  key 白名单的同一动机）。

随之删除的代码：`sanitize_shared_project_layer`、`ConfigWarning::IgnoredProjectKey`、
`overrides::{remove_toml_path, remove_toml_table_key}`。

> 注：§1.1 第 3 条「安全边界清晰」与 §2.1 借鉴 Codex denylist 的表述是历史背景，
> 当前实现不再保留该边界。

---

## 9. 环境变量策略

当前仓库里 provider 自己还在读 env。P1 后应统一成下面的规则：

### 9.1 CLI 入口负责

- 用 `clap` 的 `env` 支持读取 `DEFECT_PROVIDER`
- 用 `clap` 的 `env` 支持读取 `DEFECT_MODEL`
- 输出规范化后的 CLI 参数给 `defect-config`

### 9.2 配置 crate负责

- 读取 XDG / HOME
- 读取配置文件
- 合并“文件层 + CLI 解析结果”
- 读取 `.env` 文件并注入进程环境的兼容逻辑

### 9.3 provider crate负责

- 只消费显式传入的 provider config
- 凭证若未显式传入，可从标准 env 读

### 9.4 统一优先级

对 `provider` / `model`：

```text
CLI flag > DEFECT_* env > config file > built-in default
```

对凭证：

```text
provider-specific env only
```

解释：

- `provider` 和 `model` 的 env 接入点归 CLI 参数解析层；配置系统只接收“已经解析好的 CLI override”
- `API key` 是鉴权输入，归 provider

---

## 10. 读取流程

推荐的加载流程：

1. 解析 `cwd`
2. 定位用户配置路径
3. 定位 git repo root 与项目配置路径
4. 读取 `<repo>/.defect/config.toml`
5. 读取 `<repo>/.defect/config.local.toml`
6. 逐层生成 `ConfigLayerEntry`，每层 load 时 `try_into::<ConfigToml>()` 校验未知 key（§11.1）
7. 解析 CLI override 为一层虚拟 TOML
8. merge 成 `merged_toml`
9. 将 `merged_toml` 反序列化为强类型 `ConfigToml`
10. 衍生 `EffectiveConfig`

建议 API：

```rust
pub struct LoadConfigOptions {
    pub cwd: PathBuf,
    pub cli_overrides: Vec<(String, toml::Value)>,
}

pub struct LoadedConfig {
    pub layers: ConfigLayerStack,
    pub effective: EffectiveConfig,
    pub warnings: Vec<ConfigWarning>,
}

pub fn load_config(opts: LoadConfigOptions) -> Result<LoadedConfig, ConfigError>;
```

---

## 11. 错误与 warning

按 [`docs/internal/errors.md`](./errors.md) 的规则，`defect-config` 应提供：

```rust
#[non_exhaustive]
pub enum ConfigError {
    Io { path: PathBuf, source: BoxError },
    Parse { path: PathBuf, source: BoxError },
    Invalid { path: PathBuf, message: String },
    Source(#[source] BoxError),
}
```

同时单独定义 warning：

```rust
pub enum ConfigWarning {
    DeprecatedKey { path: PathBuf, old: String, new: String },
    // InactiveSection / McpToolRenamed 等前向预留变体见 types.rs
}
```

P1 建议：

- 用户层 parse error：直接 fail
- 项目层 parse error：直接 fail
- 未知 key：直接 fail（见下）
- 项目层不再做 denylist sanitize（§8）

### 11.1 未知 key 校验：`deny_unknown_fields`，不手维护白名单

早期实现用一份手维护的字面量白名单（`is_known_config_key` 等 ~200 行）逐键比对
未知 key，命中则发 `ConfigWarning::UnknownKey`。问题是它和 `XxxSection` struct 是
两份真相源：每加字段要改多处，漏抄会静默把合法 key 误报成未知。

现在改为单一真相源：每个 `XxxSection` struct 加 `#[serde(deny_unknown_fields)]`，
未知 key 由 serde 在 decode 时直接报 `ConfigError::Invalid`，不再 warning。校验
在 **每层 load 时单独 try_into** 跑（而非合并后一次性 decode），这样：

- 错误信息能带上「来自哪个文件」的 provenance（合并后 decode 只能报 `<merged>`）；
- CLI override 层的拼写错误也能被抓到。

唯一的结构性例外是 `ProvidersSection`——它用 `#[serde(flatten)]` 接住开放的
自定义 provider 名（`[providers.<任意名>]`），serde 规定 flatten 不能和
`deny_unknown_fields` 共存。解法是分层施加：外层 `ProvidersSection` 不加 deny
（保留 provider 名开放），内层 `ProviderSection` 加 deny（校验字段名）。这样
`providers.siliconflow`（自定义名）被接受，`providers.siliconflow.bogus`（错字段）
仍报错。

`[hooks]` 段不走 `ConfigToml::try_into`（见 `crates/config/src/hooks.rs` 顶部注释），
所以 `ConfigToml` 用一个 `#[serde(default)] hooks: toml::Value` 吸收字段把它放过，
hooks 自己的解析器做 schema 校验并 loud error。

---

## 12. 与当前代码的接线方式

当前 `crates/cli/src/main.rs` 里有三块临时逻辑：

1. `.env` 加载
2. `--provider` / `--model`
3. `build_provider()` 内各 provider 默认 model

P1 落地后应变成：

- `main.rs` 只解析原始 CLI 参数
- `defect-config` 产出 `LoadedConfig`
- `main.rs` 只做装配：
  - 从 `LoadedConfig.effective` 取当前 provider
  - 从 `LoadedConfig.effective` 取 turn config
  - 从 `LoadedConfig.effective` 取 tool config
  - 把 provider-specific config 传给 `defect-llm`

也就是说：

- `main.rs` 不再自己决定 precedence
- `DEFECT_PROVIDER` / `DEFECT_MODEL` 继续由 `clap` 的 `env` 机制处理；`defect-config` 不重复解析它们
- `defect-llm` 不再承担产品配置选择逻辑

---

## 13. 测试矩阵

P1 至少覆盖这些测试：

1. 用户配置单层加载成功
2. 用户 + 项目 merge，项目覆盖同名标量
3. 用户 + 项目 + 本地项目 merge，本地项目覆盖共享项目层
4. 递归表对象保留非冲突 key
5. CLI override 覆盖本地项目层
6. `DEFECT_PROVIDER` / `DEFECT_MODEL` 等价于 CLI 上层覆盖
7. 共享项目层设置 `providers.openai.base_url` 直接生效（不再剥离，§8）
8. `config.local.toml` 设置 `providers.openai.base_url` 生效
9. 相对路径字段按声明文件目录解析
10. 数组字段整体替换而非拼接
11. 缺失配置文件不报错
12. TOML 语法错误带 path

如果要补一条 golden test，建议断言：

- `LoadedConfig.layers`
- `LoadedConfig.warnings`
- `LoadedConfig.effective`

三者一起稳定输出。

新增建议覆盖：

13. `mcp.enabled_servers` 引用未定义 server 时返回 `ConfigError::Invalid`
14. `mcp.servers.<name>` 的 transport-specific 必填字段校验
15. CLI 组装时，入口默认启用的 MCP server 能在不传 `session/new.mcp_servers` 的情况下生效
16. `session/new.mcp_servers` 与入口默认 MCP server 重名时，会话级配置覆盖默认配置

---

## 14. P1 实现顺序

建议按下面顺序落：

1. `docs/internal/config.md`
2. `defect-config` 的原始 schema 与 `ConfigError`
3. layer discovery + TOML parse + 逐层未知 key 校验（§11.1）
4. merge
5. `EffectiveConfig` 映射
6. `crates/cli/src/main.rs` 改为走 `defect-config`
7. precedence / warning / path 解析测试

---

## 15. 最终决策

P1 采用：

- **TOML**
- **default < user < project < project-local < CLI**
- **`.env` 不算配置层**
- **数组默认整体替换**
- **未知 key 由 `deny_unknown_fields` 逐层 hard fail，不维护字面量白名单（§11.1）**
- **项目层不做 denylist sanitize；所有层 key 一视同仁（§8，宗旨最小化）**
- **`config.local.toml` 作为机器本地覆盖层，默认高于共享项目层**
- **保留 layer stack 和 warnings，方便 debug / doctor / 后续 session metadata**
- **入口配置已支持 `mcp.enabled_servers` + `mcp.servers.<name>`，并在 CLI 启动时注入默认 MCP server**

这是比当前实现更统一、比 Codex/OpenCode 更轻的一版。
