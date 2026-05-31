# Defect 缺陷机器人

Defect 是一个高可配置性，高兼容性，节省资源的无头通用agent

Defect 本仓库不提供任何ui实现，你可以通过ACP协议，在您喜欢的任何前端中来调用它

Defect 的远期目标是：让您在冰箱上也能运行Agent

## 特点

1. **开放**，我们开放每一层可配置的配置，让Defect做到高度的可定制，可裁剪
2. **紧凑**，我们争取用最小的代价，实现最全面的功能，我们的可执行文件严格控制在20M以内，裁剪后只有12M，专为沙盒场景设计
3. **ACP 原生**，我们不绑定一个cli，而是鼓励您通过ACP来和Defect交互

## 安装

目前，本项目还在开发中，您可以通过在项目下执行`just install-force`来安装当前编译产物。

## 配置

Defect 使用 TOML 配置文件，按下面的优先级从低到高合并：

1. 内建默认值
2. 用户配置：`$XDG_CONFIG_HOME/defect/config.toml`，否则 `~/.config/defect/config.toml`
3. 项目共享配置：`<repo>/.defect/config.toml`（随仓库提交，团队共享）
4. 项目本地覆盖：`<repo>/.defect/config.local.toml`（机器本地，建议加入 `.gitignore`）
5. CLI override（`--config key.path=value`，以及 `--provider` / `--model`、`DEFECT_PROVIDER` / `DEFECT_MODEL`）

凭证（`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` 等）只通过环境变量注入，不写入 TOML。出于安全考虑，仓库共享层（`config.toml`）中的 `default.provider`、`providers.*.base_url`、`http.proxy` 等出站/凭证相关字段会被忽略并产生 warning；如需本地覆盖请放到 `config.local.toml`。

一份可直接起步的最小配置（放在 `~/.config/defect/config.toml` 或 `<repo>/.defect/config.local.toml`）：

```toml
[default]
provider = "deepseek"
model = "deepseek-v4-pro"

[providers.deepseek]
default_model = "deepseek-v4-pro"
models = ["deepseek-v4-pro", "deepseek-v4-flash"]

# 全局 system prompt overlay，所有会话生效
[base_prompt]
text = "你是名为 defect 的助手"

# 项目级 prompt，默认会从 repo root 到 cwd 逐级拼接 AGENTS.md
[prompt]
file = "AGENTS.md"
```

更完整的字段说明、合并语义、安全边界与 MCP / HTTP 代理等高级配置，参见 [`docs/internal/config.md`](docs/internal/config.md)。

## 二次开发/裁剪

`defect-cli` 已拆成 `lib` + `bin/cli`：所有装配 helper（provider / tool / hook / mcp / sandbox / http / storage / tracing）都在 `defect_cli::*` 模块导出，主 binary `src/bin/cli.rs` 只剩三段拼装逻辑。要做二次开发：

1. 把 `crates/cli/src/bin/cli.rs` 复制进你的 crate
2. 按需替换其中任意一步——例如换 provider 就跳过 `providers::build_registry`，自己组 `ProviderRegistry::new(...)`；换 hook 就传一份自己的 `Arc<dyn HookEngine>`
3. 其他 helper 直接 `use defect_cli::{tools, mcp_servers, paths, ...}` 复用

裁剪建议：

- 不需要 hook 系统？传 `Arc::new(NoopHookEngine)` 即可，引擎本身零开销
- 不需要 MCP？skip `session_tool_factory`
- 不需要持久化？skip `observe_session` / `session_loader`
- 想缩二进制体积？我们后续会给 `defect-cli` 加可选 feature，按 provider / tool 维度编译期裁剪——届时跟着 feature flag 关即可
