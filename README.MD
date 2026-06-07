# Defect 缺陷机器人

Defect 是一个高可配置性，高兼容性，节省资源的无头通用agent

Defect 本仓库不提供任何ui实现，你可以通过ACP协议，在您喜欢的任何前端中来调用它

## 特点

1. **开放**，我们开放每一层可配置的配置，让Defect做到高度的可定制，可裁剪
2. **紧凑**，我们争取用最小的代价，实现最全面的功能，我们的可执行文件严格控制在20M以内，裁剪后只有12M，专为沙盒场景设计
3. **ACP 原生**，我们不绑定一个tui，而是鼓励您通过ACP来和Defect交互

## 安装

### 预构建二进制（推荐）

每个 `v*` tag 会通过 GitHub Actions 跨平台构建并发布到 [Releases](https://github.com/vast-enterprise/defect/releases)，覆盖 linux（gnu / musl）、macOS、windows。

**用 cargo-binstall（秒级，无需从源码编译）：**

```bash
cargo install cargo-binstall   # 一次性安装 binstall 本体
cargo binstall defect-cli      # 自动按当前平台拉取对应 release 二进制
```

**或直接下载 release asset（连 cargo 都不需要）：**

```bash
VERSION=v0.1.0
TARGET=x86_64-unknown-linux-musl   # 通用 linux 容器推荐 musl 版（静态、免 glibc 依赖）
curl -fsSL "https://github.com/vast-enterprise/defect/releases/download/${VERSION}/defect-${VERSION}-${TARGET}.tar.gz" | tar -xz
./defect --version
```

校验完整性：

```bash
curl -fsSLO "https://github.com/vast-enterprise/defect/releases/download/${VERSION}/SHASUMS256.txt"
sha256sum -c SHASUMS256.txt --ignore-missing
```

> ⚠️ musl 版为全静态二进制，**不含 bedrock provider**（aws-lc-sys 在 musl 上构建脆弱）。需要 AWS Bedrock 请选 `x86_64-unknown-linux-gnu` 等 gnu target。

平台与 target 对照：

| 环境 | target |
|---|---|
| 通用 linux 容器（alpine / distroless） | `x86_64-unknown-linux-musl`（无 bedrock） |
| 标准 linux / 需要 bedrock | `x86_64-unknown-linux-gnu` |
| linux arm64 | `aarch64-unknown-linux-gnu` |
| macOS (Apple Silicon / Intel) | `aarch64-apple-darwin` / `x86_64-apple-darwin` |
| windows | `x86_64-pc-windows-msvc` |

### 从源码安装

需要完整 Rust 工具链，走 `dist` profile（fat LTO，体积最小、编译较慢）：

```bash
# 远程仓库直接安装
cargo install --git https://github.com/vast-enterprise/defect defect-cli --bin defect --profile dist --locked

# 或在仓库内
just install-force
```

## 配置

Defect 使用 TOML 配置文件，按下面的优先级从低到高合并：

1. 内建默认值
2. 用户配置：`$XDG_CONFIG_HOME/defect/config.toml`，否则 `~/.config/defect/config.toml`
3. 项目共享配置：`<repo>/.defect/config.toml`（随仓库提交，团队共享）
4. 项目本地覆盖：`<repo>/.defect/config.local.toml`（机器本地，建议加入 `.gitignore`）
5. CLI override（`--config key.path=value`，以及 `--provider` / `--model`、`DEFECT_PROVIDER` / `DEFECT_MODEL`）

凭证（`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` 等）只通过环境变量注入，不写入 TOML——配置里用 `api_key_env` 指定环境变量名，而非密钥本身。不该提交的本地字段（如个人 `base_url`、`http.proxy`）请放进 `config.local.toml` 并加入 `.gitignore`；Defect 不替你做"出站字段安全裁剪"，配置项要么按写的生效，要么（未知 key）带文件路径硬失败。

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
