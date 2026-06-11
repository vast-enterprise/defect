# Defect 配置参考

本文档完整描述 Defect 的配置文件系统：文件位置、合并语义、所有 TOML section 与字段，以及 provider / hook / skill / MCP 等子系统的配置方式。

> 配置 schema 的唯一真相源是 `crates/config/src/types.rs`（以及 `hooks.rs` / `mcp.rs` / `profiles.rs` / `skills.rs`）。本文档若与代码不符，以代码为准。

---

## 1. 配置文件位置与合并

Defect 使用 TOML 配置文件，按下面的优先级**从低到高**逐层合并（高层覆盖低层的同名标量；数组/表的合并语义见各 section 说明）：

| 优先级 | 层 | 路径 | 说明 |
|---|---|---|---|
| 1（最低） | 内建默认值 | — | 见本文档各字段的「默认」列 |
| 2 | 用户层 | `$XDG_CONFIG_HOME/defect/config.toml`，否则 `~/.config/defect/config.toml` | 跨项目的个人配置 |
| 3 | 项目共享层 | `<repo-root>/.defect/config.toml` | 随仓库提交、团队共享 |
| 4 | 项目本地层 | `<repo-root>/.defect/config.local.toml` | 机器本地覆盖，建议加入 `.gitignore` |
| 5（最高） | CLI override | `--config key.path=value`（可重复）、`--provider` / `--model` / `--sandbox` / `--log-format`、环境变量 `DEFECT_PROVIDER` / `DEFECT_MODEL` | 命令行最终决定 |

- **repo root 检测**：从 `cwd` 向上逐级查找含 `.git` 的目录；找不到则项目层与项目本地层都不加载。
- **`--local` 模式**：锚定到 `<repo-root>/.defect/`，**完全忽略用户层**（配置、`agents/`、`skills/` 全部跳过），所有状态只读写项目根 `.defect/`。适合在沙箱/容器里得到可复现的、不受宿主机用户配置影响的行为。

### 未知 key 一律硬失败

任何拼错的 key 或不存在的字段都会立即报错，并带上**出错的文件路径**。配置项要么生效，要么报错，没有"写了但没用"的灰色地带——拼错的字段不会被静默忽略。

> 例外：`[providers]` 顶层允许任意自定义 provider 名（`[providers.<任意名字>]`），所以不会因"未知 provider 名"报错；但每个 provider section **内部**的字段仍然严格校验。

### 凭证只走环境变量

API key 等凭证**不写入 TOML**。配置里只用 `api_key_env` 指定**环境变量名**，运行期从该环境变量读取实际密钥。常用：`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `DEEPSEEK_API_KEY`。

> 配置里写什么就生效什么，不区分"敏感字段"。把不该提交进仓库的东西（如本地 `base_url`、代理）放进 `config.local.toml`（默认已被 `.gitignore` 忽略），而不要写进随仓库共享的 `config.toml`。

---

## 2. 顶层 section 一览

`config.toml` 的顶层 section：

| Section | 作用 |
|---|---|
| `[default]` | 默认 provider / model 选择 |
| `[base_prompt]` | 全局 system prompt overlay（所有会话生效） |
| `[prompt]` | 项目级 prompt（默认拼接 `AGENTS.md`）+ per-provider / per-model overlay |
| `[turn]` | 单轮循环行为：请求上限、上下文压缩、重试、并发、子 agent 深度 |
| `[capabilities]` | 全局能力开关（目前仅 `web_search`） |
| `[providers]` / `[providers.<name>]` | LLM provider 配置 |
| `[tools.*]` | 本地工具参数：`bash` / `fs` / `fetch` / `search` / `background` |
| `[sandbox]` | 权限模式 |
| `[tracing]` / `[tracing.otlp]` / `[tracing.langfuse]` | 可观测性 |
| `[mcp]` | MCP server 配置 |
| `[http]` / `[http.proxy]` | HTTP 客户端栈与代理 |
| `[[hooks.<event>]]` | hook 流水线（数组语义，特殊合并；见第 9 节） |

---

## 3. `[default]` — 默认 provider / model

```toml
[default]
provider = "anthropic"      # 见下；默认 "defect"（内置 echo provider）
model = "claude-sonnet-4-5"
```

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `provider` | string | `"defect"` | provider 名。内置：`defect` / `anthropic` / `openai` / `deepseek` / `litellm`；任何其他值视为自定义 provider，必须有对应的 `[providers.<name>]` section，否则报错 |
| `model` | string | 见下 | 默认模型 id |

**model 解析顺序**：`default.model` → `[providers.<provider>].default_model` → provider 内建默认。内建默认：

| provider | 内建默认 model |
|---|---|
| `defect` | `echo` |
| `anthropic` | `claude-sonnet-4-5` |
| `openai` | `gpt-4o-mini` |
| `deepseek` | `deepseek-chat` |
| `litellm` | 无（必须显式给 `default.model` 或 `default_model`） |
| 自定义 | 无（必须显式给） |

若最终解析不出 model，启动报错。

> `defect` provider 是内置的占位 provider：把用户最近一条消息原样回显（model id `echo`），无需任何凭证。它是 `default.provider` 的兜底默认，便于在没配 key 时也能跑通链路。

---

## 4. `[providers]` — LLM provider 配置

四个内置 provider 各有固定 section 名，自定义 provider 用任意 section 名：

```toml
[providers.anthropic]
api_key_env = "ANTHROPIC_API_KEY"
default_model = "claude-opus-4"
models = ["claude-sonnet-4-5", { id = "claude-opus-4", name = "Opus 4" }]

[providers.openai]
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
organization = "org-xyz"

# 自定义 provider：section 名即 provider 名，用 `--provider my_gw` 选中
[providers.my_gw]
protocol = "openai-chat"
base_url = "https://gateway.internal/v1"
api_key_env = "MY_GW_API_KEY"
default_model = "gpt-4o"
```

每个 provider section（`[providers.<name>]`）的字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `protocol` | `"anthropic-messages"` / `"openai-chat"` | wire 协议。内置 provider 已知协议，自定义 provider 必须指明 |
| `base_url` | string | 自定义 API base URL（默认走该 provider 官方地址） |
| `default_model` | string | 该 provider 的默认 model（覆盖内建默认，被 `default.model` 覆盖） |
| `models` | array | 模型候选列表，供运行时切换模型时选择。两种写法见下 |
| `display_name` | string | UI 显示名 |
| `api_key_env` | string | **API key 的环境变量名**（如 `"ANTHROPIC_API_KEY"`），不是 key 本身 |
| `organization` | string | OpenAI organization id |
| `project` | string | OpenAI project id |
| `aws` | table | Bedrock/AWS 配置，见下 |
| `headers` | table<string,string> | 自定义 HTTP 头 |
| `capabilities` | table | per-provider 能力覆盖，见第 6 节 |
| `reasoning_effort` | enum | 推理强度（OpenAI 兼容协议），见下 |

### `models` 的两种写法

```toml
# 纯 id（显示名回退到 id 本身）
models = ["gpt-4o", "gpt-4o-mini"]
# 表形式：长 id 配短显示名
models = [{ id = "claude-opus-4-20250514", name = "Opus 4" }]
# 两种可混用
models = ["claude-sonnet-4-5", { id = "claude-opus-4", name = "Opus 4" }]
```

### `reasoning_effort`

取值（snake_case）：`none` / `minimal` / `low` / `medium` / `high` / `xhigh`。1:1 映射 OpenAI wire 枚举。配置层不区分模型，原样透传，由上游校验（`xhigh` 仅 `gpt-5.1-codex-max+`，`none` 仅 `gpt-5.1+`）。

### Bedrock / AWS

```toml
[providers.anthropic.aws]
profile = "my-aws-profile"   # 可选
region = "us-east-1"         # 可选
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `profile` | string | AWS 配置文件名 |
| `region` | string | AWS 区域 |

Bedrock 凭证由 AWS SDK 链处理（IAM / 环境 / profile），不用 `api_key_env`。

> **provider 选择是 `(vendor, model)` 对**：同一个 model id 可以在多个 provider 下出现（如官方 Anthropic 与 Bedrock 同时配 `claude-sonnet-4-5`），靠 provider 区分。`litellm` 复用 OpenAI 协议实现，跟随 `provider-openai` 编译 feature。

### 按 provider 裁剪二进制

provider 实现受编译 feature 控制（`provider-anthropic` / `provider-bedrock` / `provider-openai` / `provider-deepseek`，默认全开）。如果选中一个未编译进当前二进制的 provider，启动会报错提示重新带 feature 编译。

---

## 5. `[base_prompt]` 与 `[prompt]`

### `[base_prompt]` — 全局 overlay

所有会话都注入的 system prompt overlay。各层只取**最后一个生效**的定义（不累加）。

```toml
[base_prompt]
text = "你是名为 defect 的助手"
# 或
file = "path/to/base.md"     # 相对该配置文件所在目录解析
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `file` | string | prompt 文件路径（相对配置文件目录） |
| `text` | string | 内联 prompt 文本 |

### `[prompt]` — 项目级 prompt

```toml
[prompt]
file = "AGENTS.md"           # 默认值；从 repo root 到 cwd 逐级拼接同名文件

[prompt.providers.anthropic]
text = "（仅 anthropic 生效的 overlay）"

[prompt.models]
"claude-opus-4" = "（仅该 model 生效的 overlay）"
```

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `file` | string | `"AGENTS.md"` | 项目 prompt 文件名 |
| `text` | string | — | 内联文本 |
| `providers.<name>.text` | string | — | per-provider overlay |
| `models.<id>` | string | — | per-model overlay |

---

## 6. `[capabilities]` — 全局能力

目前只有 hosted web search。本地 grep/glob 不属于能力层，由 `[tools.search]` 独立管理。

```toml
[capabilities.web_search]
mode = "delegate"    # delegate | disabled（默认 disabled）
```

| 字段 | 取值 | 默认 | 说明 |
|---|---|---|---|
| `web_search.mode` | `delegate` / `disabled` | `disabled` | `delegate` = 委托给 provider 托管的 web search（provider 不支持则会话启动失败） |

可在 provider 级覆盖：

```toml
[providers.anthropic.capabilities.web_search]
mode = "delegate"
```

provider 级不写则跟随全局设置。

---

## 7. `[turn]` — 单轮循环行为

```toml
[turn]
request_limit = 50
request_limit_mode = "adaptive"   # fixed | adaptive | unbounded
compact_threshold_tokens = 150000
compact_ratio = 0.85
compact_soft_ratio = 0.7
microcompact_ratio = 0.6
subagent_max_depth = 4
```

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `system_prompt` | string | — | 会话级 system prompt 覆盖 |
| `request_limit` | u32 | 自适应 32 | 单轮 LLM 请求上限的初始值 |
| `request_limit_mode` | `fixed` / `adaptive` / `unbounded` | `adaptive` | 见下；省略时裸 `request_limit = N` 即 `adaptive`，向后兼容 |
| `compact_threshold_tokens` | u64 | — | 上下文压缩的绝对 token 阈值 |
| `compact_ratio` | f64 | 0.85 | **硬**压缩水位（占 context_window 比例），同步阻塞压缩 |
| `background_compact_enabled` | bool | — | 是否启用后台全量压缩（超软水位时异步摘要，不阻塞当前轮） |
| `compact_soft_ratio` | f64 | 0.7 | **软**压缩水位，触发后台压缩 |
| `microcompact_enabled` | bool | — | 是否启用微压缩（清理旧轮超大 `tool_result`，不调 LLM） |
| `microcompact_ratio` | f64 | 0.6 | 微压缩水位 |
| `max_llm_retries` | u32 | — | LLM 重试上限 |
| `max_concurrent_tools` | usize | — | 工具并发执行数 |
| `max_hook_continues` | u32 | 3 | `before_turn_end` hook 强制续转的最大次数 |
| `subagent_max_depth` | u32 | 1 | 子 agent 垂直递归深度；默认 1 = 主 agent 能派子 agent 但子 agent 不能再派（常见的非递归策略），调大以支持"主 agent → 协调子 agent → 工作子 agent"这类嵌套编排；`0` = 禁止派发任何子 agent（顶层工具集不含 `spawn_agent`） |

**三档压缩水位约束**（违反则启动报错）：每个 ratio 必须在 `(0, 1]`，且 `microcompact_ratio ≤ compact_soft_ratio < compact_ratio`。

### `request_limit_mode` — 请求上限策略

`request_limit`（数字 N）与 `request_limit_mode` 组合决定单轮 LLM 调用次数的上限：

| mode | 含义 | N |
|---|---|---|
| `adaptive`（默认） | 起始 N，每成功执行一个工具就 +1，让确实在推进的轮次不被硬切断 | 必填 |
| `fixed` | 硬上限 N，不扩张 | 必填 |
| `unbounded` | 无上限 | 忽略 |

只写 `request_limit = N`（不写 mode）即 `adaptive`。`fixed` / `adaptive` 必须给 N，否则启动报错。

> goal 模式（`--goal`）下，单轮撞到 `request_limit` 不会让目标半途而废：会重开新一轮继续推进，由 `max_hook_continues` 控制总轮数上限。

---

## 8. `[tools.*]` — 本地工具

### `[tools.bash]`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `default_timeout_ms` | u64 | 30000 | 默认命令超时 |
| `max_timeout_ms` | u64 | 600000 | 超时上限 |
| `output_max_bytes` | usize | 1 MiB | 单条命令捕获的 stdout/stderr 合并上限，超出部分丢弃并计入 truncated（本地 shell backend；REPL / oneshot / ACP 本地模式都生效） |

### `[tools.fs]`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `read_default_limit` | u32 | 2000 | 默认读取行数 |
| `read_max_limit` | u32 | 5000 | 读取行数上限 |

### `[tools.fetch]`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `enabled` | bool | true | 是否注册 fetch 工具 |
| `default_timeout_secs` | u32 | 30 | 默认超时 |
| `max_timeout_secs` | u32 | 120 | 超时上限 |
| `max_response_bytes` | u64 | 5 MiB | 响应体上限 |
| `default_format` | `markdown`/`html`/`text` | `markdown` | 默认输出格式 |
| `html_to_markdown` | bool | true | HTML 转 Markdown |
| `follow_redirects` | bool | true | 跟随重定向 |

### `[tools.search]`

本地 `search`（grep/glob）参数，与 `[capabilities.web_search]` 完全独立，注册与否只看 `enabled`。

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `enabled` | bool | true | 是否注册 search 工具 |
| `default_head_limit` | u32 | 100 | 默认结果条数 |
| `max_head_limit` | u32 | 1000 | 结果条数上限 |
| `max_file_size_bytes` | u64 | 16 MiB | 单文件大小上限 |
| `max_result_bytes` | u64 | 256 KiB | 结果总字节上限 |
| `max_walk_files` | u64 | 100000 | 遍历文件数上限 |
| `respect_gitignore_default` | bool | true | 默认尊重 `.gitignore` |

### `[tools.background]`

后台子 agent 进度视图（主 agent 通过 `inspect_background_task` 看到的"最近 N 块"）。

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `default_recent_blocks` | usize | 10 | `inspect` 未带 `recent_blocks` 时返回的最近块数 |
| `block_text_limit` | usize | 0 | 单块正文字符上限；`0` = 只保留摘要/元数据 |
| `finished_tasks_cap` | usize | 64 | 任务表里保留的已完成任务条数上限，超出按完成顺序淘汰最旧的；限制长会话累积的内存占用（运行中的任务不计入，始终可查/可中断） |

---

## 9. `[sandbox]` — 权限模式

```toml
[sandbox]
mode = "ask-writes"    # read-only | ask-writes | open | deny-all（默认 ask-writes）
```

| 取值 | 含义 |
|---|---|
| `read-only` | 只读工具放行，写操作拒绝 |
| `ask-writes` | 写操作逐个询问（默认） |
| `open` | 全部放行（CI 友好） |
| `deny-all` | 全部拒绝 |

CLI `--sandbox` 覆盖此项；`--yolo` 等价 `--sandbox open`。注意 `--repl` 始终强制 `open`。

---

## 10. `[tracing]` — 可观测性

```toml
[tracing]
filter = "info,defect_agent=debug"   # tracing-subscriber EnvFilter
format = "text"                      # 日志输出格式：text | jsonl

[tracing.otlp]
endpoint = "http://localhost:4317"

[tracing.langfuse]
enabled = true
host = "https://cloud.langfuse.com"
public_key = "pk-..."
secret_key = "sk-..."
```

### `[tracing]`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `filter` | string | `info,toac=warn` | EnvFilter 表达式（也可用 `RUST_LOG` 环境变量，优先级最高） |
| `format` | `text` \| `jsonl` | `text` | 日志（stderr）输出格式；`jsonl` 每行一个 JSON 对象。也可用命令行 `--log-format` 覆盖。注意这是诊断日志，与控制 stdout 事件流的 `--format` 无关 |

### `[tracing.otlp]`

| 字段 | 类型 | 说明 |
|---|---|---|
| `endpoint` | string | OTLP 上报端点 |

### `[tracing.langfuse]`

默认关闭。若 `enabled = true` 但缺 key，启动时会打印警告并禁用上报。

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `enabled` | bool | false | 是否上报到 Langfuse |
| `host` | string | `https://cloud.langfuse.com` | Langfuse host |
| `public_key` | string | — | public key |
| `secret_key` | string | — | secret key |
| `flush_interval_ms` | u64 | 内建默认 | 刷新间隔 |
| `max_batch` | usize | 内建默认 | 单批最大事件数 |

---

## 11. `[mcp]` — MCP server

```toml
[mcp]
enabled_servers = ["fs", "github"]

[mcp.servers.fs]
transport = "stdio"
command = "mcp-server-fs"
args = ["--root", "/work"]
env = { LOG = "debug" }

[mcp.servers.github]
transport = "http"           # http | sse
url = "https://mcp.example.com/github"
headers = { Authorization = "Bearer ..." }
```

### `[mcp]`

| 字段 | 类型 | 说明 |
|---|---|---|
| `enabled_servers` | array<string> | 启用的 server 名；引用未定义的 server 会硬失败 |
| `servers.<name>` | table | server 定义 |

### `[mcp.servers.<name>]`

`transport` **必填**（不推断），并据此约束其余字段：

| 字段 | 类型 | 适用 transport | 说明 |
|---|---|---|---|
| `transport` | `stdio` / `http` / `sse` | — | 必填 |
| `command` | string | stdio（必填） | 子进程命令；http/sse 下**不允许** |
| `args` | array<string> | stdio | 命令参数；http/sse 下不允许 |
| `env` | table<string,string> | stdio | 环境变量 |
| `url` | string | http/sse（必填） | 远程端点；stdio 下**不允许** |
| `headers` | table<string,string> | http/sse | 自定义头；stdio 下不允许 |

> 所有 MCP 工具在会话里以 `mcp__<server>__<name>` 的名字出现（双下划线分隔），避免与内置工具撞名。server 名请只用字母、数字、`_`、`-`：模型 API 对工具名的字符集有要求（`^[a-zA-Z0-9_-]{1,128}$`），server 名或上游工具名里的点号、空格等会被上游拒绝。

### `.mcp.json`（生态标准格式）

除了 TOML `[mcp]`，defect 还读 **repo 根**的 `.mcp.json`（Claude Code / Cursor 的事实标准 schema）。仅在项目根查找（同 `.defect/` 的 git 根探测）。

```json
{
  "mcpServers": {
    "fs":   { "command": "npx", "args": ["-y", "@x/fs"], "env": { "ROOT": "/work" } },
    "docs": { "url": "https://example.com/mcp", "headers": { "x-key": "..." } }
  }
}
```

- **transport 自动推断**：有 `command` 即 stdio，有 `url` 即远程（默认 http，加 `"type": "sse"` 则 sse）。无需像 TOML `[mcp]` 那样显式写 `transport`——直接粘贴生态里的 `.mcp.json` 即可。
- **定义即启用**：写进 `.mcp.json` 的 server 直接生效，不需要再列白名单。
- **与 TOML 同名时 TOML 优先**：若某 server 在 `.mcp.json` 和 `[mcp.servers.<name>]` 都有定义，用 TOML 那份，并打印一条提示。
- 格式错误（未知字段、缺 `command`/`url`）会带文件路径报错。

---

## 12. `[http]` — HTTP 客户端栈

所有字段省略即用内建默认值。

```toml
[http]
total_timeout_ms = 600000
transport_retries = 2
initial_backoff_ms = 200

[http.proxy]
mode = "from-env"            # from-env | disabled | explicit
http_proxy = "http://proxy:8080"
https_proxy = "http://proxy:8080"
no_proxy = ["localhost", "127.0.0.1"]
```

### `[http]`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `total_timeout_ms` | u64 | 600s | 单请求总超时 |
| `transport_retries` | u8 | 2 | 传输层重试次数（不含首次）；`0` 禁用重试 |
| `initial_backoff_ms` | u64 | 200 | 重试初始退避 |
| `user_agent` | string | 编译期默认 | 覆盖 `User-Agent` |
| `proxy` | table | — | 代理子配置 |

### `[http.proxy]`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `mode` | `from-env` / `disabled` / `explicit` | `from-env` | `from-env` 读 `HTTP_PROXY`/`HTTPS_PROXY`；`explicit` 用下面字段 |
| `http_proxy` | string | — | 仅 `explicit` 生效 |
| `https_proxy` | string | — | 仅 `explicit` 生效 |
| `no_proxy` | array<string> | — | 仅 `explicit` 生效 |

---

## 13. `[[hooks.<event>]]` — hook 流水线

hook 用**数组语义**：`[hooks]` 是一个表，键是事件名，值是该事件下的条目数组。跨层合并是 **append + 去重**（而非 TOML 默认的整表覆盖），以防项目本地层静默删掉上游 hook。

```toml
# 在 before_tool_apply 事件上挂一个命令 hook
[[hooks.before_tool_apply]]
name = "fmt-check"
match = { tool_glob = "edit_*" }
handler = { type = "command", shell = "bash", command = "cargo fmt --check" }

# 挂一个 LLM prompt hook
[[hooks.before_turn_end]]
handler = { type = "prompt", system = "判断是否需要继续", render = { type = "json" } }

# 禁用上游某个 hook
[[hooks.disable]]
event = "before_tool_apply"
match = { tool_glob = "edit_*" }
handler = { type = "command", shell = "bash", command = "cargo fmt --check" }
```

### 合法事件名

事件名拼错会**硬失败**。完整列表（来自 `defect_agent::hooks::step::ALL_EVENT_NAMES`）：

`after_session_enter`、`after_turn_enter`、`before_ingest`、`after_ingest`、`before_compact`、`after_compact`、`before_generate`、`after_generate`、`before_permission`、`after_permission`、`before_tool_apply`、`after_tool_apply`、`after_tool_batch`、`before_turn_end`。

特殊键 `disable` 不是事件，而是禁用指令数组。

### hook 条目结构

| 字段 | 说明 |
|---|---|
| `name` | 可选显示名（仅 tracing/观测用，不参与去重/禁用匹配） |
| `match` | 事件匹配器（见下），省略则匹配该事件全部触发 |
| `handler` | 处理器（见下），必填 |

### `match` 匹配器

| 字段 | 说明 |
|---|---|
| `tool` | 工具名精确匹配（仅 `*ToolUse*` 类事件） |
| `tool_glob` | 工具名 glob 匹配 |
| `safety` | `SafetyClass` 过滤（仅 `PreToolUse`），任一命中即触发 |

### `handler` 处理器（`type` 区分三种）

**`type = "builtin"`** — 进程内 Rust 处理器：

| 字段 | 说明 |
|---|---|
| `name` | 内置处理器名（如 skill-manifest / skill-triggers） |

**`type = "command"`** — 外部命令。两种互斥形态：

- `argv` 形态：直接 spawn，不经 shell。
  - `argv`（必填，非空）、`argv_windows`（Windows 覆盖，省略则回退 `argv`）、`cwd`、`env`、`timeout_sec`
- `shell` 形态：经指定 shell 执行。
  - `shell`（必填：`sh`/`bash`/`pwsh`/`cmd`，或 `{ program = "...", args = [...] }` 自定义）、`command`（必填）、`cwd`、`env`、`timeout_sec`

不允许混用 `argv` 与 `shell`/`command`；`argv_windows` 只对 argv 形态有效。

**`type = "prompt"`** — 调用 LLM：

| 字段 | 说明 |
|---|---|
| `model` | 省略则用会话默认 model |
| `system` | system prompt（必填） |
| `render` | `{ type = "json" }`（直接喂 JSON 化的事件）或 `{ type = "template", template = "..." }`（handlebars 模板） |
| `timeout_sec` | 超时 |

---

## 14. Profile（子 agent）目录

子 agent 配置不在 `config.toml` 里，而是放在 `agents/` 目录，每个 profile 一项。`spawn_agent` 工具按 profile 的 `description` 让 LLM 选择派发对象。

- 项目层：`<repo-root>/.defect/agents/`
- 用户层：`$XDG_CONFIG_HOME/defect/agents/`（`--local` 模式跳过）
- 同名时项目层覆盖用户层；同名同时存在目录式与单文件式 = 硬失败。

两种格式：

**目录式** `agents/<name>/`：
```
agents/reviewer/
├── config.toml
└── system.md          # 默认 prompt 文件，可用 [prompt].file 改
```

`config.toml` 字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `description` | string | **必填**，供 `spawn_agent` 选择 |
| `model` | string | 子 agent model 覆盖；省略则继承父会话当前 model。也可写成 `[default] model`（与顶层 config 同键）——两者等价，但**不可同时设**，否则硬失败 |
| `prompt.file` | string | prompt 文件路径（相对 profile 目录，沙箱化防 `../` 逃逸）；默认 `system.md` |
| `prompt.text` | string | 内联 prompt 文本（与顶层 `[prompt] text` 一致）；与 `prompt.file` **互斥**，同设则硬失败。仅目录式可用 |
| `inherit_project_prompt` | bool | 默认 `false`。设为 `true` 时，子 agent 的系统提示会带上项目 `AGENTS.md`（build/测试/架构等项目约定），方便它了解所在项目；但不会带上主 agent 的身份提示。需要项目上下文的子 agent（如 code-reviewer）才开 |
| `tools.allow` | array<string> | 工具白名单（支持 **glob**，见下）；默认 `["read_file", "search"]` |
| `sampling.max_tokens` / `.temperature` / `.top_p` / `.top_k` | — | 采样参数；省略则沿用父会话的设置（含 `reasoning_effort`） |
| `request_limit` / `request_limit_mode` | u32 / enum | 该子 agent 单轮的请求上限，键与语义同顶层 `[turn]`。省略时为固定 32 次（不随父会话变化，避免子 agent 失控） |
| `[hooks.*]` | — | 该 profile 自己的 hook（不支持 `disable`，见下） |

**`tools.allow` 支持 glob**：每一项都是 glob（和 hook `tool_glob`、skill 触发同一套），写全名就是精确匹配。可以用 `mcp__ange__*` 一次放行某个 MCP server 的全部工具，或用 `read_*`。包括 MCP 工具在内的所有工具名都能匹配。若某个模式一个工具都没匹配到（比如 server 前缀拼错），会报错而不是静默放空。

> **子 agent 默认沿用主 agent 的配置**：上表之外的单轮行为（上下文压缩、重试、并发等）都继承父会话的 `[turn]` 设置。少数行为是固定的、不随父变化：子 agent 不会向用户弹权限确认（写操作直接拒绝）、不参与 `--goal` 循环、不能起后台任务、不能新增 MCP server（但能用父会话已连的）。

> profile 的 `[hooks]` 只声明这个子 agent 自己要跑的 hook，不支持 `disable`（`disable` 是用来删除*其他配置层*的 hook 的，而 profile 没有其他层）。

**单文件式** `agents/<name>.md`：frontmatter（`+++` 为 TOML，`---` 为 YAML，需开启 `yaml` feature）写上表字段，正文即 system prompt。单文件式不支持 `[prompt]` 表（prompt 就是正文）。

---

## 15. Skill 目录

技能放在 `skills/` 目录，每个技能一个子目录，必须含 `SKILL.md`。

- 项目层：`<repo-root>/.defect/skills/`
- 用户层：`$XDG_CONFIG_HOME/defect/skills/`（`--local` 模式跳过）
- 同名时项目层**整体替换**用户层（不合并）。

```
skills/my-skill/
├── SKILL.md           # 必填：frontmatter + 正文
├── scripts/           # 可选资源
└── refs/              # 可选资源
```

`SKILL.md` frontmatter 字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | string | **必填**，必须与目录名一致 |
| `description` | string | **必填**，L1 manifest 描述（软上限 200 字符，超出告警） |
| `always` | bool | `true` = 始终注入 system prompt |
| `triggers.globs` | array<string> | 文件 glob 自动激活 |
| `triggers.keywords` | array<string> | 提示关键词自动激活 |
| `allowed-tools` | array<string> | 预留给工具门禁 |

无效 glob / 缺 frontmatter / `name` 与目录名不符 = 硬失败。

---

## 16. 最小起步配置

### `defect init` — 自动生成全局配置

`defect init` 扫描环境里的 provider api-key（`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `DEEPSEEK_API_KEY`），用检测到的 key **实际调用该 provider 的 list-models API** 拉取真实模型列表，写入全局 `~/.config/defect/config.toml`。模型 id **绝不硬编码**——list-models 失败即硬失败，不回退猜测。

```bash
defect init                 # 交互式（inquire：多选 provider → 选默认 → 确认）
defect init --yes           # 非交互（CI）：检测到单个 key 时直接写
defect init --yes --default-provider deepseek   # 多个 key 时必须显式指定默认 provider
defect init --default-model deepseek-v4-pro     # 默认模型（须在 live 列表内，否则报错）
defect init --force         # 覆盖已存在的全局配置
```

- 多个 key 且 `--yes` 时**必须**给 `--default-provider`——defect 不替用户从"哪个 key 恰好存在"猜默认 provider。
- 交互 prompt 需 `init` 编译 feature（默认开）；`--yes` 非交互路径不依赖该 feature。
- Bedrock 不在 init 范围内（走 AWS 凭证链、无单一 key、list-models 不打 API）。

### 手写最小配置

放在 `~/.config/defect/config.toml` 或 `<repo-root>/.defect/config.local.toml`：

```toml
[default]
provider = "deepseek"
model = "deepseek-chat"

[providers.deepseek]
api_key_env = "DEEPSEEK_API_KEY"   # 实际 key 放环境变量

# 全局 system prompt overlay，所有会话生效
[base_prompt]
text = "你是名为 defect 的助手"

# 项目级 prompt：默认从 repo root 到 cwd 逐级拼接 AGENTS.md
[prompt]
file = "AGENTS.md"
```

运行：

```bash
export DEEPSEEK_API_KEY=sk-...
defect --provider deepseek
```
