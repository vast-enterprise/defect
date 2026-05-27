# `fetch` 内置工具设计

`fetch` 是 defect 的网络读取工具：拉一个 URL、按格式（markdown / html / text）渲染、按超时与大小上限保护。本文沉淀工具的形状、与 ACP 的对位、配置层入口、与 [`bash`] / [`fs`] 的边界，以及 P1 故意不做的部分。

设计原则按依赖顺序：

1. **`fetch` 是本地 [`Tool`]，不是 capability**——它的语义是「执行一次可控的网络读取」，本质更像 `fs.read` 的网络版，不是「模型是否拥有外部检索能力」这一类协商问题。详见 [`capabilities.md` §13.6](./capabilities.md)。
2. **以 ACP 为导向**——产出的字段直接对位 [`ToolCallUpdateFields`] / [`ToolCallContent`]，复用 [`ToolKind::Fetch`]。
3. **HTTP 栈复用 [`defect-http`]**——超时、重试、代理、UA 头部都从 [`HttpClientConfig`] 拿，不在 fetch 工具内重新做一份。
4. **不围绕 hosted fetch 建模**——即便 Anthropic 有 `web_fetch` hosted tool，本工具的主架构也不为它服务；hosted fetch 不是 P1/P2 的必需项，演进口子见 §8。

[`ToolCallUpdateFields`]: https://docs.rs/agent-client-protocol-schema/0.13/agent_client_protocol_schema/v1/struct.ToolCallUpdateFields.html
[`ToolCallContent`]: https://docs.rs/agent-client-protocol-schema/0.13/agent_client_protocol_schema/v1/enum.ToolCallContent.html
[`ToolKind::Fetch`]: https://docs.rs/agent-client-protocol-schema/0.13/agent_client_protocol_schema/v1/enum.ToolKind.html
[`Tool`]: ./tool-trait.md
[`bash`]: ./tools-bash.md
[`fs`]: ./tools-fs.md
[`defect-http`]: ../../crates/http/
[`HttpClientConfig`]: ./config.md

---

## 1. P1 实装现状

| 部件 | 状态 |
|---|---|
| [`FetchToolConfig`] schema（`enabled` / 超时 / 大小 / 格式 / `html_to_markdown` / `follow_redirects`） | ✅ |
| `[tools.fetch]` 段解析 + `EffectiveConfig.tools.fetch` 字段 | ✅ |
| `Tool` 实现（`crates/tools/src/fetch/`） | ❌ |
| `defect-http` 客户端按 [`HttpClientConfig`] 装配 | ✅（基础栈） |
| `html` → `markdown` 渲染管线 | ❌（依赖待定） |
| ACP `ToolKind::Fetch` 路径 | 未连通（工具未实装） |
| MCP 同名 `fetch` 命名空间化（`mcp.<server>.fetch`） | ✅（与所有 MCP 工具一致） |

**P1 = schema-only**：本地 `fetch` 配置可解析、`EffectiveConfig` 可读取；但运行时没有 `FetchTool` 实例可注册到 [`ToolRegistry`]。LLM 在 P1 看不到 `fetch` 工具——MCP server 暴露的 `fetch` 仍走 `mcp.<server>.fetch` 注册名（[`capabilities.md` §6.2](./capabilities.md)）。

本文按"P2 落地后该长什么样"写。所有 §3–§6 的 trait / schema / execute 细节是设计契约，不是 P1 实装。需要看 P1 已落地的部分参见 §7（配置）与 §9（演进口子）。

[`FetchToolConfig`]: ../../crates/config/src/types.rs
[`ToolRegistry`]: ./session.md

---

## 2. 为什么 `fetch` 不是 capability

[`capabilities.md`](./capabilities.md) 把 `search` 抽成 capability，原因是 provider-hosted search 与 local search 的执行机制根本不同——hosted 是模型直接拿到外部检索结果、agent 无法逐次拦截。`fetch` 不存在这个张力：

| 维度 | `search` | `fetch` |
|---|---|---|
| provider-hosted 实装是否对称 | Anthropic / OpenAI 都有原生 hosted search | 仅 Anthropic 有 `web_fetch`；OpenAI、DeepSeek 没有 |
| 行为可控性 | hosted 是 provider 黑盒 | URL / 超时 / 大小 / 格式都需要客户端可调 |
| 审批模型 | hosted 由 provider 服务端执行，无 call site | 本地工具天然支持 [`SafetyClass`] |
| 与 [`Tool`] trait 的契合度 | 低（execute 时序对不上） | 高（`fetch` 就是异步 IO + 输出） |

结论：

> hosted fetch 不是当前设计中心，也不是 P1/P2 的必需项。

本工具的主架构由本地 HTTP fetch 决定。如果以后要支持 hosted `web_fetch`，应单独立项（参考 [`capabilities.md`](./capabilities.md) `HostedCapabilities` 的延展模式：先加 `fetch: bool` 字段、再加 `[capabilities.fetch]` mode），而不是反向污染本工具的 schema。

[`SafetyClass`]: ./tool-trait.md

---

## 3. 工具名片

```rust
ToolSchema {
    name: "fetch".to_string(),
    description: "Fetch a URL and return its content. \
                  Supports HTTP/HTTPS only. Renders HTML to markdown by default; \
                  raw HTML / plain text via `format`. Times out after `timeout_secs` \
                  (default 30s; max configurable via [tools.fetch]). \
                  Truncates responses larger than `max_response_bytes`.".to_string(),
    input_schema: json!({
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Absolute http:// or https:// URL. \
                                Other schemes are rejected."
            },
            "format": {
                "type": "string",
                "enum": ["markdown", "html", "text"],
                "description": "Output format. Defaults to the `default_format` \
                                configured in [tools.fetch] (markdown out of the box). \
                                `markdown` runs the html→markdown pipeline; \
                                `html` returns raw HTML; `text` strips tags but keeps text."
            },
            "timeout_secs": {
                "type": "integer",
                "minimum": 1,
                "description": "Per-call timeout in seconds. Defaults to \
                                `default_timeout_secs` from [tools.fetch]. \
                                Capped by `max_timeout_secs`."
            }
        },
        "required": ["url"]
    }),
}
```

字段取舍：

- **只支持 `http` / `https`**——`file://` / `data:` / `ftp://` 都拒，避免 LLM 用 `file:///etc/passwd` 绕过 [`fs`] 的工作区边界。
- **没有 `method` / `headers` / `body`**——P2 只做 `GET`。`POST` / 自定义 header 等到「LLM 真要做 API 调用」场景出现时再加，避免 P2 工具表面就声明无人使用的口子。LLM 想做 RESTful 调用现在仍走 [`bash`] (`curl`)。
- **没有 `auth` / cookie**——同上。auth 信息进 LLM 上下文是泄漏隐患，P2 不在 schema 上开这个口子。
- **没有 `follow_redirects` 单次覆盖**——重定向策略由 [`tools.fetch.follow_redirects`] 统一定，不让 LLM 单次开关。理由：避免「LLM 因为某次失败把 follow 关掉、后续 turn 也跟着关」的状态污染。
- **`format` 默认从配置读**——`tools.fetch.default_format = "markdown"` 时不传 `format` 即得 markdown。这是 §7 配置的语义。

[`tools.fetch.follow_redirects`]: #7-配置入口

---

## 4. 安全等级（`safety_hint`）

```rust
fn safety_hint(&self, _args: &Value) -> SafetyClass {
    SafetyClass::ReadOnly
}
```

**一律返回 `ReadOnly`**——`fetch` 不写本地状态，副作用全部是「向外发 HTTP 请求」。理由与 [`fs::read_file`](./tools-fs.md) 同：本地世界没变化，配合 [`ReadOnlyPolicy`] 让"只读模式"用户能跑这个工具。

外部副作用（POST 引发的服务端写入、计费等）由两条边界兜底：

1. P2 schema 不暴露 `POST` / 自定义 method（§3）——`GET` 是 RFC 定义的 safe / idempotent。
2. URL 本身的 destructive 性质（例如某 `GET` 触发了服务端 destructive 操作）属于服务端 API 设计问题，工具层不解析 URL 推断；用户用 [`AskWritesPolicy`] / [`SandboxPolicy`] 在 session 层面控制网络访问。

未来引入 `POST` / `PUT` / `DELETE` 时（不在 P2），`safety_hint` 改成按 method 分类：`GET` / `HEAD` 仍 `ReadOnly`，`POST` / `PUT` / `PATCH` → `Mutating`，`DELETE` → `Destructive`。届时 `[tools.fetch]` 加 `allowed_methods` 字段即可。

[`ReadOnlyPolicy`]: ./sandbox-policy.md
[`AskWritesPolicy`]: ./sandbox-policy.md
[`SandboxPolicy`]: ./sandbox-policy.md

---

## 5. `describe(args)`：UI 自描述

```rust
ToolCallUpdateFields {
    title: Some(format!("Fetch {}", truncate(url, 80))),
    kind:  Some(ToolKind::Fetch),
    locations: None,         // URL 不是文件路径，不填 locations
    content:   None,         // 执行期填
    raw_input: None,         // 主循环填
    raw_output: None,        // 终态填
    status:    None,         // 主循环填
}
```

- `title` 用 `"Fetch <url>"`，截到 80 字符。客户端 UI 一眼看出"这是一次网络读取"。
- `kind = Fetch` 命中 ACP 既有的语义。
- 不填 `locations`——`ToolCallLocation::path` 是文件路径，URL 不是；客户端的 follow-along 对 URL 没用武之地。

---

## 6. `execute`

```text
                ctx.http.get(url)                  // defect-http 客户端
                  ─ timeout / proxy / UA / retry from HttpClientConfig
                  ─ stream body chunks
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
        body chunks         content-type         status code
            │                   │                   │
            └────────► accumulate body ◄────────────┘
                                │   (cap at max_response_bytes; overflow tracked)
                                ▼
                      render_by_format(buf, content_type, format)
                                │
            ┌──────────┬────────┴────────┬──────────┐
            ▼          ▼                 ▼          ▼
         status<400  status>=400       cancel    timeout/network
            │          │                 │          │
            ▼          ▼                 ▼          ▼
       Completed   Completed(           Failed(    Failed(
       (content:    content with        Canceled)  Execution(io_err))
        Text(rendered),                  )
        raw_output)
```

P2 只发**一帧** `Completed`（不发中间 `Progress`），与 [`bash`](./tools-bash.md) 同款理由——ACP `ToolCallUpdateFields::content` 是 *replace* 语义，对 5 MiB 量级 body 多帧 Progress 等于 `O(N²)` 字节。

### 6.1 HTTP 栈

`fetch` 不直接 `reqwest::Client::new()`。它走 [`ToolContext::http`](./tool-trait.md)（P2 引入的字段，与 `ctx.fs` / `ctx.shell` 同形态），拿到 session 级 `Arc<dyn HttpClient>`。装配责任在 [`AgentCore::create_session`]：

- 客户端配置由 [`HttpClientConfig`] 决定——超时（`timeout_ms`）、代理（`proxy`）、重试策略（`retry_max_attempts` / `retry_initial_backoff_ms`）、UA（`user_agent`）。
- 每个 session 一个客户端实例（连接池复用）；`fetch` 工具不感知客户端构造细节。
- 单次调用的 `timeout_secs` 覆盖 `HttpClientConfig::timeout_ms`，但不能超过 `tools.fetch.max_timeout_secs`（超就 clamp 到上限，并在 raw_output 标记）。

[`HttpClientConfig`] 同时被 [`McpHttpTransport`] / [`AnthropicAdapter`] / [`OpenAiAdapter`] 共用——`fetch` 与上层 LLM / MCP 走同一条 HTTP 栈，不重复造轮子。

[`AgentCore::create_session`]: ../../crates/agent/src/session/default.rs
[`McpHttpTransport`]: ../../crates/mcp/

### 6.2 输出渲染

| `format` | content-type | 行为 |
|---|---|---|
| `markdown`（默认） | `text/html` / `application/xhtml+xml` | 走 html→markdown 管线（§6.3） |
| `markdown` | `text/markdown` / `text/plain` / 其他 text/* | 直接返回原文 |
| `markdown` | 二进制 / 未知 | `Failed(Execution("unsupported content-type for markdown format: <ct>"))` |
| `html` | `text/html` / `application/xhtml+xml` | 返回原始 HTML |
| `html` | 非 HTML | `Failed(Execution("not HTML: <ct>"))` |
| `text` | 任意 text/* | 返回原文 |
| `text` | `text/html` | 抽出 `<body>` 文本（去标签） |
| `text` | 二进制 | `Failed(Execution("binary content-type: <ct>"))` |

`fetch` 不返回二进制——defect 当前 ACP 路径是 `Content::Text`，没有 image / blob payload 的对位。需要图片拉取等 LLM 走多模态接口（不在 P2 范围）。

### 6.3 html → markdown 管线

P2 选型未定。两条候选：

- **[`htmd`](https://crates.io/crates/htmd)**——纯 Rust，零外部依赖，配置简单。
- **[`html2md`](https://crates.io/crates/html2md)**——成熟度高，但维护活跃度待观察。

不在 P2 设计阶段固化 crate；落地 PR 时再做 spike。两条硬约束：

1. **`tools.fetch.html_to_markdown = false` 时 bypass**——用户能在不要 markdown 的场景关掉整个管线（节省 CPU、避免渲染歧义）。`format = "markdown"` + `html_to_markdown = false` 时退化为「返回原始 HTML 但 wrap 一句 warning」。
2. **失败 → fail loud**——html parse / convert 抛错时返回 `Failed(Execution("html-to-markdown conversion failed: <reason>"))`；不静默回退到原始 HTML，否则 LLM 会拿到与 schema 不一致的输出。

### 6.4 大小 / 重定向 / 取消

- **`max_response_bytes`**（P2 默认 5 MiB）——超过即截断，content 末尾追加 `\n[response truncated; remaining N bytes dropped]`，`raw_output.truncated = true`。理由与 [`bash`](./tools-bash.md) 1 MiB 上限同：保护 LLM context、保护 agent 进程内存。
- **`follow_redirects`**（P2 默认 `true`）——由 [`HttpClientConfig`] 决定 redirect policy；fetch 不做单次覆盖（§3）。redirect 链超长（reqwest 默认 10 跳）时报 `Failed(Execution(io_err))`，错误信息含跳数。
- **取消**：`ctx.cancel.cancelled()` 中断 body stream，drop `Response` 让连接归还连接池；event = `Failed(ToolError::Canceled)`。

### 6.5 终态

```rust
struct FetchOutput {
    status: u16,
    content_type: Option<String>,
    bytes_received: u64,
    bytes_returned: u64,
    truncated: bool,
    redirects: u32,
    elapsed_ms: u64,
}
```

`raw_output = serde_json::to_value(FetchOutput { ... })`——给 LLM / 客户端机读字段。

终态映射规则：

| 退出形态 | event | 说明 |
|---|---|---|
| `2xx` | `Completed` | content 是渲染后的正文 |
| `3xx`（被 follow 完不再有最终 2xx——异常） | `Completed` | content 含响应体（可能为空），raw_output.status 标终态 |
| `4xx` / `5xx` | `Completed`（**仍然是 Completed**） | content 末尾追加 `[http status: N]`，raw_output 含 status |
| 网络错误（DNS / connect / TLS） | `Failed(Execution(io_err))` | 错误信息含原因 |
| 超时 | `Failed(Execution("timed out after Xs"))` | — |
| body 解析失败（content-type 不匹配 format） | `Failed(Execution(...))` | 见 §6.2 |
| `ctx.cancel` 触发 | `Failed(Canceled)` | — |
| URL scheme 非 http/https | `Failed(InvalidArgs(...))` | 喂回 LLM 改 args |

为什么 `4xx` / `5xx` 是 `Completed` 而不是 `Failed`：与 [`bash`](./tools-bash.md) 非零 exit code 同款理由——HTTP 错误码是**业务结果**，不是工具调用失败。LLM 看到 status 后能自己决定下一步（重试 / 换 URL / 报告用户）。只有"agent 自身没法跑这次请求"才走 `Failed`。

---

## 7. 配置入口

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

对位 [`FetchToolConfig`]：

```rust
#[non_exhaustive]
pub struct FetchToolConfig {
    pub enabled: bool,
    pub default_timeout_secs: u32,
    pub max_timeout_secs: u32,
    pub max_response_bytes: u64,
    pub default_format: FetchFormat,
    pub html_to_markdown: bool,
    pub follow_redirects: bool,
}

#[derive(Default)]
pub enum FetchFormat {
    #[default]
    Markdown,
    Html,
    Text,
}
```

字段语义：

- **`enabled`**——`false` 时本地 `fetch` 工具不注册到 [`ToolRegistry`]。MCP `fetch` 仍走 `mcp.<server>.fetch` 命名空间，不受影响（[`capabilities.md` §6.2](./capabilities.md)）。
- **`default_timeout_secs` / `max_timeout_secs`**——单位秒。LLM 传 `timeout_secs` 超过 `max` 时 clamp 到 `max` 并在 raw_output 标记，不报错。
- **`max_response_bytes`**——硬截断。
- **`default_format`**——LLM 不传 `format` 时使用。
- **`html_to_markdown`**——见 §6.3。
- **`follow_redirects`**——传给 [`HttpClientConfig`] 的 redirect policy；具体跳数上限 reqwest 默认 10。

### 7.1 没有 per-provider 覆写

P1 / P2 **不**支持 `[providers.<p>.tools.fetch]`。`fetch` 是全局本地工具，per-provider 启停或参数差异在当前没有真实需求——同一份 `fetch` 在 Anthropic 与 OpenAI 下行为应一致。

真出现 per-provider 差异时再开口子（缺省值不变，不算 breaking）。这条同时也避免任何「per-provider 启用 hosted fetch」的暗示——hosted fetch 不是当前设计中心，参见 §2 / §9。

### 7.2 Inactive section

P1 schema 已落地但工具未实装，`[tools.fetch]` 段在 P1 完整解析、`EffectiveConfig.tools.fetch` 字段可读、但 LLM 看不到 `fetch` 工具。这段时间内**不**发 `ConfigWarning::InactiveSection`——理由：用户写 `[tools.fetch]` 是为 P2 上线后立即生效准备的，发 warning 反而误导。P2 落地后这段消失（只要 `enabled = true` 就生效）。

---

## 8. 与 [`bash`] / [`fs`] / `search` 的边界

| 操作 | 用哪个工具 | 理由 |
|---|---|---|
| `GET https://example.com` | `fetch` | 本工具的核心场景 |
| `curl -X POST ...` | `bash` (P2) | P2 只做 `GET`；POST 等扩展上线再说 |
| `git clone https://...` | `bash` | git 不是「读 URL」语义；走 shell |
| 读本地 `./README.md` | `fs.read_file` | URL 不接 `file://`；本地文件归 fs |
| "搜一下 X" → 多个候选页 | `search` capability | 检索 vs 拉取一个具体 URL；详见 [`capabilities.md`](./capabilities.md) |
| "搜完后拉第一个结果" | `search` + `fetch` | 模型先拿 sources，再逐个 `fetch` |

`fetch` 与 `search` 是互补关系——`search` 给 URL 列表，`fetch` 把单个 URL 拉下来精读。两者不竞争、不替代。

---

## 9. P2 不做（演进口子）

下列每条都是诚实的「feature gap」，当前要么 fail loud（schema 拒绝）要么走 [`bash`]，**不会**静默走错路径。

- **`POST` / `PUT` / `DELETE` / 自定义 header / body**：P2 schema 不暴露。LLM 想做 API 调用走 `bash("curl ...")`。引入时机：用户出现「LLM 真的需要写 RESTful API」的明确需求；schema 加 `method` / `headers` / `body` 字段，`safety_hint` 改为按 method 分类（§4）。
- **auth / cookie 注入**：P2 不做。auth secret 进入 LLM 上下文是泄漏，引入时需要先有 secret 管理层（不让 secret 流到 prompt）。
- **hosted fetch 来源**：P2 不接 Anthropic `web_fetch_*`。如要加，按 `search` 的 capability 模式扩展——`HostedCapabilities { search: bool, fetch: bool }`、`[capabilities.fetch]` mode、装配期裁决——而不是塞进本工具的 schema。详见 [`capabilities.md`](./capabilities.md) §13.6 / §14.3。
- **流式 body 渲染**：P2 一次性 `Completed`。需要「边下载边在客户端滚」时，与 [`bash`](./tools-bash.md) §8 一并解决——等 ACP `content` 加 append 语义、或引入 `Terminal` 形态时再做。
- **结构化 / 二进制 payload**：P2 拒绝二进制 content-type。多模态拉取等 ACP 协议演进出 `read_resource` / `Image` content kind 后再上。
- **fetch 缓存 / etag**：P2 每次都打网络。缓存层引入需要先确定缓存目录的生命周期（session 内 vs 跨 session），与 [`bash`](./tools-bash.md) 的 spill-to-disk 同期解决。
- **robots.txt / 礼貌限流**：P2 不读 robots、不主动限速。等用户出现真实抓取场景再加（届时同时考虑加 `[tools.fetch.robots_policy]` 配置）。
- **html→markdown 引擎切换**：P2 选定一个 crate 后，schema 不暴露引擎选择。多引擎共存的成本 > 收益。

---

## 10. 测试矩阵（P2 落地时）

每条都写成 `#[tokio::test]`，放在 `crates/tools/src/fetch/tests.rs`。所有外网调用走 [`wiremock`] 或 `httpbin` mock，不打真实公网。

| # | 场景 | 验证 |
|---|---|---|
| 1 | `GET https://mock/200` 返回 `text/markdown` | event = Completed；content = body 原文；raw_output.status = 200 |
| 2 | `GET https://mock/200` 返回 `text/html`，format=markdown | content 是 markdown；调过 html→markdown 管线 |
| 3 | `format=html`，content-type 是 markdown | event = `Failed(Execution("not HTML"))` |
| 4 | `format=text`，content-type 是 html | content 不含 `<` 标签 |
| 5 | `GET https://mock/404` | event = Completed；content 末尾含 `[http status: 404]`；raw_output.status = 404 |
| 6 | `GET https://mock/500` | 同 #5 |
| 7 | `GET file:///etc/passwd` | event = `Failed(InvalidArgs)`；scheme 拒绝 |
| 8 | 网络错误（DNS 失败 / connect refused） | event = `Failed(Execution(io_err))` |
| 9 | timeout（mock 5s 延迟，timeout_secs=1） | event = `Failed(Execution("timed out"))` |
| 10 | response > `max_response_bytes` | event = Completed；content 末尾含 `[response truncated]`；raw_output.truncated = true |
| 11 | redirect 5 跳到 200 | event = Completed；raw_output.redirects = 5 |
| 12 | redirect 链超 reqwest 默认 10 跳 | event = `Failed(Execution(io_err))` |
| 13 | `tools.fetch.follow_redirects = false` + 302 | event = Completed；status = 302；不跟随 |
| 14 | `tools.fetch.html_to_markdown = false` + format=markdown + html body | content 是原始 HTML + warning 注释 |
| 15 | html→markdown 管线 panic（mock 返回畸形 html） | event = `Failed(Execution("html-to-markdown conversion failed"))` |
| 16 | `ctx.cancel` 在 body 读取中触发 | event = `Failed(Canceled)`；连接归还 |
| 17 | `tools.fetch.enabled = false` | `FetchTool` 未注册到 [`ToolRegistry`]；LLM 看不到 fetch schema |
| 18 | 单次 `timeout_secs` > `max_timeout_secs` | clamp 到 max；raw_output 标记（不报错） |
| 19 | content-type 是 `image/png` 等二进制 | event = `Failed(Execution("binary content-type"))` |
| 20 | 真实 e2e：deepseek prompt "fetch https://example.org" → `fetch` → 总结 | TurnEnded = `EndTurn`；至少一次 ToolCallStarted/Finished |

#7（scheme 越界）/ #10（大小截断）/ #15（管线失败）是 §1 设计原则 4「不留坑」的回归基线。

[`wiremock`]: https://crates.io/crates/wiremock

---

## 11. 落地节奏（P2 时）

1. **`crates/tools/src/fetch/`**（新模块）——`FetchTool` + html→markdown 渲染：
   - `mod.rs`：`pub struct FetchTool` 实现 [`Tool`]；私有子模块 `render.rs`（§6.2 / §6.3）、`schema.rs`（§3 名片）。
   - `tests.rs`：§10 全部用例。
2. **`crates/agent/src/tool.rs`**——[`ToolContext`] 加 `http: &dyn HttpClient` 字段；[`ToolContext::new`] 签名调整。
3. **`crates/agent/src/session/`**——`AgentCore::create_session` 装配 session 级 `Arc<dyn HttpClient>`（按 [`HttpClientConfig`] 构造）；`TurnRunner` 把 `http` 注入 [`ToolContext`]。
4. **`crates/cli/`**——默认装配把 `FetchTool` 塞进 [`DefaultAgentCoreBuilder`]；e2e example 注册 fetch，verify §10 #20。
5. **更新 [`TODO.MD`](../../TODO.MD)**——`fetch` 工具行翻到「已完成」。
6. **更新 [`tools-fs.md`](./tools-fs.md) / [`tools-bash.md`](./tools-bash.md) 的 §1 总览**——fs / bash / fetch 三件套共同出现在 `Tool` trait 承载列表里。

---

## 12. 决议

1. `fetch` 是本地 [`Tool`]，不是 capability
2. P2 只做 `GET`；schema 不暴露 method / header / body / auth
3. HTTP 栈复用 [`HttpClientConfig`] / [`defect-http`]，不在工具内重做
4. `format` 三态：`markdown`（默认，html→markdown 渲染） / `html`（原始） / `text`（去标签）
5. hosted fetch 不进 P1/P2；如要支持按 `search` capability 模式扩展，不污染本工具
6. P1 当前是 schema-only：[`FetchToolConfig`] 落地、`Tool` 实现待 P2
