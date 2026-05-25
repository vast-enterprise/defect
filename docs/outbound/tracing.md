# Tracing 接入设计

`defect` 跑在 stdio ACP 模式下，stdout 是协议线、stderr 是日志线。`tracing` 用作
**唯一**的结构化日志通道——错误处理、调试输出、生产排障都走它。本文沉淀：

1. 各 crate 在哪些位置发 span / event、字段叫什么
2. CLI 如何初始化 subscriber，以及"stdio 不能污染 stdout" 这一约束如何落地
3. 与 [`docs/internal/errors.md`](../internal/errors.md) 的衔接：错误的诊断字段
   如何不被重复打印
4. OTLP exporter 留口（P2，本文只画接口）

## 1. 设计原则

1. **Library crate 只发 event / span，不安装 subscriber**。`tracing` 自己的
   规则——只有 binary（`defect-cli`）能调 `tracing_subscriber::fmt().init()`。
   Lib 安装 subscriber 会让宿主应用没法接管输出。
2. **永远 `with_writer(std::io::stderr)`**。stdio ACP 模式下 stdout 是 wire；
   `init()` 默认写 stdout 会把 SSE / JSON-RPC 帧夹进日志，客户端解码必炸。
3. **错误产生处不要 `tracing::error!`**。错误是 Rust 类型，由调用方在边界
   决定如何上日志。这条来自 [`docs/internal/errors.md`](../internal/errors.md) §8。
   多次打印同一个错误是常见污染源。
4. **结构化字段优于 `format!`**。`tracing::warn!(tool = %name, count = n, "...")`
   比 `tracing::warn!("name={name} count={count}")` 可被结构化 sink 直接
   消费、grep 起来也更稳定。
5. **`%` 用于 `Display`、`?` 用于 `Debug`、`= value` 用于直接写入**。`%err`
   走 `Display`（短文案），`?err` 走 `Debug`（含 source chain）。错误对象
   通常用 `%err` 打 message + 让 source chain 跟随 [`tracing-error`] feature
   保留——本 v0 不引入 `tracing-error`，单纯 `%err` 已足够。
6. **span 顺着调用链下传，不要起新的 root span**。一个 turn 是一棵树：
   `prompt_turn` → `llm_call` → `tool_call`。下游 crate 不应该 `info_span!`
   自起无 parent 的 root span，否则父子关系断裂。

## 2. 模块约定

下表规定每个 crate 的 instrument 边界。新加 instrument 点对照本表。

### 2.1 `defect-acp`

| 位置 | 形态 | 字段 |
|------|------|------|
| `serve_on` 顶层 | `info_span!("acp_serve")` | `transport`（`stdio` / `channel`） |
| `run_prompt_turn` | `info_span!("acp_prompt_turn")` 包裹 | `session_id`（短哈希） |
| handler `initialize` | `tracing::info!` 单条 | `version` |
| handler `session/new` | `tracing::info!` | `cwd` |
| handler `session/cancel` | `tracing::info!` | `session_id` |
| `spawn_permission_request` | `info_span!("acp_request_permission")` | `session_id`, `tool_call_id` |
| 错误返回前 | `tracing::warn!` 一次 | `?err`，**不要在 ACP wire 投错前再 `error!`** |

边界：**这一层是日志的最后一站**。下游 `TurnError` / `ProviderError` 在被
投影成 `AcpError::into_wire_error` 后，由 `respond_with_error` 上层
`run_prompt_turn` 用 `tracing::warn!(?err)` 打一条结构化日志即可。

### 2.2 `defect-agent`

| 位置 | 形态 | 字段 |
|------|------|------|
| `Session::run_turn` 主循环 | `info_span!("turn")` 包裹整个 future | `session_id`, `turn_idx` |
| 每次 LLM 调用 | `info_span!("llm_call")` | `vendor`, `model`, `attempt` |
| 每个 tool 执行 | `info_span!("tool_call")` | `tool`, `tool_call_id` |
| permission wait | `info_span!("await_permission")` | `tool_call_id` |
| context compaction | `tracing::info!` | `tokens_before`, `tokens_after` |
| retry 决策 | `tracing::info!` | `?retry_hint`, `attempt`, `next_attempt_in` |

`#[tracing::instrument(skip_all, fields(...))]` 是首选写法——`skip_all` 防止
默认把 `&self` / 参数 `?{:?}` 全打出来（噪音 + 可能含密钥）。需要的字段在
`fields(...)` 显式声明。

### 2.3 `defect-llm`

由 toac 生成的 wire 代码自动 emit `INFO toac: request uri=... headers=...`
事件——已经存在。**问题点**：headers 里含 `authorization: Bearer <key>`，
日志会打出明文。

→ 在 `defect-llm` 里加 *redacting wrapper*：从 toac wire 出来的请求经过
`AuthorizationRedactor` 中间层（或者在 SSE / HTTP 层套一层 fmt layer 把
`authorization` header 替换为 `Bearer <redacted>`）。具体实现位置见 §5.2。

provider crate 自己再加：

| 位置 | 形态 | 字段 |
|------|------|------|
| `LlmProvider::send` 入口 | `info_span!("provider_send")` | `vendor`, `endpoint`, `model`, `request_id` (resp) |
| 流首次出 chunk | `tracing::debug!` | `kind`（`text` / `tool_use` / 等） |
| 流终止 | `tracing::debug!` | `?stop_reason`, `usage`（如有） |
| 解析失败 | `tracing::warn!` | `?err`, `body_excerpt`（截断到 256 字节） |

### 2.4 `defect-tools`（待实现）

| 位置 | 形态 | 字段 |
|------|------|------|
| `Tool::execute` 入口 | 由调用方（turn 主循环）已经包了 `tool_call` span，工具实现内部不再起 span，只发 event |
| fs.read 进入 | `tracing::debug!` | `path`, `bytes`（结果） |
| fs.write 进入 | `tracing::info!` | `path`, `bytes_written` |
| bash spawn | `tracing::info!` | `command_first_word`, `pid` |
| bash 异常退出 | `tracing::warn!` | `exit_code`, `signal` |

### 2.5 `defect-cli`

只做 subscriber init + 装配。详见 §3。

## 3. Subscriber 初始化（CLI 边界）

CLI `main.rs` 与 examples 现在各自有一份 subscriber init。统一形态：

```rust
fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .init();
}
```

要点：

- `with_writer(std::io::stderr)` —— §1 第 2 条
- `with_target(true)` —— 输出 `defect_agent::session::turn` 这样的 module path，
  排障时知道日志来自哪
- `with_ansi(stderr.is_terminal())` —— 终端跑时上色，被 redirect / pipe 时
  关闭，避免日志文件里塞 ANSI 转义
- `EnvFilter` 默认 `info`，由 `RUST_LOG` 覆盖

实现位置：抽成 `defect-cli` 里的 `tracing_init.rs`（或同等私有 fn），
`main.rs` 与 `examples/deepseek_e2e.rs` 共用。examples 不能 `use`
binary crate 的私有 fn，所以这条会成为 `pub fn` 暴露在 `defect-cli` lib
形态——或者更简单：放进 `defect-cli` 的 `examples/common/mod.rs`（与
`crates/llm/examples/common/mod.rs` 同款形式）。

**v0 选 `examples/common/mod.rs`**——避免给 binary crate 加 lib target。

## 4. EnvFilter 与默认级别

默认 `info`，可被 `RUST_LOG` 覆盖。约定：

| 级别 | 用途 |
|------|------|
| `error` | 不可恢复的失败（subscriber init 失败这种本身不会上日志的另算） |
| `warn` | 偏离正常路径但已容错处理（bad provider response、permission cancelled、tool task panic） |
| `info` | turn 开始 / 结束、provider 装配、subscriber init |
| `debug` | 流式 chunk 概况、工具进入 / 退出、retry 决策 |
| `trace` | 单字符级 SSE 字节、JSON 解析每一步——不要默认开 |

调试 ACP 桥接：`RUST_LOG=defect_acp=debug,defect_agent=debug,info`。
调试 LLM provider：`RUST_LOG=defect_llm=debug,toac=debug,info`。

## 5. 与错误层 / 安全的衔接

### 5.1 错误诊断字段（与 errors.md §8 衔接）

`ProviderError::request_id`、`AgentError::McpStartup::server` 这类诊断字段
在错误产生处不上日志。日志在**接住错误的边界**写：

```rust
// 在 acp::serve::run_prompt_turn 里：
Ok(Err(err)) => {
    tracing::warn!(
        kind = ?err,                   // Debug 形态，含 source chain
        request_id = err.request_id(), // 有 helper 的话；没有就单独 match 取
        "turn failed"
    );
    return responder.respond_with_error(AcpError::Turn(err).into_wire_error());
}
```

这样同一个 `ProviderError` 只在 acp 边界打一次，turn 主循环 / provider 实现
都不会重复打。

### 5.2 凭证脱敏

toac 生成的 wire 代码现在会输出：

```
INFO toac: request uri=... headers={"authorization": "Bearer sk-..."}
```

这是日志的"结构化 = 只把字段 attach 进去而不是字符串拼接"原则被坏的反例：
authorization header 被原样打进事件里。

修复路径（按代价递增）：

a. 把 toac 这条事件的级别提到 `debug`，让默认 `info` 跑不到——临时方案，
   不解决"开 debug 时仍泄露"的问题。

b. 在 `defect-llm` 里加一个 reqwest middleware，在请求被 toac 序列化之前
   把 authorization header 改成 redacted 形态再发——拒绝；这会破坏真实请求。

c. 给 toac 加 hook，让生成的事件 emit 时跑一个 redactor。这是正解，但需要
   改 codegen 模板。

d. **本 v0 选**：用 [`tracing_subscriber::fmt::format`] 自定义 event formatter，
   在 fmt layer 里识别字段名是 `headers`、值里包含 `Bearer ` 时替换。
   实现成本最低且对所有未来源都有效。

实现细节：在 `examples/common/mod.rs` 与 `defect-cli/src/tracing_init.rs`
（或同等位置）的 subscriber builder 里挂上一个自定义 `FormatEvent`
实现，把 `Bearer\s+\S+` / `sk-[A-Za-z0-9_-]+` 替换成 `<redacted>`。

> 现状：v0 落地时先做 (a) —— 把 toac 的请求事件级别拉到 debug，立刻关闭
> 默认级别下的明文风险；(d) 列入后续 work。

### 5.3 stdout 污染防护

CLI / examples / 任何 ACP 服务端形态——subscriber 必须 `with_writer(std::io::stderr)`。
这是约定也是 review check 项。新增 binary 时对照本节确认。

## 6. 实现路线（v0）

按以下顺序落地，每步可独立 commit：

1. **抽 `tracing_init`**：在 `crates/cli/examples/common/mod.rs` 新建一个共用
   helper（或就用 `crates/llm/examples/common/mod.rs` 的形态——已经有同名
   helper）。`crates/cli/src/main.rs` 与 `crates/cli/examples/deepseek_e2e.rs`
   共用。
2. **降 toac request 级别**：把 `defect-llm` 里 toac 的请求事件级别拉到 debug
   （或 codegen 输出层加 EnvFilter 覆盖）；默认 `info` 不再泄露 authorization。
   实施时具体做法见 §5.2 (a)。
3. **turn / llm_call / tool_call span**：给 `Session::run_turn` /
   `LlmProvider::send` / `Tool::execute` 加 `#[tracing::instrument(skip_all, fields(...))]`。
4. **acp 桥接 span**：`run_prompt_turn` 包 `info_span!("acp_prompt_turn")`，
   `spawn_permission_request` 包 `info_span!("acp_request_permission")`。
5. **错误边界日志**：把 `respond_with_error` 之前的 `format!`-only
   分支改成 `tracing::warn!(?err, "...")`+ 投影。
6. **OTLP exporter（P2，不在 v0 做）**：见 §7。

## 7. OTLP exporter（P2 草图）

留接口，不实现。预期形态：

```rust
// crates/cli/src/tracing_init.rs（届时新建）
pub fn init_tracing(opts: TracingOpts) -> anyhow::Result<()> {
    let registry = tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(fmt_layer_stderr());

    if let Some(otlp) = opts.otlp_endpoint {
        let exporter = opentelemetry_otlp::new_exporter()
            .tonic()
            .with_endpoint(otlp);
        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(exporter)
            .install_batch(opentelemetry_sdk::runtime::Tokio)?;
        registry
            .with(tracing_opentelemetry::layer().with_tracer(tracer))
            .init();
    } else {
        registry.init();
    }
    Ok(())
}
```

P2 引入时 §3 的 `tracing_subscriber::fmt().init()` 退化成上面这种 layered
形态。span 字段约定（§2.1–§2.4）已经按 OpenTelemetry semconv 风格挑过，
迁移时不需要重命名。

[`tracing-error`]: https://docs.rs/tracing-error
[`tracing_subscriber::fmt::format`]: https://docs.rs/tracing-subscriber/latest/tracing_subscriber/fmt/format/
