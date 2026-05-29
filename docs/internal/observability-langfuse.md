# 可观测性独立化 & Langfuse 接入

把 tracing / 用量追踪等可观测性能力从 `defect-cli` 抽出，沉淀到独立 crate
`defect-observability`；首个对外集成是 **Langfuse**（LLM tracing / 用量分析平台）。

```text
                ┌──► defect-acp           (翻译成 SessionUpdate / PromptResponse)
AgentEvent ────┼──► defect-storage        (投影成 journal，供 resume)
                ├──► tracing-subscriber    (结构化日志，stderr)
                └──► defect-observability  (Langfuse trace/generation/span 上报) ← 本文
```

本文沉淀新 crate 的定位、事件→Langfuse 的映射、上报协议、配置 schema 与关键取舍。
读之前建议先读 [`event-model.md`](./event-model.md)（事件流形状）与
[`session.md`](./session.md) §5（`EventEmitter` 的 fan-out / backpressure 语义）。

---

## 1. 定位与边界

### 1.1 为什么独立成 crate

当前可观测性散在 `defect-cli`：`tracing_init.rs` 做 `tracing-subscriber` 初始化，用量
只是 `AgentEvent` 流上的字段（`TurnEnded.usage` / `LlmCallFinished.usage`），无任何
外部导出。独立化的动机：

1. **解耦 cli 装配**。cli 的 `bin/cli.rs` 是"拼装样板"，二次开发者照着改。把可观测性
   的初始化收敛到一个 `defect_observability::init(...)` 入口，cli 只调一行，后续加
   OTLP / Langfuse / 本地用量汇总都不再动 cli。
2. **复用 `SessionObserver` 接缝**。Langfuse 上报本质是"再挂一个事件流消费者"，
   `defect-storage::StorageObserver`（`crates/storage/src/lib.rs:51`）就是现成范本——
   `on_session_created` 里 `session.subscribe()` 拿独立 mpsc 流、`tokio::spawn` 消费。
   把它放进 `defect-agent`/`defect-cli` 都不合适：前者会让核心 crate 背上 HTTP 上报依赖，
   后者会让"换 cli"的人重写遥测。独立 crate 最干净。
3. **为后续 OTLP 留位**。`defect-config` 里已有 `TracingConfig { filter, otlp }` 脚手架
   （`crates/config/src/types.rs:700`），但 `otlp.endpoint` 当前无人消费。OTLP 导出未来
   也落在这个 crate，与 Langfuse 共用一套 `init` 入口。

### 1.2 crate 范围

| 模块 | 职责 |
|---|---|
| `tracing_init` | 从 `defect-cli` 搬过来的 `tracing-subscriber` 初始化（零行为变化） |
| `langfuse::observer` | `LangfuseObserver: impl SessionObserver`，每 session 订阅事件流 |
| `langfuse::projector` | `AgentEvent` 流 → Langfuse ingestion 事件（核心翻译逻辑，纯函数易测） |
| `langfuse::ingest` | 批量缓冲 + POST `/api/public/ingestion`（后台任务，可丢弃降级） |
| `langfuse::model` | ingestion 事件的 serde 结构体 |
| `langfuse::config` | `LangfuseConfig`，由 `defect-config` 的 `TracingConfig` 投影而来 |
| `usage`（可选，二期） | 进程级用量聚合，给本地 `--usage` 汇总用 |

### 1.3 显式配置，不自动探测

遵循既有约定（provider 不靠"哪个 env-key 恰好存在"自动选择）：Langfuse **默认关闭**，
必须 `[tracing.langfuse].enabled = true` 显式打开，且 `public_key` / `secret_key` 缺失
即视为配置错误（启动 warn 并禁用，不静默跑）。详见 §5。

---

## 2. SessionObserver 接缝（范本：StorageObserver）

`SessionObserver`（`crates/agent/src/session.rs:144`）在 session 创建成功后被调用，
多个 observer 天然叠加。`LangfuseObserver` 照抄 `StorageObserver` 的形状：

```rust
impl SessionObserver for LangfuseObserver {
    fn on_session_created(
        &self,
        session: Arc<dyn Session>,
        info: SessionCreateInfo,
    ) -> Result<(), BoxError> {
        let mut events = session.subscribe();        // 独立 mpsc 流
        let ingest = self.ingest.clone();            // 共享上报句柄（Arc）
        let session_id = info.id.clone();
        let provider = session.provider_info().vendor.clone();
        tokio::spawn(async move {
            let mut proj = TraceProjector::new(&session_id, &provider);
            while let Some(ev) = events.next().await {
                for cmd in proj.project(ev) {         // AgentEvent → Vec<IngestionEvent>
                    ingest.enqueue(cmd);              // 非阻塞，满了丢弃（见 §4）
                }
            }
            ingest.flush().await;                     // 流结束（session drop）冲刷残留
        });
        Ok(())
    }
}
```

cli 接线（`crates/cli/src/bin/cli.rs:85` 旁，仅 +1 行）：

```rust
    .observe_session(storage.clone())
    .observe_session(langfuse.clone())   // ← 新增
```

> **backpressure 警告**。`EventEmitter::emit` 是串行 await 所有订阅者，慢消费者会反压
> **agent 主循环**（这是 storage "不丢"语义有意为之的）。Langfuse 是旁路遥测，**绝不能**
> 反压主循环——所以 `enqueue` 必须是**非阻塞、立即返回**的（写入有界 channel，满了丢弃），
> 真正的网络 IO 在 `ingest` 的独立后台任务里做。详见 §4。

---

## 3. 事件 → Langfuse 映射

### 3.1 trace 粒度：每个 turn 一个 trace

一次 prompt turn（`TurnStarted` → `TurnEnded`）对应 Langfuse 里一个 **trace**；trace 内：

- 每次 LLM provider 调用 → 一个 **generation**（带 model / usage / input / output）
- 每个工具调用 → 一个 **span**
- 上下文压缩 → trace 上一个 **event**

`sessionId` 用 defect 的 `SessionId`，使 Langfuse 能按会话把同一会话的多个 turn-trace
串起来看脉络，但每个 trace 本身边界清晰、与 `usage` 累加边界（turn 级）一致。

#### 跨进程重启的会话关联

**策略：跟随 ACP 会话语义，observer 层不擅自归并。** defect 的 `SessionId` 现在是真实
UUID v4（`new_session_id()`，`crates/agent/src/session/default.rs`）：

- 客户端走 `session/load`（resume）→ 沿用客户端传回的原 `SessionId`（`serve.rs:465`），
  Langfuse 里仍是**同一个 session**，历史 turn 的 trace 与新 turn 的 trace 都挂在它下面。✅
- 客户端走 `session/new` → `SessionId::new(new_session_id())`（`serve.rs:434`）是全新 UUID，
  Langfuse 里是**新 session**，与上次对应不上。

这是客户端的会话语义，不是 observer 能（或该）修正的。如果将来要"同一项目/目录的多次
开会自动归并"，那是产品决策，应在更上层（用 cwd 哈希做 sessionId 或写 metadata）实现，
不混进事件投影。本期不做。

### 3.2 映射表

| AgentEvent | Langfuse ingestion 动作 |
|---|---|
| `TurnStarted` | `trace-create`：新 traceId，name=`turn`，sessionId=会话 id，metadata.provider，startTime |
| `UserPromptCommitted { content }` | 暂存为 trace 的 `input`（用户 prompt 文本，于 `TurnEnded` 写出，或随 trace-create 即写） |
| `LlmCallStarted { model, attempt }` | `generation-create`：parentObservationId=trace，model，metadata.attempt，startTime；压入 call 栈 |
| `AssistantText { content }` | 累积进当前 generation 的 `output`（文本增量拼接） |
| `AssistantThought { content }` | 累积进当前 generation 的 `output`（标记为 thinking，或单独字段） |
| `LlmCallFinished { model, attempt, usage, error }` | `generation-update`：endTime，`usage`（见 §3.3），level=ERROR + statusMessage if `error` |
| `ToolCallStarted { id, name, fields }` | `span-create`：name=工具名，input=参数，startTime；记 id→spanId |
| `ToolCallProgress { id, fields }` | 累积进对应 span 的 output（可选；增量进度） |
| `ToolCallFinished { id, fields }` | `span-update`：endTime，output=结果，level 由 `fields.status`（失败→ERROR） |
| `PolicyDecision` / `PermissionResolved` | trace 上的 `event`（审计信号，可选；非 LLM 用量核心） |
| `ContextCompressed { tokens_before, tokens_after }` | trace 上的 `event`：记 before/after token，name=`context_compaction` |
| `TurnEnded { reason, usage }` | `trace-update`：endTime，output=最终助手文本，metadata.stopReason，汇总 usage → 触发 flush |

### 3.3 usage 字段映射

defect 的 `Usage`（`crates/agent/src/llm/chunk.rs:76`）→ Langfuse generation `usage`：

| defect `Usage` | Langfuse usage |
|---|---|
| `input_tokens` | `input`（即旧 `promptTokens`） |
| `output_tokens` | `output`（即旧 `completionTokens`） |
| `input + output` | `total` |
| `cache_read_input_tokens` | `usageDetails.cache_read`（Langfuse 自定义用量明细） |
| `cache_creation_input_tokens` | `usageDetails.cache_creation` |

各字段 `Option<u64>`，`None` 视为缺省不上报（不写 0，避免污染聚合）。

### 3.4 时间戳缺口

`AgentEvent` **不带时间戳**（设计上只表达语义边界）。Langfuse 的 trace/generation/span
强依赖 start/end time 算 latency。解法：**消费端收到事件时用 `SystemTime::now()` 打点**。
因为事件流是顺序、近实时的（emit→消费仅隔 mpsc 一跳），这个近似对 latency 展示足够好。

> 若日后要精确 latency，应在 `defect-agent` 主循环发事件时带上单调时钟戳，那是独立的
> 上游改动，不在本接入范围。

### 3.5 turn / call 配对状态

`TraceProjector` 是**有状态**的逐 session 投影器（与 storage 的 `RecordProjector` 同构）：

- 当前 turn 的 `trace_id` + 暂存的 user input / 累积的 assistant output；
- 一个 LLM-call 栈（`attempt` 重试时 `LlmCallStarted` 会多次，每次一个 generation）；
- `ToolCallId → span_id` 映射（工具调用的 Started/Progress/Finished 跨事件配对）。

`traceId` / `observationId` 的生成：

- **traceId**：在 `TurnStarted` 时生成**一次** `Uuid::new_v4()`，存进 projector 当前 turn
  状态，该 turn 内所有 ingestion 事件复用它。
- **generation / span id**：派生自当前 `trace_id` + 进程内 call 序号 /
  `ToolCallId`，如 `{trace_id}-gen-{call_seq}`、`{trace_id}-tool-{tool_call_id}`。
  trace_id 全局唯一 → 派生 id 也唯一。

> ⚠️ **不要用 `{session_id}-turn-{turn_seq}` 这类进程内自增 id 当 traceId。** turn_seq
> 在 `session/load`（resume）后会从 0 重新计数，与上次进程产生的 trace id 撞——Langfuse
> 会把两段本不相干的 turn 误并成一个 trace。traceId 必须 turn 级随机，resume 不重置。
> （早期设计稿犯过这个错，这里特别标注。）

进程内同一 turn 的重复上报（flush 重试）仍幂等：同一 traceId 由 Langfuse 去重。

---

## 4. 上报器 ingest：批量 + 可丢弃降级

### 4.1 协议

Langfuse 批量摄取端点：`POST {host}/api/public/ingestion`，body：

```json
{ "batch": [ { "id": "...", "type": "trace-create", "timestamp": "...", "body": { ... } }, ... ] }
```

- **认证**：HTTP Basic，`public_key:secret_key`。
- **批量**：攒满 N 条 **或** 隔 T 秒冲刷一次，降低请求数。
- **传输**：**复用 `defect-http::build_http_stack` 输出的 `HttpStack`**
  （`crates/http/src/lib.rs:158`）。它是一条 type-erased tower service，
  `call(toac::Request) -> http::Response<hyper::body::Incoming>`，已经叠好超时 /
  transport 重试 / 代理 / UA / trace 全套层栈——和 LLM provider 走的是同一份基础设施，
  代理 / TLS / UA 策略天然统一。

  关键认识：`toac::Request` **不是** LLM 专用的请求类型，它就是标准的
  `http::Request<toac::body::Body>`（`toac/src/request.rs:7`）；而 `toac::body::Body`
  的 doc 明确写它能 `adapt any http_body::Body<Data = Bytes> (from hyper, reqwest, etc.)`
  （`toac/src/body.rs`）。所以构造一条普通 JSON POST 就是：

  ```rust
  let req: toac::Request = http::Request::builder()
      .method(http::Method::POST)
      .uri(format!("{host}/api/public/ingestion"))
      .header(http::header::AUTHORIZATION, basic_auth)        // public:secret
      .header(http::header::CONTENT_TYPE, "application/json")
      .body(toac::body::Body::new(http_body_util::Full::new(Bytes::from(json))))?;
  let resp = http_stack.call(req).await?;                     // 复用整条栈
  ```

  > 因此**不需要**另起 hyper client、不需要单独抠 connector、也完全不碰
  > `defect-agent::HttpClient`（那是 fetch 工具的 GET-only trait，与本路径无关）。
  > cli 装配时把已建好的 `HttpStack`（或其 builder 配置）注入给 `LangfuseObserver` 即可。

### 4.2 可丢弃降级（关键取舍）

Langfuse 是**旁路遥测，可丢可降级**，与 storage 的"不丢"语义相反：

- `enqueue` 写入**有界** channel（如容量 1024 条 ingestion 事件）；
- channel 满 → **丢弃并 `tracing::warn!` 计数告警**（如每丢 N 条 warn 一次），**不**阻塞；
- 后台 flush 任务负责 batch + POST；
- POST 失败 → `tracing::warn!`，**不重试**（避免堆积反压；遥测丢点可接受）；
- session 流结束时 `flush()` 尽力冲刷残留，但不保证送达。

这条原则的硬约束：**任何 Langfuse 故障（网络挂、key 错、平台慢）都不得影响 agent 正常工作**。

---

## 5. 工具内部调用的上下文透传（embedding / 后端）

### 5.1 问题

工具（内置或 MCP）执行时若自己调了 LLM / embedding / 检索后端，我们希望这些调用也作为
**当前 turn-trace 下的子 observation** 出现，从而和 session 关联。但今天的 `ToolContext`
（`crates/agent/src/tool.rs:123`）只有 `cwd / cancel / fs / shell / http`——**不含
session_id、不含 trace 上下文、不含事件总线**。工具完全不感知自己属于哪个 session / turn，
所以工具内的 embedding 调用没有任何线索能挂回 trace。必须显式透传一个可观测性上下文。

### 5.2 方案：显式扩 `ToolContext`（已选定）

`ToolContext` 是 `#[non_exhaustive]`，且唯一 cross-crate 构造入口是 `ToolContext::new`，
加字段不破坏现有实现。新增一个可选 obs 上下文：

```rust
/// 注入给工具的可观测性上下文。Langfuse 未启用时为 None，工具据此跳过上报。
#[derive(Clone)]
pub struct ObsContext {
    pub session_id: String,
    pub trace_id: String,            // 当前 turn 的 trace id
    pub parent_span_id: String,      // 本次工具调用对应的 span id（作为 embedding 调用的 parent）
    pub ingest: Arc<dyn ObsSink>,    // 上报句柄，与 turn 上报共用一个后台 ingest
}
```

工具调 embedding 时，用 `obs.parent_span_id` 当 `parentObservationId` 建一个子 generation，
经 `obs.ingest` 上报。Langfuse 里就形成 `turn-trace → tool-span → embedding-generation`
的嵌套，天然与 session 关联。

> **为何不用 tracing task-local 隐式传播**：那样工具签名不变，但引入隐式全局态，与
> `tool.rs:119` 的明确设计取向（"显式 struct 而非环境变量 / thread-local，避免隐式全局
> 状态"）相悖。本项目一贯走显式注入，保持一致。

### 5.3 装配上的依赖倒挂注意

`ObsContext` 的类型（`ObsSink` trait）应定义在 `defect-agent`（与 `ToolContext` 同 crate），
**具体实现**在 `defect-observability`。否则 `defect-agent` 会倒挂依赖 observability crate。
turn 主循环构造 `ToolContext`（`turn.rs:1304` / describe 路径 `:579`）时，把当前 turn 的
trace_id + 即将创建的工具 span_id 填进 `ObsContext`。Langfuse 未启用时整个字段为 `None`，
零成本。

> **范围**：扩 `ToolContext` + 定义 `ObsSink` trait 属于本期；让某个具体工具真的去调
> embedding 并上报，是该工具自己的事，按需接入。

---

## 6. 配置 schema

扩展既有 `[tracing]`（`crates/config/src/types.rs:700` 的 `TracingConfig`）：

```toml
[tracing]
filter = "info,toac=warn"        # 既有

[tracing.langfuse]               # 新增
enabled = true                   # 默认 false；不开则 observer 不挂
host = "https://cloud.langfuse.com"
public_key = "${LANGFUSE_PUBLIC_KEY}"   # 支持 env 占位；或纯走 env，config 只留 enabled/host
secret_key = "${LANGFUSE_SECRET_KEY}"
flush_interval_ms = 2000         # 可选，默认 2s
max_batch = 100                  # 可选
```

config 侧改动：

- `types.rs`：`TracingConfig` 加 `langfuse: Option<LangfuseConfig>`；新增
  `LangfuseConfig { enabled, host, public_key, secret_key, flush_interval_ms, max_batch }`
  + 其 `(pub(crate))` Deserialize section（对齐 `OtlpTracingSection` 写法）。
- `loader.rs`：投影 section → 有效配置（对齐 `loader.rs:387` 对 `otlp` 的处理）。
- **redact / sanitize**：`secret_key` 是凭据。`loader.rs:574` 已有"shared project config
  不许重定向 endpoint / 凭据"的 sanitize 规则，Langfuse 的 host/secret 必须纳入同一套
  规则与 path 白名单（`loader.rs:801` / `:895` 附近的 `tracing.otlp.*` 列表要补
  `tracing.langfuse.*`）。

### 6.1 显式校验

`enabled = true` 但缺 key → 启动 `tracing::warn!` 并**禁用** Langfuse（不 panic、不静默成功）。
符合既有"不要做错误的 v0 实现：要么明确报错、要么做对"约束。

---

## 7. 落地顺序

1. **建 crate + 搬 `tracing_init`**：零行为变化，先让 workspace 编译通过；cli 改调
   `defect_observability::init_tracing`。
2. **`model.rs` + `ingest.rs`**：纯上报路径，先用假事件打通 Langfuse 联调（验证 auth /
   batch / 端点）。
3. **`projector.rs`**：事件翻译 + 单测（参照 `storage` 的 `RecordProjector` 测试风格，
   喂 `AgentEvent` 序列断言产出的 ingestion 事件）。trace_id 用 turn 级 `Uuid::new_v4()`。
4. **`LangfuseObserver` + cli 接线**。
5. **config 扩展 + redact + 校验**。
6. **`ObsContext` + `ObsSink` trait**（§5）：扩 `ToolContext`，turn 主循环填充；让工具内
   embedding/后端调用能挂回 trace。具体工具接入按需。

> 已完成（前置）：`SessionId` 改真实 UUID v4（`new_session_id()`），是跨重启关联与
> trace_id 稳定性的基础。

---

## 8. 未决 / 待二期

- **OTLP 导出**：`OtlpTracingConfig` 脚手架已在，未来在本 crate 加 OTLP 导出，与 Langfuse
  共用 `init`。本期不做——下面记录为什么这期选 ingestion 而非 OTLP（评估于 2026-05）。

  Langfuse 文档把 `/api/public/ingestion` 标为 "Legacy endpoint"、推荐 OTLP
  （`/api/public/otel`）。但权衡后本期仍走 ingestion：

  | 维度 | Ingestion（现选） | OTLP |
  |---|---|---|
  | 依赖体积 | 零新增（已用 hyper/http 栈） | 重 SDK（`opentelemetry` + `-sdk` + `-otlp` + `tracing-opentelemetry`，protobuf 还要 `prost`），或自己手写编码 |
  | 数据映射 | 直接命中 Langfuse 数据模型（`usageDetails` 等） | 经 **演进中** 的 GenAI semantic conventions，attribute 命名会漂移 |
  | 维护面 | 一套 Langfuse model | OTLP 规范 + GenAI 约定两套，且要学 `langfuse.*` 命名空间精确控制落点 |
  | 通用性 | 仅 Langfuse | 可同时发 Jaeger / Datadog 等多后端 |

  关键事实（查 Langfuse docs）：

  1. **GenAI 语义约定仍在演进**（文档原话 "still evolving"）。OTLP 下 usage/model 靠 span
     attribute（`gen_ai.usage.input_tokens` 等）被 Langfuse 反向解析，约定一变就可能解析不出。
     ingestion 的 `usageDetails` 是 Langfuse 自有稳定模型，直接命中。
  2. **workspace 当前零 OTEL 依赖**（424 个 crate 里无 opentelemetry/prost/tonic）。引入与本
     repo 体积优先约束（release profile `opt-level="s"` / `strip`）正面冲突。
  3. **OTLP 端点支持 HTTP/JSON**（`Content-Type: application/json`，非强制 protobuf、无需
     gRPC）——所以理论上可不引 SDK、手写 `ExportTraceServiceRequest` JSON，但那等于自维护
     "OTLP 信封 + GenAI 约定" 两套规范，且建在演进中的靶子上，比 ingestion 的 `model.rs` 复杂得多。
  4. **id 格式更硬**：OTLP 要 traceId 16 字节 hex / spanId 8 字节 hex；ingestion 用字符串
     `parentObservationId` 引用，与我们的 UUID 方案更契合。

  **何时该重估**：若要把可观测性接到 Langfuse 之外（Jaeger/Grafana/Datadog 等多后端），
  OTLP 的"通用性"就从成本变成投资，那时再上 OTLP（"Legacy" 标签是这个方向的长期信号）。
  我们既不需要多后端、又体积敏感，故本期 ingestion 更稳。
- **本地用量汇总**（`usage` 模块）：进程级 token 聚合 + `--usage` 输出，本期不做。
- **精确 latency**：需上游主循环发事件带时钟戳，本期用 `SystemTime::now()` 近似。
- **成本计算**：当前全栈只有 token 计数、无 USD 定价；Langfuse 侧可配 model price 自行算。
