# Defect 架构

Defect 是一个 **headless** 通用编码 agent：自身不提供 UI，前端（Zed 等）通过 **ACP**（Agent Client Protocol）接入，第三方工具通过 **MCP** 接入。本文用流程图梳理整体结构、各循环、数据流向、启停与钩子。

> 配置项的真相源是 `crates/config/src/types.rs`，钩子/事件的真相源是 `crates/agent/src/hooks/step.rs` 与 `crates/agent/src/event.rs`。本文若与代码不符，以代码为准。

---

## 1. 定位与关键决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 对外协议 | Zed 的 ACP | 现成规范，前端直接接入，无需自定义协议 |
| LLM provider | Anthropic Messages + OpenAI-compatible（含 Bedrock/DeepSeek/LiteLLM） | 覆盖主流后端 |
| 工具扩展 | 内置 trait + MCP 双轨 | 内置工具走 crate 内 trait（性能/语义），第三方走 MCP |
| 沙箱 | 策略决策层（read-only / ask-writes / open / deny-all） | OS 级隔离作为未来可插拔后端 |
| 会话持久化 | jsonl append-only，可 resume | 索引化存储留待演进 |

---

## 2. Crate 分层

`defect-agent` 是**核心叶子**：定义 Session/Turn/Event 以及 `LlmProvider`/`Tool`/`HookEngine` 等 trait，**不依赖任何其它 workspace crate**。所有基础设施 crate 反向依赖它（拿到 trait 去实现），`defect-cli` 在最外层把一切组装起来。

```mermaid
graph TD
    cli[defect-cli<br/>组装 + 入口]
    acp[defect-acp<br/>ACP server]
    llm[defect-llm<br/>provider 实现]
    mcp[defect-mcp<br/>MCP client]
    tools[defect-tools<br/>内置工具]
    config[defect-config<br/>配置加载/合并]
    obs[defect-obs<br/>tracing/Langfuse]
    storage[defect-storage<br/>jsonl 持久化]
    http[defect-http<br/>HTTP 栈]
    sandbox[defect-sandbox<br/>策略原语]
    agent[defect-agent<br/>核心：Session/Turn/Event + traits]

    cli --> acp & llm & mcp & tools & config & obs & storage & http & agent
    acp --> agent & tools & storage
    llm --> agent & http
    mcp --> agent & http
    tools --> agent & config
    config --> agent
    obs --> agent & http
    storage --> agent
    http --> agent
    sandbox -.no internal deps.- sandbox

    style agent fill:#2b6cb0,color:#fff
    style cli fill:#2f855a,color:#fff
```

依赖方向恒为「**指向 agent**」，保证核心不被基础设施反向污染（例如 config 依赖 agent，而非反过来——配置层复用 agent 的 `TurnConfig` / `BackgroundProgressConfig` 等结构作为真相源）。

---

## 3. 启动流程

入口 `crates/cli/src/bin/cli.rs::main`。`defect init` 子命令走单独的配置生成路径后退出；正常启动按下图组装并路由到一个运行模式。

```mermaid
flowchart TD
    start([main]) --> parse[解析 CLI args<br/>clap → CliArgs]
    parse --> initcmd{Command::Init?}
    initcmd -->|是| init[defect init<br/>探测 key + live list-models<br/>写全局 config] --> done0([exit])
    initcmd -->|否| load[load_config<br/>default→user→project→local→CLI 合并]
    load --> trace[init_tracing<br/>+ 打印 config warnings]
    trace --> build[CliAgentBuilder.build<br/>见 §3.1]
    build --> route{运行模式<br/>优先级}
    route -->|--goal / --message| oneshot[oneshot::run<br/>单次 run_turn + 退出码]
    route -->|--repl| repl[repl::run<br/>交互循环]
    route -->|默认| serve[acp::serve_with_resume<br/>stdio ACP server]
```

### 3.1 Agent 组装（CliAgentBuilder.build）

```mermaid
flowchart LR
    subgraph build[CliAgentBuilder.build]
      direction TB
      prof[发现 profiles / skills<br/>项目层 + 用户层]
      reg[build_registry<br/>provider 注册表 + TurnConfig]
      pt[build_process_tools<br/>bash/fs/fetch/search/spawn_agent]
      pol[解析 sandbox policy]
      hk[build_main_session_engine<br/>hooks + skill + goal-gate]
      st[StorageObserver<br/>jsonl 订阅]
      lf[Langfuse observer]
      mcpf[McpToolFactory<br/>per-session MCP 连接器]
      core[DefaultAgentCore]
      prof --> reg --> pt --> pol --> hk --> st --> lf --> mcpf --> core
    end
    core --> built[BuiltCliAgent<br/>agent + turn_config + sandbox_mode<br/>+ shell_output_max_bytes + goal]
```

`BuiltCliAgent` 携带 `shell_output_max_bytes`（来自 `[tools.bash].output_max_bytes`），三个前端（REPL / oneshot / ACP 本地模式）构造 `LocalShellBackend` 时都用它，避免某条入口静默丢配置。

---

## 4. Turn 主循环（核心）

`crates/agent/src/session/turn.rs::TurnRunner::run`。一个 turn 是一个状态机：注入用户输入 → 反复「压缩检查 → 调 LLM → 跑工具」直到模型自愿结束或撞上限，期间在固定 14 个挂载点触发 hook。

```mermaid
flowchart TD
    A([run prompt]) --> B[before_ingest hook<br/>可改写/拦截]
    B --> C[记录 rollback 边界<br/>history.len]
    C --> D[追加 user message → history]
    D --> E[after_ingest hook]
    E --> F[TurnStarted 事件]
    F --> G[after_turn_enter hook<br/>可注入上下文 / Break]
    G --> loop{{主循环}}

    loop --> MC[manage_context<br/>三档压缩水位 见 §6]
    MC --> RB[build_request<br/>history→CompletionRequest<br/>修复孤立 tool_use]
    RB --> BG[before_generate hook<br/>可改 model / 注入 / Break]
    BG --> LLM[call_llm_with_retry<br/>+ drain stream]
    LLM --> LF[LlmCallFinished 事件<br/>per-call usage]
    LF --> AG[after_generate hook]
    AG --> AM[追加 assistant message → history]
    AM --> STOP{stop reason}

    STOP -->|Refusal / MaxTokens| EXIT[退出循环]
    STOP -->|EndTurn 或 撞 request_limit| TE[before_turn_end hook<br/>goal-gate 在此]
    TE -->|Continue 注入反馈| RESET[重置请求预算] --> loop
    TE -->|放行| EXIT
    STOP -->|ToolUse| PERM

    subgraph 工具执行
      PERM[before_permission hook] --> DEC[decide_permissions<br/>Ask/Allow/Deny]
      DEC --> AP[after_permission hook]
      AP --> RUN[并发跑工具<br/>max_concurrent_tools]
      RUN --> OV[丢弃超 context 的结果]
      OV --> ATB[after_tool_batch hook<br/>可注入 / Break]
      ATB --> TR[追加 tool_results → history]
    end
    TR --> loop

    EXIT --> END[TurnEnded 事件<br/>累计 usage]
    END --> Z([返回 StopReason])

    LLM -.出错.-> ERR[history.truncate rollback<br/>TurnAborted 事件] --> Z
```

**hook 挂载点全集**（`ALL_EVENT_NAMES`，拼错即硬失败）：`after_session_enter`、`after_turn_enter`、`before_ingest`、`after_ingest`、`before_compact`、`after_compact`、`before_generate`、`after_generate`、`before_permission`、`after_permission`、`before_tool_apply`、`after_tool_apply`、`after_tool_batch`、`before_turn_end`。全部**同步**串行触发；单个 handler 超时/panic/出错按降级表跳过（默认超时 5s，可按 hook 配 `timeout_sec`）。

---

## 5. 会话驱动与三种循环

`DefaultSession::run_turn` 用 `turn_lock` 保证单会话同一时刻只跑一个 turn。turn 之间如何续接，取决于模式：

```mermaid
flowchart TD
    subgraph oneshot[--message / --goal]
      o1[run_turn] --> o2{goal 模式?}
      o2 -->|否| o3[输出 + 退出码]
      o2 -->|是| o4[goal-gate 在 before_turn_end<br/>检查 GoalState.is_reached]
      o4 -->|未达成且未超 max_hook_continues| o1
      o4 -->|达成 / 耗尽| o5[退出码<br/>未达成=非零]
    end

    subgraph repl[--repl 交互]
      r1[读 stdin 一行] --> r2[run_turn] --> r3[事件渲染到终端] --> r1
    end

    subgraph acp[ACP server 默认]
      a1[session/prompt] --> a2[run_turn] --> a3[SessionUpdate 通知回客户端] --> a1
    end
```

**后台任务续转**：`spawn_agent { run_in_background: true }` 起一个子会话，完成后把 `BackgroundOutcome` 入队；`DefaultSession::run_turn` 在下一轮开始前 drain 队列，把结果作为**前缀块**拼到用户 prompt 前。ACP/REPL 的事件 pump 在任务完成时被 `Notify` 唤醒，可主动起一轮自治 turn 处理结果。

---

## 6. 三档上下文压缩

水位都按模型 `context_window` 的比例推导（Bedrock 等不暴露 window 的 provider 需在 `[providers.x.models]` 显式声明 `context_window`，否则压缩无从触发）。在主循环每次调 LLM 前的 `manage_context` 统一编排：

```mermaid
flowchart TD
    EST[估算 history token] --> M{≥ micro 0.6?}
    M -->|是| MICRO[微压缩<br/>清旧轮超大 tool_result<br/>不调 LLM / 不删消息]
    M -->|否| SOFT
    MICRO --> SOFT{∈ soft 0.7, hard 0.85?}
    SOFT -->|是| BGC[后台全量压缩<br/>异步摘要 不阻塞当前轮<br/>单点 CompactionSlot]
    SOFT -->|否| HARD
    BGC --> HARD{≥ hard 0.85?}
    HARD -->|有后台任务在飞| WAIT[等待其完成]
    HARD -->|无| SYNC[同步压缩兜底<br/>阻塞当前轮 保证不超窗]
    WAIT --> SPLICE
    SYNC --> SPLICE[splice_prefix<br/>摘要消息替换被压前缀<br/>history→Arc 回写]
```

约束（启动校验，违反硬失败）：每个 ratio ∈ (0,1]，且 `micro ≤ soft < hard`。三档各自有独立开关。

---

## 7. 数据流：一条消息的旅程

```mermaid
flowchart LR
    UI[ACP 客户端] -->|ContentBlock| ING[before_ingest hook]
    ING -->|Message User| HIST[(History<br/>Vec Message)]
    HIST --> REQ[build_request<br/>+ ToolSchema + SamplingParams]
    REQ -->|CompletionRequest| PROV[LlmProvider]
    PROV -->|wire: anthropic/openai| NET[provider HTTP]
    NET -->|流式 ProviderChunk| DRAIN[drain stream<br/>累积 text/tool_use/thinking + usage]
    DRAIN -->|Message Assistant| HIST
    DRAIN -->|ToolUse| TOOL[ToolRegistry.invoke]
    TOOL -->|ToolResult| HIST
    DRAIN -.AgentEvent.-> BUS[(EventEmitter<br/>broadcast)]
    HIST -.AgentEvent.-> BUS
    BUS --> ACPOUT[defect-acp → SessionUpdate]
    BUS --> STORE[defect-storage → jsonl]
    BUS --> LANG[defect-obs → Langfuse]
```

跨边界的关键类型：`ContentBlock`（ACP 线格式/事件）、`Message`/`MessageContent`（agent 内部历史）、`CompletionRequest`/`ProviderChunk`（provider 协议面）。`defect-llm` 内的 wire 层（toac 代码生成 + quirk strip 补丁）负责 agent 类型 ↔ 各家 vendor JSON 的互转。

---

## 8. 事件系统

`crates/agent/src/event.rs::AgentEvent`，由 `EventEmitter` 广播，多订阅者各自消费、互不阻塞。

```mermaid
flowchart TD
    subgraph 事件源[turn 主循环 / 会话]
      direction LR
      E1[TurnStarted/Ended/Aborted]
      E2[UserPromptCommitted]
      E3[AssistantText/Thought]
      E4[ToolCallStarted/Progress/Finished]
      E5[PolicyDecision/PermissionResolved]
      E6[LlmCallStarted/Finished]
      E7[ContextCompressed/Microcompacted]
      E8[Subagent 嵌套包裹]
    end
    EM[(EventEmitter)] --> S1[ACP: SessionUpdate 回前端]
    EM --> S2[storage: jsonl append]
    EM --> S3[obs: Langfuse trace/generation]
    EM --> S4[REPL/oneshot: 终端/stdout 渲染]
    事件源 --> EM
```

`Subagent` 事件用 `ancestor_path` 扁平化携带子 agent 层级，Langfuse projector 据此重建 trace→step→(llm_call+tools) 分层。

---

## 9. 生命周期：启停与钩子

```mermaid
sequenceDiagram
    participant F as 前端/CLI
    participant C as DefaultAgentCore
    participant S as DefaultSession
    participant O as Observers

    F->>C: create_session / load_session
    C->>C: 组装 per-session ToolRegistry<br/>(内置 + MCP) + 应用 profile allowlist
    C->>S: 构造 Session(history/events/permissions)
    C->>O: on_session_created (storage 订阅 / Langfuse 连接)
    Note over S: after_session_enter hook<br/>(skill manifest / goal 注入)
    loop 每个 turn
      F->>S: run_turn(prompt)
      S->>S: §4 turn 主循环
      S-->>O: AgentEvent 流
    end
    F->>S: drop / 会话结束
    S->>S: 取消 BackgroundTasks token<br/>flush history / 关 MCP 连接<br/>Langfuse trace 收尾
```

**启停要点**：MCP 连接是 per-session、由 ToolRegistry 持有，会话 drop 时关闭；后台任务挂在会话级 `CancellationToken` 下，会话结束统一取消；`--local` 模式锚到 repo-root/.defect 并完全忽略用户层。
