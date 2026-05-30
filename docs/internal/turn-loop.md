# Turn 主循环设计

`Turn` 主循环是 `defect-agent` 的"心脏"——把 LLM provider 的流式输出、Tool 的并发执行、sandbox policy 的权限决策、History 的压缩这些子机能编排成一次符合 ACP `session/prompt` 语义的 turn。本文沉淀主循环的状态机、并发模型、取消语义、错误归类与配置口子。

`Session::run_turn` 的契约见 [`session.md`](./session.md) §3；本文是它的"内部实现规约"。

## 1. 总体形状

```text
                    ┌─ append user prompt to History
run_turn(prompt) ───┤
                    └─ enter loop ──────────────────────────────────┐
                       │                                            │
                       │  step 1: build CompletionRequest            │
                       │           from History::snapshot()          │
                       │  step 2: maybe compact (token threshold)    │
                       │  step 3: provider.complete(req)             │
                       │  step 4: drain provider stream:             │
                       │           - emit AssistantText / Thought    │
                       │           - accumulate ToolUse fragments    │
                       │           - accumulate Usage                │
                       │  step 5: on Stop:                           │
                       │           - EndTurn   → return EndTurn      │
                       │           - Refusal   → return Refusal      │
                       │           - MaxTokens → return MaxTokens    │
                       │           - ToolUse   → step 6              │
                       │  step 6: 决策每个 tool_use（串行）          │
                       │           - parse args                      │
                       │           - policy.classify → Allow/Deny/Ask│
                       │           - Ask: emit PolicyDecision(Ask)   │
                       │                  await resolve_permission   │
                       │  step 7: spawn 已批准的 tool（JoinSet 并发）│
                       │  step 8: 收集所有 tool 终态 → tool_result   │
                       │  step 9: append assistant + tool_results    │
                       │           to History                        │
                       │  step 10: 检查 turn_request_count           │
                       │           - 超 max_turn_requests → 终止      │
                       │  step 11: continue loop                     │
                       └────────────────────────────────────────────┘
```

主循环以**显式状态机**而非"递归 await"实现：每一步都显式地命名、emit 对应 `AgentEvent`、检查 cancel。递归方案在排障时栈被工具流 / LLM 流嵌套淹没，状态机扁平化以后单一函数体可读。

## 2. 顶层签名与状态

```rust
pub(crate) struct TurnRunner<'a> {
    session_id: &'a SessionId,
    history: &'a dyn History,
    tools: &'a dyn ToolRegistry,
    provider: &'a dyn LlmProvider,
    policy: &'a dyn SandboxPolicy,
    events: &'a EventEmitter,        // mpsc fan-out 入口
    permissions: &'a PermissionGate, // resolve_permission 的等待表
    cancel: CancellationToken,
    config: &'a TurnConfig,
}

pub(crate) struct TurnState {
    /// 累计 LLM 调用次数（不含工具）。命中上限 → MaxTurnRequests 终止。
    request_count: u32,
    /// 本 turn 的累计 token 用量。
    usage: Usage,
}

pub(crate) struct TurnConfig {
    /// LLM 调用次数上限。`None` 表示不限。详见 §6.1。
    pub max_turn_requests: Option<u32>,
    /// 压缩阈值的绝对值显式覆盖（token 数）。`None` 时按 compact_ratio 推算。
    pub compact_threshold_tokens: Option<u64>,
    /// 压缩阈值占模型 context_window 的比例。默认 `Some(0.85)`。详见 §4。
    pub compact_ratio: Option<f64>,
    /// 单次 LLM 调用的最大重试次数。详见 §7。
    pub max_llm_retries: u32,
    /// 一次性允许的并发 tool 数。`0` 视为不限。
    pub max_concurrent_tools: usize,
}
```

`TurnRunner` 是 `DefaultSession::run_turn` 内部的临时 owner——所有引用都是 `&'a` 借用 `Session` 内部字段，turn 结束后 `TurnRunner` 析构、`Session` 继续存活。

`SandboxPolicy` 与 `PermissionGate` 在本文之外的设计文档里展开（[`sandbox-policy.md`](./sandbox-policy.md) 待写）。本文只规定主循环对它们的调用形态。

## 3. 状态机正文（一次 LLM ↔ Tool 往返）

每一轮的伪代码：

```rust
loop {
    if cancel.is_cancelled() { return Ok(StopReason::Cancelled); }

    maybe_compact(&mut state).await?;

    let req = build_request(history.snapshot(), tools.schemas());
    let provider_stream = call_llm_with_retry(&req, &mut state).await?;

    let outcome = drain_provider_stream(provider_stream, &mut state).await?;
    match outcome.stop {
        StopReason::EndTurn      => return Ok(StopReason::EndTurn),
        StopReason::Refusal      => return Ok(StopReason::Refusal),
        StopReason::MaxTokens    => return Ok(StopReason::MaxTokens),
        StopReason::StopSequence => return Ok(StopReason::EndTurn), // 折叠
        StopReason::ToolUse      => { /* fall through */ }
    }

    // outcome.tool_uses: Vec<ToolUseAccumulated>
    let approved = decide_permissions(&outcome.tool_uses).await?;
    let results  = run_tools_concurrently(approved).await?;
    history.append(assistant_message(outcome));
    history.append(tool_result_message(results));

    if let Some(cap) = config.max_turn_requests {
        if state.request_count >= cap {
            return Ok(StopReason::MaxTurnRequests);
        }
    }
    // 没有 cap 或未到 cap：进入下一轮
}
```

返回类型沿用 [`session.md`](./session.md) §3 的 `Result<AcpStopReason, TurnError>`。`AcpStopReason` 与 LLM 的 `StopReason` 字段重叠的部分直接转换，剩下的（`Cancelled` / `MaxTurnRequests`）由主循环自己产出。

### 3.1 build_request

```rust
fn build_request(history: Vec<Message>, tools: Vec<ToolSchema>) -> CompletionRequest {
    CompletionRequest {
        model: config.model.clone(),
        system: config.system_prompt.clone(),
        messages: history,
        tools,
        tool_choice: ToolChoice::Auto,
        sampling: config.sampling.clone(),
    }
}
```

`ToolChoice` v0 固定 `Auto`；未来可以由调用方在 `run_turn` 入参里覆盖（trait 签名暂不变，加可选参数走 builder）。

### 3.2 drain_provider_stream

provider 流的语义见 [`llm-trait.md`](./llm-trait.md) §1。主循环消费它的同时维护 5 个累加器：

```rust
struct DrainOutcome {
    stop: StopReason,
    usage: Usage,
    text: Vec<ContentBlock>,        // 助手最终消息的内容
    tool_uses: Vec<ToolUseAccumulated>,
}

struct ToolUseAccumulated {
    id: String,           // LLM 给的 tool_use_id
    name: String,
    args_buf: String,     // ArgsDelta 拼接缓冲
}
```

处理规则：

| ProviderChunk | 主循环动作 |
| --- | --- |
| `MessageStart` | 记录 model id（用于 `LlmCallStarted` 已发出的 `attempt`），不发事件 |
| `TextDelta { text }` | emit `AssistantText { content: ContentBlock::text(text) }`；同时累入 `outcome.text` |
| `ThinkingDelta { text }` | emit `AssistantThought { content: ContentBlock::text(text) }`；累入 thinking buffer 用于 history append |
| `ThinkingSignature { signature }` | 仅累入 thinking 元数据（v0 不上 wire；多轮 thinking 复用时塞回 history） |
| `ToolUseStart { id, name }` | 在 `tool_uses` 表里 push 新条目 |
| `ToolUseArgsDelta { id, fragment }` | 找到对应条目，append 到 `args_buf` |
| `ToolUseEnd { id }` | no-op：v0 等流结束再 dispatch（见决策记录），仅作为完整性检查 |
| `Stop { reason }` | 设置 `outcome.stop = reason`，跳出 drain 循环 |
| `Usage(u)` | 逐字段累加到 `state.usage` 与 `outcome.usage` |

drain 期间检查 `cancel`：在 `next().await` 处用 `tokio::select!`，cancel 触发即 drop 流（取消 provider 调用）、emit `LlmCallFinished { error: Some("cancelled") }`、整个 turn 走 §5 的取消路径。

**v0 决定**：text / tool_use 在流上交织（`text → tool_use → text → tool_use → stop`）的情况下，主循环**等到 stop 后**再 dispatch 工具。理由：
- 简化状态机（drain 与 dispatch 不并发）
- tool 通常比 LLM 慢一个数量级，"边收边做"的并发收益小
- cancel 路径简单（只一处 await 工具）

### 3.3 decide_permissions：决策串行

```rust
async fn decide_permissions(tool_uses: &[ToolUseAccumulated]) -> Result<Vec<Approved>, TurnError> {
    let mut approved = Vec::new();
    for tu in tool_uses {
        let tool = tools.get(&tu.name).ok_or(TurnError::Internal(...))?;
        let args: serde_json::Value = match serde_json::from_str(&tu.args_buf) {
            Ok(v) => v,
            Err(e) => {
                // 失败回喂 LLM —— 不进 TurnError，记一个失败的 tool result
                approved.push(Approved::FailedArgs { id, reason: e.to_string() });
                continue;
            }
        };
        let hint = tool.safety_hint(&args);
        let id = ToolCallId::from(tu.id.clone());
        emit ToolCallStarted { id, fields: tool.describe(&args).fields };
        match policy.classify(&tu.name, hint, &args) {
            PolicyDecision::Allow => {
                emit PolicyDecision { id, decision: Allow };
                approved.push(Approved::Run { id, tool, args });
            }
            PolicyDecision::Deny => {
                emit PolicyDecision { id, decision: Deny };
                approved.push(Approved::Denied { id });
            }
            PolicyDecision::Ask => {
                emit PolicyDecision { id, decision: Ask };
                let outcome = permissions.wait(id, cancel.clone()).await;
                emit PermissionResolved { id, outcome };
                match outcome {
                    Selected { option_id } if policy.option_allows(option_id) =>
                        approved.push(Approved::Run { id, tool, args }),
                    Selected { .. } =>
                        approved.push(Approved::Denied { id }),
                    Cancelled => return Ok(StopReason::Cancelled), // 见 §5
                }
            }
        }
    }
    Ok(approved)
}
```

**为什么决策串行**：用户面对多个 `request_permission` 弹窗时，串行才能保证"上一个回答完才看到下一个"。多个 tool 并发请权限会让客户端 UI 状态混乱。

**为什么执行并发**：决策完成后已有明确的 Allow 列表，并发 spawn 不会触发新的用户交互；工具之间彼此独立（fs 读 / bash / mcp 调用），并发能显著缩短 turn 时间。

### 3.4 run_tools_concurrently

```rust
async fn run_tools_concurrently(approved: Vec<Approved>) -> Vec<ToolResult> {
    let mut joinset = JoinSet::new();
    let mut denied_results = Vec::new();

    for a in approved {
        match a {
            Approved::Run { id, tool, args } => {
                let ctx = ToolContext { cwd: &session.cwd, cancel: cancel.child_token() };
                let stream = tool.execute(args.clone(), ctx);
                joinset.spawn(drive_tool_stream(id, stream));
            }
            Approved::Denied { id } => {
                emit ToolCallFinished {
                    id,
                    fields: ToolCallUpdateFields::failed("denied by policy"),
                };
                denied_results.push(ToolResult::denied(id));
            }
            Approved::FailedArgs { id, reason } => {
                emit ToolCallFinished {
                    id,
                    fields: ToolCallUpdateFields::failed(format!("invalid args: {reason}")),
                };
                denied_results.push(ToolResult::error(id, reason));
            }
        }
    }

    // max_concurrent_tools == 0 ⇒ None（不限并发，快路径）。否则所有 tool task
    // 共享一个 Semaphore：每个 task 驱动工具流之前先 acquire_owned 抢 permit、
    // task 结束归还。给同 turn 一次发 N 个 spawn_agent（fanout）一个上限。
    let semaphore = (config.max_concurrent_tools > 0)
        .then(|| Arc::new(Semaphore::new(config.max_concurrent_tools)));
    // ...每个 joinset.spawn 内：let _permit = match &semaphore { Some(s) => Some(s.clone().acquire_owned().await?), None => None };

    let mut results = denied_results;
    while let Some(res) = joinset.join_next().await {
        results.push(res.expect("tool task panicked"));
    }

    results
}
```

`drive_tool_stream` 转发 `ToolEvent::Progress` 为 `AgentEvent::ToolCallProgress`、`Completed`/`Failed` 为 `ToolCallFinished`，并把工具最终输出装进 `ToolResultBody` 返回。

**panic 处理**：tool task panic 视为 invariant 破坏，`expect` 后冒泡到 `TurnError::Internal`（仍然先把已经在 history 里的 assistant message 落盘——见 §5 的"局部已 emit 事件"语义）。tool 自己的可恢复错误走 `ToolEvent::Failed`，不应触发 panic。

### 3.5 history append

完成一轮"LLM 输出 → 工具结果"后，主循环把这一对追加到 history：

```rust
history.append(Message {
    role: Role::Assistant,
    content: assistant_content_blocks(outcome),  // text + thinking + tool_use 元素
});
history.append(Message {
    role: Role::User,
    content: tool_results.into_iter().map(MessageContent::ToolResult).collect(),
});
```

`MessageContent` 已经按 [`request.rs`](../../crates/agent/src/llm/request.rs) §`MessageContent` 的形状统一好——OpenAI 风格的"分离 assistant + tool message"由 codec 在 provider 层翻译，主循环只用 Anthropic 风格的"messages 数组里都是 turn"。

## 4. 压缩（编排在主循环，不在 History）

摘要要调 LLM，存储抽象（`History`）够不到 provider，所以压缩编排放在主循环这层
（对齐 codex / opencode / Claude Code）。`History` 只提供 `snapshot` / `replace` /
`record_input_tokens` / `token_estimate`。实现见 `session/turn/compact.rs`。

```rust
async fn maybe_compact(&self) -> Result<(), TurnError> {
    let Some(threshold) = self.compact_threshold() else { return Ok(()) };
    let Some(estimate)  = self.history.token_estimate() else { return Ok(()) };
    if estimate < threshold { return Ok(()); }

    // run：选边界 → 调 LLM 摘要 → 重建 [summary ++ tail] → history.replace。
    // None = 无安全边界，最佳努力跳过（不杀 turn）。
    let Some(report) = compact::run(self, threshold).await? else { return Ok(()) };
    emit ContextCompressed { report.tokens_before, report.tokens_after };
    Ok(())
}
```

**阈值解析**（`compact_threshold`）：

1. `compact_threshold_tokens`（绝对值）显式覆盖；
2. 否则 `model_info(model).context_window * compact_ratio`（默认 `0.85`）；
3. 两者都拿不到 → 不压缩。

**计量**：`token_estimate` 以上一次调用回报的真实输入 token 为基线 + 其后新增消息的
`chars/4` 增量（详见 [`session.md`](./session.md) §4）。主循环在每次 `LlmCallFinished`
后调 `history.record_input_tokens(input + cache_read + cache_creation)` 刷新基线。

**保留策略**（结构化摘要 + 边界，Claude 型）：

- `select_boundary`：把历史切成「待摘要 head」+「原样保留 tail」。边界**对齐到轮次
  起点**（含非 ToolResult 内容的真实 user 消息），保证 tail 以合法 user 轮开头、且
  绝不切散 `tool_use`↔`tool_result` 配对（两个 wire codec 都不校验配对，必须自保）。
  tail 预算 `clamp(threshold/4, 2k, 8k)`（对齐 opencode）。无安全边界返回 `None`。
- `summarize`：用当前 provider/model 对 head 跑一次 `tool_choice=None` 的子请求，按
  固定结构化 markdown 模板产摘要；检出旧摘要（`SUMMARY_PREFIX` 起头）走 `<previous
  -summary>` 增量合并；head 里的超长 tool_result 截到 ~2k 字符、图片剥离。
- 重建：`[合成 assistant 摘要消息(SUMMARY_PREFIX + 正文)] ++ tail`，`history.replace`。
  摘要用 assistant 角色 + tail 必以真实 user 轮开头 ⇒ 角色交替对两家 codec 都合法。

**约定**：

- 失败（无边界 / provider 错 / 空摘要 / 取消）一律最佳努力降级、跳过本次，不杀 turn
- 压缩只在 LLM 调用之前发生，不打断已经 in-flight 的工具
- 可选未做项（Phase 2）：microcompact——原地把旧 tool_result 替换成占位、不调 LLM

## 5. 取消语义

cancel 是**正常路径**——主循环捕获 `cancel.is_cancelled()` 或 `cancel.cancelled().await` 后：

1. 取消所有 in-flight 工具：`JoinSet` 内的 task 通过 `ctx.cancel`（child token）感知后退出
2. 清空尚未 dispatch 的 approved 工具
3. 如果在 LLM stream 中，drop provider stream（trait 契约：drop 视为取消）
4. **不**追加未完成的 assistant message 到 history（避免半截消息）
5. 已经发出的 `AgentEvent` 不撤回（事件流是 append-only）
6. emit `TurnEnded { reason: Cancelled, usage: state.usage }`
7. 返回 `Ok(StopReason::Cancelled)`

**pending request_permission 的 cancel**：`PermissionGate::wait` 内部用 `tokio::select!` 监听 `cancel.cancelled()` 与 oneshot receiver。cancel 先到则返回 `PermissionResolution::Cancelled`，主循环把它当作"用户拒绝该工具"处理；与此同时 acp 桥接层那一侧也已经 respond `RequestPermissionOutcome::Cancelled`（见 [`acp-bridge.md`](../inbound/acp-bridge.md) §4）。

幂等：cancel 后的 turn 终止再次 cancel 是 no-op。

## 6. Turn 上限

### 6.1 max_turn_requests 是软上限

```rust
pub enum TurnRequestLimit {
    Unbounded,                 // 不设上限
    Fixed(u32),                // 固定上限
    Adaptive { initial: u32, expand_on_progress: bool }, // 进展中可扩
}
```

`TurnConfig::max_turn_requests` 实际对应 `TurnRequestLimit`。语义：

- `Unbounded`：完全不计数，不做检查。允许真正"长跑"的 agent 任务（codex-style 长周期工作流）
- `Fixed(N)`：达到 N 后下一轮调用前返回 `StopReason::MaxTurnRequests`
- `Adaptive { initial, expand_on_progress: true }`：每当本轮**有 tool_use 被批准执行**视为"在推进"，计数上限自动 +1（有节制地扩容）；如果某一轮只有助手文本无工具调用、而上限又到了，按 `Fixed` 终止

为什么不用纯 `Unbounded`：模型偶尔会陷入"call tool → 看 result → call same tool → ..."的死循环；adaptive 提供一个"当真的有进展才放行"的简单启发式，避免烧 token。具体进展判定可以演进，trait 不变。

v0 默认值：`Adaptive { initial: 32, expand_on_progress: true }`。

### 6.2 与 token 上限的关系

`max_turn_requests` 是**调用次数**上限，与 token 数无关。token 是在 §4 由 `compact` 钩子负责。两个上限独立：模型可以跑 200 次小调用而 token 不爆，也可以一次 LLM 调用就把 context 撑满。

## 7. LLM 重试

```rust
async fn call_llm_with_retry(req: &CompletionRequest, state: &mut TurnState)
    -> Result<ProviderStream, TurnError>
{
    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        state.request_count += 1;
        emit LlmCallStarted { model: req.model.clone(), attempt };

        let result = provider.complete(req.clone(), cancel.clone()).await;
        match result {
            Ok(stream) => {
                emit LlmCallFinished { model, attempt, usage: Usage::default(), error: None };
                return Ok(stream);
            }
            Err(err) => {
                let hint = err.retry_hint();
                emit LlmCallFinished {
                    model, attempt,
                    usage: Usage::default(),
                    error: Some(err.to_string()),
                };
                if attempt >= config.max_llm_retries || hint == RetryAction::DoNotRetry {
                    return Err(TurnError::Provider(err));
                }
                if let Some(delay) = hint.backoff() {
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }
}
```

要点：
- 重试**不发** `AgentEvent::TurnEnded`——是 turn 内部错误而非 turn 终止
- `LlmCallFinished` 的 `error` 字段让 storage / tracing 看得到失败链路
- `RetryAction` / `RetryHint` 来自 [`llm-trait.md`](./llm-trait.md) §6.4
- 用尽重试后 `TurnError::Provider`，事件流上只发了 N 个 `LlmCallFinished`、没有 `TurnEnded`——acp 桥接层据 future outcome 决定 respond JSON-RPC `Error`（见 [`session.md`](./session.md) §3.1）

stream 中途出现的 `Err(ProviderError)`（流上的错误，不是 future 的错误）走同一套重试逻辑：drop 当前 stream、attempt + 1、重新 `provider.complete`——前提是已经收到的 chunk 还没 push 进 history（见 §3.5：history.append 只在 stop 之后发生）。

## 8. 错误归类速查

| 来源 | 归宿 | 备注 |
| --- | --- | --- |
| 用户 `cancel_turn()` | `Ok(StopReason::Cancelled)` | §5 |
| LLM EndTurn / Refusal / MaxTokens | `Ok(StopReason::*)` 直接转 ACP | §3 |
| LLM ToolUse | 进入工具决策 | §3.3 |
| LLM 单次调用失败可重试 | retry，事件流上发 `LlmCallFinished{error}` | §7 |
| LLM 重试用尽 | `Err(TurnError::Provider(_))` | §7 |
| 工具 args 解析失败 | tool_result with is_error=true，回喂 LLM | §3.3 |
| 工具执行失败（`ToolEvent::Failed`） | 同上 | §3.4 |
| 工具 task panic | `Err(TurnError::Internal(_))` | §3.4 |
| `ToolRegistry::get(name)` miss（LLM 编了不存在的工具名） | tool_result with is_error=true，回喂 LLM | §3.3 |
| 压缩失败（无边界 / provider 错 / 空摘要 / 取消） | 最佳努力跳过本次，不杀 turn | §4 |
| 命中 `MaxTurnRequests` | `Ok(StopReason::MaxTurnRequests)` | §6 |

`PermissionResolution::Cancelled` 在 §3.3 中触发 `Ok(StopReason::Cancelled)`——因为只有用户 cancel 整个 turn 时才会让 permission 走 cancel 分支（见 §5）。

## 9. 配置默认值（v0）

| 配置 | 默认值 | 备注 |
| --- | --- | --- |
| `max_turn_requests` | `Adaptive { initial: 32, expand_on_progress: true }` | §6 |
| `compact_threshold_tokens` | `None` | 绝对值显式覆盖；`None` 时按 `compact_ratio` 推算 |
| `compact_ratio` | `Some(0.85)` | 阈值 = `context_window * 0.85`；模型不公开 window 时不压缩 |
| `max_llm_retries` | `3` | provider transient error 默认 3 次 |
| `max_concurrent_tools` | `0`（不限） | `>0` 时所有 tool task 共享 Semaphore 限并发；主要约束同 turn 多个 spawn_agent 的 fanout |

所有默认值都过 `defect-config`，最终落到 `TurnConfig`；本文不细化 config 加载路径。

## 10. 演进口子

- **mid-stream tool dispatch**：流上看到 `ToolUseEnd` 立即 dispatch 不等 `Stop`。需要把 drain 与 dispatch 改为并发——`drain_provider_stream` 内部 spawn 工具 task；cancel 路径要新增 "drain task ↔ tools task" 的协调。trait 不动。
- **批量 prefetch context**：在 LLM 调用之前并发 fetch 一组 RAG 资源、装进 system message。需要在 §3.1 之前插一个 hook（可能进 trait 化，比如 `PromptEnricher`），但本文范围之外。
- **多 turn 并发**：`Session::run_turn` 串行的契约（[session.md §3.2](./session.md#32-单-turn-互斥)）目前由实现用 mutex 保证；要并发时 trait 需要返回 turn handle 而非 future。
- **adaptive 进展判定**：当前用"本轮有 tool_use 被批准"作为进展信号；可演进成"工具结果对 LLM 输出有影响"（更准但更复杂）。
- **streaming tool_result 回 LLM**：v0 工具完整跑完才送回 LLM；future 想让 LLM 边收 tool 进度边推理。需要 provider 协议层支持 streaming user message——v0 不做。

## 11. 落地节奏

trait 不需要再扩。具体类型按下列顺序写：

1. `EventEmitter`（mpsc fan-out 实现，配合 `Session::subscribe`）
2. `PermissionGate`（`DashMap<ToolCallId, oneshot::Sender<PermissionResolution>>`）
3. `TurnRunner` 主体（按本文 §3 的状态机）
4. `DefaultSession` 把 `run_turn` 委托给 `TurnRunner::run`
5. e2e：mock provider + mock tool + mock policy 跑一次 turn

测试策略详见 [`docs/testing/e2e.md`](../testing/e2e.md)（待写）。
