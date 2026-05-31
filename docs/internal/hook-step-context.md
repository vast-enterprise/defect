# Hook Step Context（typestate + 信封）设计

> 状态：**已落地**。14 个 step 类型全部实现（`crates/agent/src/hooks/step.rs`），经
> `HookEngine::dispatch` 接进 turn 主循环的全部挂载点，旧 `HookEvent`/`HookOutcome` 引擎已移除。
> 本文描述**当前实现**；与最初设计（[`docs/proposals/sync-hook-control-model.md`]）的偏差在文中标注。

## 1. 一句话

每个挂载点（§6 of proposal）对应**一个独立的 step 类型**（typestate）。同一份 step state 被两种
hook 消费——内部 Rust hook 直接拿强类型，用户配置 hook 经 JSON 信封——**能力对等、可见面一致**，
只是表达媒介不同。

## 2. 两条公理

**公理一：typestate。** 不用一个带 variant 字段的大 enum，而是每个挂载点一个具体 struct。好处：
- 可见面在编译期锁死——`BeforeGenerate` 类型里没有 tool result 字段，越权访问编译不过。
- **`Option` 的有/无直接编码"已产出 / 将产出"**：已产出的数据是非 Option（不用 unwrap、不留
  "这里怎么会 None"的雷）；将产出的是 `Option`（`None` = 真去跑这一步；**被填上 = short-circuit**，
  用填的值当产出、跳过真实调用）。

**公理二：调用型 vs 变更型不对称。** 别硬把所有 step 套进同一个壳：
- **调用型**（Generate / ToolApply / Permission）：有可分离的入参 `x` 和产出 `y`，形如 `y = f(x)`。
  `before` 暴露**可改的入参** + **将产出的 `Option<y>`**；填上 `Option` = short-circuit。
  这一族里 **"拦掉一个工具" 不再是特例**——它字面上就是 `before ToolApply` 里 `result = Some(合成)`。
- **变更型**（Compact / Ingest）：原地改 history，没有可分离的"结果对象"能提前填。它们的 short-circuit
  退化成 **veto**（Compact：别压缩）或 **rewrite**（Ingest：改写待摄入输入），不走"填 Option"。

## 3. 借用模型：step 是 owned 数据，不持任何引用

> **与 proposal 的偏差**：proposal 设想 step 持 `&dyn History`、hook 通过 `ctx.history.append`
> 注入。实现没这么做——选了更简单、无生命周期纠缠的 **owned** 形态。

每个 step 是一个**自包含的 owned struct**（无 `&`、无 `&mut`、无生命周期参数，故天然 `Send`，能跨
`dispatch` 的 `.await`）。注入与改写都落在 step 自己的字段上：

- **注入** → step 上的 `Vec<ContentBlock>` 字段（`additional_context` / `feedback`）。`apply_verdict`
  把 verdict 的 `additional_context` 追加进去；内部 Rust hook 直接 push。
- **改入参** → step 上的 owned 字段（`args: Value` / `model: String` / `input: Vec<..>`）。
- **填产出（short-circuit）** → step 上的 `Option` 字段（`result` / `assistant_text` / `resolved`）。

`dispatch` 返回后，**调用方（turn 主循环）读回 step 上的字段**，自己决定怎么落地：续命反馈作为
user 消息 `append` 进 history、改过的 args 喂给工具、填好的 result 当工具输出……

这样 history 的实际写入仍由主循环做（它本就持有 `history`），step 只承载"hook 想要什么"。好处：
step 不碰 `TurnRunner` 的任何借用，`HookEngine::dispatch(&mut dyn HookStep, ctx)` 是干净的 trait
对象签名，没有"和 `&'a` 字段打架"的问题。

## 4. 控制流返回值：`HookControl`

> **与 proposal 的偏差**：proposal 写 `std::ops::ControlFlow<TurnEnd, ()>`，但 std 的 `ControlFlow`
> 只有两支，装不下 `Skip`。实现用自定义枚举 `HookControl`。

`apply_verdict` 返回、`dispatch` 合并出的 [`HookControl`]：
- `Proceed` —— 不干预控制流（step 上的数据改动仍生效）。对应信封 `control: null` / 缺省。
- `Break { reason: AcpStopReason }` —— 结束 turn，带停止原因。任何 step 可用。
- `Continue` —— 不结束、回循环顶再转一轮。仅 `before turn-end` 有意义。
- `Skip` —— 跳过本 step 的真实调用。仅 `before Compact`（veto 压缩）有意义。

控制流**无 payload**：注入/改写已经落在 step 字段上（§3），不用再"带"出来。停止点的不变量仍在——
`before turn-end` 返回 `Continue` 时其 `feedback` 字段必须非空（否则 LLM 下一轮立刻又说完 → 死循环），
但"反馈"是 step 上的 `feedback` 字段，不是 `Continue` 的负载。

**多 handler 合并**（`run_step_pipeline` / `dispatch`）：数据轴累积（每个 verdict 依次落到同一个 step，
后者看到前者改写后的状态），控制轴早退（任一非 `Proceed` 即停止 pipeline 并返回该指示）。

## 5. Step 类型 ↔ 信封 schema 一览

下表每行：内部 Rust struct 的关键字段（`&` 只读 / `&mut` 可改 / `Option` 将产出）+ 对外 JSON 信封。
信封是 step state 的投影——**用户 hook 在信封里看到的字段和 Rust hook 拿到的一致**；用户 hook 在
输出 JSON 里"填将产出的字段"等价于 Rust hook 填那个 `Option`。

每个 step 实现 `HookStep` 两个方法：`to_envelope(&self) -> Value`（喂 command stdin / prompt 模板）、
`apply_verdict(&mut self, &Value) -> Result<HookControl, VerdictError>`（把 handler 的 JSON 输出
解析回 step + 控制流）。

> **通用头**：`dispatch` 在每个 step 的 `to_envelope()` 产物上**统一并入** `{session_id, cwd,
> hook_event}`（step 自身不持 ctx，故由引擎补上）。下表只列 step 专属字段，通用头不重复。

### 5.1 作用域 step（无产出，可注入 / 可 break）

| step | Rust 关键字段（owned） | 信封专属字段 | 输出侧可控 |
| ---- | ------------- | -------------- | ---------- |
| `after_session_enter` | `cwd: String`，`source: New\|Resume`，`additional_context: Vec` | `{cwd, source}` | `additional_context`（注入 system 后缀）、`control: break` |
| `after_turn_enter` | `is_subagent: bool`，`agent_type: Option<String>`，`additional_context: Vec` | `{is_subagent, agent_type}` | `additional_context`、`control: break` |

### 5.2 Ingest（变更型：rewrite 输入 / veto）

| step | Rust 关键字段（owned） | 信封专属字段 | 输出侧可控 |
| ---- | ------------- | -------------- | ---------- |
| `before_ingest` | `input: Vec<ContentBlock>`，`source: User\|Continuation` | `{input（拼接文本）, source, input_len}` | `input`（字符串或字符串数组，整条改写）、`control: break`（拒该 turn） |
| `after_ingest` | `committed_len: usize`，`additional_context: Vec` | `{committed_len}` | `additional_context`（注入） |

> Ingest 的 short-circuit 是 `break`（拒掉这个 turn），不是"填结果"——它没有可分离产出。
> 空摄入轮 `input` 为空，hook 仍可注入或 break。
> **与 proposal 偏差**：`after_ingest` 信封是 `committed_len`（长度）而非 `committed_input`（内容）。

### 5.3 Compact（变更型：veto only）

| step | Rust 关键字段（owned） | 信封专属字段 | 输出侧可控 |
| ---- | ------------- | -------------- | ---------- |
| `before_compact` | `token_estimate: u64`，`threshold: u64` | `{token_estimate, threshold}` | `control: skip`（否决本次压缩）、`control: break` |
| `after_compact` | `tokens_before: u64`，`tokens_after: u64`，`additional_context: Vec` | `{tokens_before, tokens_after}` | `additional_context` |

> Compact 无"提前填结果"——它原地重写 history，没有结果对象。short-circuit = `skip`（veto），
> 落地为"`before Compact` 返回 skip → 不调 `compact::run`"。

### 5.4 Generate（调用型：改 request / short-circuit）

| step | Rust 关键字段（owned） | 信封专属字段 | 输出侧可控 |
| ---- | ------------- | -------------- | ---------- |
| `before_generate` | `model: String`，`message_count: usize`，`attempt: u32`，`assistant_text: Option<String>` | `{model, message_count, attempt}` | `model`（改模型）、填 `assistant`（合成文本 short-circuit 跳过 LLM）、`control: break` |
| `after_generate` | `model: String`，`usage: Usage`，`stop: AcpStopReason`，`error: Option<String>` | `{model, usage, stop_reason, error}` | — （观察；要干预下一轮走 before turn-end） |

> **与 proposal 偏差**：`before_generate` 只暴露 `model`（不是整个 `&mut CompletionRequest`）+
> `assistant_text: Option<String>`（合成文本，不是完整 `Message`）。改 sampling 等其它 request 字段
> 的能力**未实现**——见 §8 已知缺口。

### 5.5 Permission（调用型：用户是外部资源；v0 仅打桩）

| step | Rust 关键字段（owned） | 信封专属字段 | 输出侧可控 |
| ---- | ------------- | -------------- | ---------- |
| `before_permission` | `tool: String`，`decision: String`（"allow"/"deny"/"ask"），`resolved: Option<bool>` | `{tool, decision}` | （未来）填 `resolved` 代答；v0 解析但**主循环不消费** |
| `after_permission` | `tool: String`，`granted: bool` | `{tool, granted}` | — |

> v0：两个边界**仅 observe**——主循环 emit step 但不消费 `resolved`（policy 仍是放行权威，见
> hooks.md §7.3）。桩留好，未来开。
> **与 proposal 偏差**：`decision` 是字符串而非 typed `PolicyDecision`（信封要 JSON 友好）。

### 5.6 ToolApply（调用型：改 args / 拦工具；per-tool + 批）

| step | Rust 关键字段（owned） | 信封专属字段 | 输出侧可控 |
| ---- | ------------- | -------------- | ---------- |
| `before_tool_apply` | `tool_name: String`，`safety: SafetyClass`，`args: Value`，`result: Option<SyntheticToolResult>` | `{tool, safety, args}` | `args`（改参）、填 `result`（拦工具=合成输出，turn 继续）、`control: break`（结束 turn） |
| `after_tool_apply` | `tool_name: String`，`is_error: bool`，`output: ToolResultBody`，`additional_context: Vec` | `{tool, is_error, output}` | `additional_context`（拼进 tool_result）、`control: break` |
| `after_tool_batch` | `results: Vec<ToolBatchEntry{tool_name,is_error}>`，`additional_context: Vec` | `{results: [{tool, is_error}]}` | `additional_context`、`control: break` |

> `before_tool_apply` 填 `result` 与 `control: break` 的区别是模型的关键：**填 result = 拦这一个工具
> （注入合成输出，turn 继续）**；**break = 结束整个 turn**。两者都曾被叫"block"，控制流完全不同。
> `safety` 字段进信封供 matcher 的 safety 过滤。信封统一用 `tool` 键（非 `id`/`name`）。

### 5.7 before turn-end（控制分叉点：默认 Break）

| step | Rust 关键字段（owned） | 信封专属字段 | 输出侧可控 |
| ---- | ------------- | -------------- | ---------- |
| `before_turn_end` | `stop_reason: AcpStopReason`，`continues_so_far: u32`，`voluntary: bool`，`feedback: Vec` | `{stop_reason, continues_so_far, voluntary}` | `control: continue`（+ `additional_context` → 填进 `feedback`）、默认（缺省）= `break` |

> 唯一默认 `Break` 的 step（缺省 verdict = `Proceed`，主循环把它当作"放停"）。`continue` 只在
> `voluntary == true` 时生效（被动停止——Refusal / MaxTokens / Cancelled / MaxTurnRequests——
> 忽略 continue，否则绕过 request cap）。`continues_so_far` 暴露给 hook 判断收手，循环内另有硬上限
> `MAX_STOP_HOOK_CONTINUES = 3` 兜底。续命反馈：verdict 的 `additional_context` 进 step 的 `feedback`，
> 主循环 dispatch 后读出、作为 user 消息注入 history（末尾角色兜底，见 sync-hook 提案 §4）。

## 6. 信封通用约定

- **输入侧**：通用头 `{session_id, cwd, hook_event}`（由 `dispatch` 统一并入）+ 各表的 step 专属字段。
  （`is_subagent` / `agent_type` 不是全局通用头，只在 `after_turn_enter` step 上携带。）
- **输出侧（handler verdict → 引擎）**：`control: "proceed" | "continue" | "break" | "skip" | null`
  （`break` 可带 `stop_reason`）、`additional_context: [string]`（每条转一个文本块）；step 专属的
  "填产出"字段（`result` / `assistant` / `resolved` / `input` / `args` / `model`）按 5.x 各表。
  null / 缺省 = 不干预。形态错误 → `VerdictError`，引擎降级（warn + 跳过该 handler）。
- **字段名是对外 API**，发布后难改——本表即冻结基线。新增字段向后兼容，改名/删字段要走破坏性变更。

## 7. 落地现状

**已落地**：14 个 step 类型 + `HookStep` trait + `HookControl` + `run_step_pipeline`
（`crates/agent/src/hooks/step.rs`）；`HookEngine::dispatch` 按 `event_name` 路由、并入通用头、跑
matcher（tool / glob / safety）、合并 verdict；turn 主循环全部挂载点经 `dispatch` 接入
（`session/turn/hooks.rs`、`session/turn/tools.rs`、`session/turn.rs`、`session/default.rs`）；
三种 handler（builtin / command / prompt）实现 `StepHandler`；旧 `HookEvent`/`HookOutcome` 引擎已删除。

## 8. 已知缺口（与 proposal 的功能差距，待后续）

- **`before_generate` 只能改 model**：proposal 设想可改整个 `CompletionRequest`（sampling 等）。
  实现只暴露 `model` + 合成文本 `assistant_text`。要全 request 干预需扩 step 字段。
- **`before_permission` 代答未接**：`resolved` 字段已解析，但主循环 v0 不消费（仅 observe）。
- **`after_tool_apply` 的 `output`** 多模态结果在信封里退化成文本摘要（图片块标注占位），非完整结构。

[`docs/proposals/sync-hook-control-model.md`]: ../proposals/sync-hook-control-model.md
