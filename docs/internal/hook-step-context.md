# Hook Step Context（typestate + 信封）设计

> 状态：**设计定稿，待落地**。是 [`docs/proposals/sync-hook-control-model.md`] 的落地第 1 步——
> 把"hook 拿 ctx、改它=注入、返回 ControlFlow"这个抽象，落成具体的 Rust 类型与对外 JSON 信封。

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

## 3. 借用模型：为什么不需要 `&mut`

proposal 里写的 `&mut LoopContext` 是没看 `History` 实现时的想当然。实际上 `History` 的所有变更方法
（`append` / `replace` / `record_input_tokens`）都是 **`&self`**（`VecHistory` 内部 `Mutex`，见
`session/history.rs`）。所以：

- **改 history 类注入** → step 持 `&dyn History` 共享引用，调 `&self` 方法即可。**无可变借用**，
  也就没有"和 `TurnRunner` 那堆 `&'a` 字段打架"的问题——这块预想的硬骨头不存在。
- **改入参类注入**（args / request 这种还没进 history 的瞬时输入）→ 它们是 `run_inner` 栈上的局部
  变量，step 持对**局部变量**的 `&mut`，与 `TurnRunner` 字段无关、不冲突。

两类注入路径天然分离，各自都无借用冲突。step 因此是个**零拥有的视图**：借 `TurnRunner` 已持有的
引用 + 当前位置的局部数据，打包传给 hook。

## 4. 控制流返回值

hook 返回 `std::ops::ControlFlow<TurnEnd, ()>`：
- `Break(TurnEnd)` —— 结束 turn，带最终 outcome。任何 step 都能 break。
- `Continue(())` —— 不结束。**无 payload**：注入已经通过 step 改 history / 改入参 / 填 Option 落地了，
  不用再"带"出来。

> proposal 之前推的"Continue 带注入物"，在 typestate 下进一步简化：注入是改 ctx 的副作用，`Continue`
> 本身空载。停止点（`before turn-end`）的不变量仍在——它的 `Continue` 必须**先往 history 注入**
> （否则 LLM 下一轮立刻又说完 → 死循环），但"注入"是调 `ctx.history.append`，不是 `Continue` 的字段。

## 5. Step 类型 ↔ 信封 schema 一览

下表每行：内部 Rust struct 的关键字段（`&` 只读 / `&mut` 可改 / `Option` 将产出）+ 对外 JSON 信封。
信封是 step state 的投影——**用户 hook 在信封里看到的字段和 Rust hook 拿到的一致**；用户 hook 在
输出 JSON 里"填将产出的字段"等价于 Rust hook 填那个 `Option`。

每个 step 实现两个方法：`to_envelope(&self) -> Value`（喂 command stdin / prompt 模板）、
`apply_verdict(&mut self, Value) -> ControlFlow<…>`（把 handler 的 JSON 输出解析回 step / 控制流）。

字段类型对齐现有代码：`Usage{input_tokens,output_tokens,cache_read_input_tokens,
cache_creation_input_tokens: Option<u64>}`、`ToolResult{id,name,body,is_error,…}`、
`PolicyDecision = Allow|Deny|Ask{options:[{id,name,kind,allows}]}`。

### 5.1 作用域 step（无产出，可注入 / 可 break）

| step | Rust 关键字段 | 信封（输入侧） | 输出侧可控 |
| ---- | ------------- | -------------- | ---------- |
| `after session enter` | `cwd: &Path`，`source: New\|Resume`，`history: &dyn History` | `{session_id, cwd, source}` | `additional_context`（注入 system 后缀）、`control: break` |
| `after turn enter` | `history: &dyn History`（本轮输入尚未摄入） | `{session_id, cwd, is_subagent, agent_type?}` | `additional_context`、`control: break` |

### 5.2 Ingest（变更型：rewrite 输入 / veto）

| step | Rust 关键字段 | 信封（输入侧） | 输出侧可控 |
| ---- | ------------- | -------------- | ---------- |
| `before Ingest` | `input: &mut Vec<ContentBlock>`，`source: User\|Continuation`，`history: &dyn History` | `{input, source, history_tail?}` | `input`（改写整条待摄入输入）、`control: break`（拒该 turn） |
| `after Ingest` | `history: &dyn History`（输入已并入） | `{committed_input}` | `additional_context`（注入） |

> Ingest 的 short-circuit 是 `break`（拒掉这个 turn / 这轮），不是"填结果"——它没有可分离产出。
> 空摄入轮（纯推理续命之外的轮）`input` 为空，hook 仍可注入或 break。

### 5.3 Compact（变更型：veto only）

| step | Rust 关键字段 | 信封（输入侧） | 输出侧可控 |
| ---- | ------------- | -------------- | ---------- |
| `before Compact` | `token_estimate: u64`，`threshold: u64`，`history: &dyn History` | `{token_estimate, threshold}` | `control: skip`（否决本次压缩）、`control: break` |
| `after Compact` | `tokens_before: u64`，`tokens_after: u64`，`history: &dyn History` | `{tokens_before, tokens_after}` | `additional_context` |

> Compact 无"提前填结果"——它原地重写 history，没有结果对象。short-circuit = `skip`（veto），
> 落地为"`before Compact` 返回 skip → 不调 `compact::run`"。

### 5.4 Generate（调用型：改 request / short-circuit）

| step | Rust 关键字段 | 信封（输入侧） | 输出侧可控 |
| ---- | ------------- | -------------- | ---------- |
| `before Generate` | `request: &mut CompletionRequest`，`assistant: Option<Message>` | `{model, message_count, attempt}` | `request` 局部字段（model/sampling…）、填 `assistant`（short-circuit 跳过 LLM）、`control: break` |
| `after Generate` | `usage: &Usage`，`stop: StopReason`，`error: Option<&str>`，`assistant: &Message`（已产出，非 Option） | `{model, usage, stop_reason, error?}` | — （观察；要干预下一轮走 before turn-end） |

> `before Generate` 填 `assistant` = 用一条合成回复跳过真实 LLM 调用（罕见，但模型上自洽）。
> request 是 `run_inner` 局部（`:252` `build_request`），改它走 `&mut`。

### 5.5 Permission（调用型：用户是外部资源；v0 仅打桩）

| step | Rust 关键字段 | 信封（输入侧） | 输出侧可控 |
| ---- | ------------- | -------------- | ---------- |
| `before Permission` | `tool: &str`，`decision: &mut PolicyDecision`，`resolved: Option<bool>` | `{tool, decision}` | 填 `resolved`（代答 allow/deny，跳过问用户）、`control: break` |
| `after Permission` | `tool: &str`，`granted: bool`（已产出） | `{tool, granted}` | — |

> v0：两个边界**仅打桩 observe**，输出侧的代答能力先不接（policy 仍是放行权威，见 hooks.md §7.3）。
> 桩留好，未来开。

### 5.6 ToolApply（调用型：改 args / 拦工具；per-tool + 批）

| step | Rust 关键字段 | 信封（输入侧） | 输出侧可控 |
| ---- | ------------- | -------------- | ---------- |
| `before ToolApply` | `id`，`name: &str`，`args: &mut Value`，`safety`，`result: Option<ToolResult>` | `{id, name, args, safety}` | `args`（改参）、填 `result`（拦工具=合成输出，turn 继续）、`control: break`（结束 turn） |
| `after ToolApply` | `id`，`name: &str`，`result: &ToolResult`（已产出），`is_error: bool` | `{id, name, output, is_error}` | `additional_context`（拼进 tool_result）、`control: break` |
| `after ToolBatch` | `results: &[ToolResult]`（已产出） | `{results: [{id,name,is_error}]}` | `additional_context`、`control: break` |

> `before ToolApply` 填 `result` 与 `control: break` 的区别是模型的关键：**填 result = 拦这一个工具
> （注入合成输出，turn 继续）**；**break = 结束整个 turn**。两者都曾被叫"block"，控制流完全不同。

### 5.7 before turn-end（控制分叉点：默认 Break）

| step | Rust 关键字段 | 信封（输入侧） | 输出侧可控 |
| ---- | ------------- | -------------- | ---------- |
| `before turn-end` | `stop_reason: AcpStopReason`，`continues_so_far: u32`，`history: &dyn History`，`voluntary: bool` | `{stop_reason, continues_so_far, voluntary}` | `control: continue`（+ 先 `ctx.history.append` 注入反馈）、默认 `break` | **缺**（=需求） |

> 唯一默认 `Break` 的 step。`continue` 只在 `voluntary == true` 时生效（被动停止——Refusal /
> MaxTokens / Cancelled / MaxTurnRequests——忽略 continue，否则绕过 request cap）。`continues_so_far`
> 暴露给 hook 自己判断收手，循环内另有硬上限兜底。

## 6. 信封通用约定

- **输入侧**：每个信封都含通用头 `{session_id, cwd, hook_event, is_subagent, agent_type?}` + 上表的
  step 专属字段。
- **输出侧（handler → 引擎）**：通用字段 `control: "continue" | "break" | "skip" | null`、
  `additional_context: [ContentBlock]`；step 专属的"填产出"字段（`result` / `assistant` /
  `resolved` / `input` / `args` / `request`）按 5.x 各表。null / 缺省 = 不干预。
- **字段名是对外 API**，发布后难改——本表即冻结基线。新增字段向后兼容，改名/删字段要走破坏性变更。

## 7. 落地边界（本步交付什么）

本步只交付**类型与信封定义** + `to_envelope`/`apply_verdict`，**不接任何挂载点**（call site 接入是
后续 PR，见 proposal §B 路线）。验证方式：每个 step 类型的 `to_envelope`/`apply_verdict` 单测
（构造 step → 序列化信封 → 喂一个 mock verdict JSON → 断言解析回的控制流 / 字段改动）。

未定/留给后续 PR：`LoopContext` 如何在 `run_inner` 里逐 step 构造（依赖循环重构——Ingest 入循环、
统一 turn-end 判定）；现有 5 个 hook 迁移；防循环硬上限默认值。

[`docs/proposals/sync-hook-control-model.md`]: ../proposals/sync-hook-control-model.md
