# 任务编排：后台任务 + session 自持 input loop

> 状态：**已落地**。本文沉淀 `run_in_background`（fire-and-forget 子任务，结果跨 turn 回流）的
> 落地设计与实现。决议见 [§7](#7-决议)：**主动续转**，§5.1 选 (b)，§5.2 选排队，§5.3 持久 pump。
> 阶段一 + 阶段二（2a + 2b）均已实现、测试通过、端到端用户可达（wire 级 e2e 证明：客户端发一次
> prompt 后，后台任务的自主续转 turn 经持久 pump 把 `session/update` 送达客户端）。
> 转正动作（章节移入 `docs/internal/`）待后续。

> **前置澄清——"主 agent 等子 agent 完成再继续"现在就支持。** 那是 `spawn_agent` 当前的默认
> （同步阻塞）行为：`run_subagent` 一路 `runner.run(prompt).await`（`spawn_agent.rs:332`）等子
> turn 跑完才返回 tool_result，主 agent 在此期间不推进。本提案的 `run_in_background` 是一个 **opt-in
> 开关**，默认仍为阻塞等待；两种模式并存，模型按任务依赖关系自选——后续步骤依赖子结果就同步等待，
> 独立的就丢后台。与 Claude `Agent` 工具的 `run_in_background`（默认 false）一致。

## 1. 动机

目标对齐 Claude Code 的 `Agent { run_in_background: true }` / 后台 Bash：

- 工具（首要场景：`spawn_agent`）可以**异步**跑，发起它的 turn **不阻塞**等它；
- 任务在后台独立跑，**跑完后结果作为新输入回流进对话**，让 agent 主动继续（"re-invoke"，
  而非被动等下次用户开口才捎带）。

当前 `spawn_agent` 是**同步阻塞**的：`run_subagent` 一路 `runner.run(prompt).await` 等子 turn
整个跑完，才把最终文本作为 tool_result 返回（`crates/agent/src/tool/spawn_agent.rs:332`）。
turn loop 同 turn 内可以并发 fanout（`JoinSet`，见 [`turn-loop.md`](../internal/turn-loop.md) §3.4），
但**没有任何任务能活过发起它的 turn**。

## 2. 先厘清两个常见误判

落地前先排掉两条看似是障碍、其实不是的路；以及一条看似简单、其实是真障碍的路。

### 2.1 不是障碍：tool_use ↔ tool_result 配对契约

直觉担心：后台任务的结果晚于当前 turn 才回来，会留下一个没有 tool_result 的 tool_use，违反
Anthropic / OpenAI 的 wire 配对契约。

**这不是问题，只要设计正确。** 后台工具调用**当场同步返回**一个 tool_result —— 内容不是任务
产物，而是"任务已启动，id=X"。配对在当轮就满足，wire 契约毫发无伤。真正的产物以**新输入**的
形式在未来某轮回流，与原 tool_use 无配对关系。Claude 正是这么做的。

### 2.2 不是障碍：回注入轨道（hook 重构已造好）

直觉担心：需要从零造一条"把外部内容塞进对话、让 loop 继续转"的机制。

**这条轨道已经存在。** hook step 重构落地了 `before_turn_end` 续命：turn 自愿停止前，hook 可返回
`Continue`，把反馈作为输入注入 history、不结束、回循环顶再转一轮
（`crates/agent/src/session/turn/hooks.rs:38` `decide_turn_end` + `:77` `append_user_feedback`）。
这与"后台结果回流 → agent 继续"是**同一形状**。后台任务只是给这条轨道**新增一个触发源**。

> 注意：是复用这条轨道的**机制**，不是把后台任务挂成 hook —— 那样语义不对。后台任务不是
> "turn 结束前的扩展点"，它是一个独立的输入来源。

### 2.3 真障碍：两次 turn 之间，session 是静止的

致命事实：**defect 现在没有一个"等下一个输入"的常驻点。**

- `on_prompt`（`crates/acp/src/serve.rs:543`）每收到一条 `session/prompt`，就 `cx.spawn` 一个
  **一次性** `run_prompt_turn`（`:561`），内部 spawn `run_turn`（`:685`）。
- `run_prompt_turn` 里的 `tokio::select!`（`:692`）两条腿是 `events.next()`（投射事件到 wire）和
  `turn_rx`（turn 结束信号）——它等的是**"turn 跑完了吗"**，不是"下一个输入是什么"。
- turn 一结束，`run_prompt_turn` 返回、responder 应答、整个结构析构。**两次 prompt 之间没有任何
  循环在跑**，session 完全静止。
- 下一个用户输入靠**客户端再发一条 JSON-RPC**，被 acp 框架的连接 loop 接住、再 spawn 新的
  `run_turn`。**"等输入"发生在 ACP 协议层，不在 session 内，且是事件驱动（收到 RPC 才动），不是
  一个停着 select 的循环。**

所以"让用户输入和后台结果竞争一个 turn"的心智模型是对的目标，但它预设的那个"竞争点"现在不存在。
**最大改动不是"加竞争"，而是把驱动权从 ACP 的一次性 spawn，下沉成 session 自持的常驻 input loop。**
竞争 `select!` 是这个 loop 的自然产物。

## 3. 目标形态

两件可分开落地的事。

### 3.1 后台任务表（独立、风险低，可先做）

挂在 `DefaultSession` 上，与 `events` / `history` / `turn_state` 同档生命周期（session 级）：

```text
DefaultSession
  ├── turn_state:  Mutex<TurnSlot>
  ├── events:      Arc<EventEmitter>
  ├── history:     Box<dyn History>
  └── background:  Arc<BackgroundTasks>   ← 新增
```

`BackgroundTasks` 持有：
- 一张"运行中"表（`JoinHandle` + 元数据），让任务**活过发起它的 turn**——这直接解决
  `run_tools_concurrently` 里 `JoinSet`-drop-即-abort 的问题（任务不再被那个局部 `JoinSet` 持有）；
- 一个 **session 级 cancel token**（不是 turn 的子 token）——后台任务的取消生命周期独立于发起它
  的 turn；
- 一个"已完成、待回流"的队列（完成的任务结果排在这里，等下一个 input loop 迭代消费）。

**工具如何注册任务：** 工具在 `execute()` 里只拿得到 `ToolContext`，看不到 `DefaultSession`
（`spawn_agent.rs:70` 注释明示 `ToolContext` 只带 cwd/fs/shell/http/cancel/current_model）。
因此给 `ToolContext` 加一个 spawner 句柄：

```rust
pub struct ToolContext<'a> {
    // ... 现有字段
    pub background: Option<&'a BackgroundSpawner>,
}
```

工具想后台跑 → `ctx.background.spawn(...)`，当场拿回"任务已启动 id=X"的同步 tool_result（满足 §2.1
的配对契约），真正的 future 交给 spawner、parked 进 `DefaultSession::background`。spawner 同时持有
session 级 token，minted 给后台任务——"谁拥有后台任务取消生命周期"这个问题落到 spawner 头上，与
现在 cwd/fs/http 经 `ToolContext` 注入是同一模式，耦合最小。

> **注册通道加在 `ToolContext`，不在 `Session` trait。** `Session` trait 的新方法留给"外部
> list / cancel 后台任务"（类似 Claude 的 `TaskList`/`TaskStop`）——那是独立控制面，v0 可缓。

### 3.2 session 自持 input loop（架构转向，真正的大改）

把"一次性 spawn run_turn"换成 session 持有的常驻循环：

```rust
// 形态示意，非最终签名
loop {
    let input = tokio::select! {
        biased;
        done = self.background.next_completed()  => Input::Background(done),
        p    = self.prompt_rx.recv()             => match p {
            Some(p) => Input::User(p),
            None    => break,   // 连接关闭
        },
    };
    self.run_turn_inner(input.into_blocks(), input.source()).await;
    // turn 间停在 select 上：用户输入与后台结果在此竞争
}
```

`run_turn` 的主体（构造 `TurnRunner` 并 `runner.run`，`default.rs:729-769`）几乎不动，被这个 loop
调用而非被 ACP 直接调用。

## 4. 与当前实现的 diff（不是整段重写）

| 关注点 | 现状 | 改后 |
| --- | --- | --- |
| turn 驱动者 | ACP `on_prompt` → `cx.spawn(run_prompt_turn)` → `run_turn`（一次性） | session 自持 input loop 消费输入、起 turn |
| ACP 角色 | **驱动者**：主动起 turn | **投递者**：把 prompt 塞进 `prompt_rx` channel |
| turn 间状态 | 静止，无循环 | 常驻 loop 停在 `select!` 等输入 |
| 后台任务 | 不存在；任务被 turn 的 `JoinSet` drop 即 abort | session 级 `BackgroundTasks`，活过 turn |
| cancel token | 全部是 turn 子 token | 后台任务用 session 级独立 token |
| 单 turn 互斥 | `run_turn` 外层 `turn_state` slot，撞上返回 `TurnInProgress`（`default.rs:716`） | 发起权收归 loop 内部，外部不再直接调 `run_turn`，互斥语义重新表述 |
| 回流注入 | 仅 hook `before_turn_end` Continue（`turn/hooks.rs:54`） | 复用同一注入写法，新增"后台完成"触发源 |

## 5. 两个已拍板的设计点

### 5.1 后台结果以什么角色注入 → **选 (b)：新增 `IngestSource::Background`**

`append_user_feedback` 现在硬编码 `Role::User`（`turn/hooks.rs:86`）。后台结果本质是**被延迟的
tool 结果**，伪装成"用户说了句话"会让模型误以为是人在发话。

**决议：新增 `IngestSource::Background`。** `step.rs` 的 `IngestSource` enum 现在只有 `User` /
`Continuation`，已为这种扩展留好位置。走正派语义贯穿 ingest 路径，不用标注过的 user 块凑合
（被否的 (a) 方案）。Claude 用 system-reminder / 通知块，本质就是这个形态。落地时：

- `IngestSource` 加 `Background` 变体；`BeforeIngest::to_envelope` 的 `source` 字段相应输出
  `"background"`，让 hook / 投影都能区分这是后台回流而非用户输入。
- 注入 history 的消息体要明确携带任务来源标识（task id + profile），措辞按"延迟工具结果回流"而非
  "用户发言"组织，避免模型误判说话人。

### 5.2 ACP responder 怎么和"过会儿才被消费的 turn"对应 → **选排队**

现状：`on_prompt` 的 responder 同步绑定一个 `spawn(run_turn)`，turn 结束即 `responder.respond`
（`serve.rs:686-756`）。改成 input loop 后，一条 `session/prompt` 进来时，turn 可能**还没轮到执行**
（loop 也许正在消化一个后台结果），responder 不能再"spawn 即绑定"。

**决议：排队，不返回 `TurnInProgress`。** prompt 投递进 channel 时把 responder 一起带上
（`prompt_rx: Receiver<(Prompt, Responder)>`），由 loop 在真正起对应 turn 时持有、turn 结束时
respond。ACP 协议语义上 `session/prompt` 期望被处理，排队比直接拒更符合预期。落地时要定义的边角：

- **后台触发的 turn 没有对应 responder**（不是任何一条 `session/prompt` 引发的）——它的事件靠
  `session/update` 通知上 wire，不 respond 任何 request。所以 loop 内部要区分两类输入：带 responder
  的（用户 prompt，turn 结束要 respond）与不带的（后台回流，只发通知）。
- **队列深度与背压**：先用有界 channel；队列满时的行为（阻塞投递 / 拒绝）落地时定，倾向有界 + 投递
  端 await（背压回压到 ACP 层）。
- **cancel 语义**：`session/cancel` 现在取消"当前 turn"（`serve.rs:567` → `cancel_turn`）。排队模型下
  要明确：cancel 是只取消正在跑的 turn，还是连带清空队列中尚未起跑的 prompt？倾向只取消在跑的那个，
  队列保留——但需在落地时与 acp-bridge 的 cancel 投影对齐。

### 5.3 ACP 持久 event pump（落地时发现，原文档遗漏）

**事实**：ACP 当前**只在一次 prompt turn 期间订阅事件流**——`session.subscribe()` 是 `run_prompt_turn`
内部唯一的订阅点（`serve.rs:679`），turn 结束即 drop。

**后果**：主动续转里，一个由**后台完成触发**的 turn（没有对应 `session/prompt` 在飞）emit 的事件
**没有任何订阅者**——客户端看不到 agent 的自主工作（assistant 文本 / 工具调用全部丢失）。

**所以阶段二在 ACP 层是两件事，不是一件**：
1. session 自持 input loop / driver（agent crate）——"主动续转"的心脏，可隔离测试；
2. ACP 层一个**跨 turn 存活的持久 event pump**：在 `session/new` / `session/load` 时起一个常驻
   task，订阅该 session 事件流、把 `session/update` 通知一路转出去（含后台触发的自主 turn）。带
   responder 的用户 prompt turn 仍走原 respond 路径，二者在 pump 里按 turn 边界对齐。

**陷阱**：工具在飞的 ACP 反向请求（fs/shell 委托）其 oneshot 被 drop 时，server 把"无人接收"
映射成 internal_error 并**撕掉整条连接**（`tools.rs:525` 注释）。持久 pump 的生命周期管理必须保证
不在错误路径上误 drop 这些 oneshot。这是整个转向风险最高的一点。

## 6. 优劣与拐点

**好处**
- 后台任务结果**主动续转**，对齐 Claude 的 agent 自驱行为，而非被动等用户。
- input loop 是干净的归宿：长跑 agent、定时唤醒、未来的多输入源（cron / webhook）都能接到同一个
  `select!` 上。
- 后台任务表与 input loop **可分两步落地**，前者独立可测、风险低。

**坏处 / 成本**
- ACP 从驱动者变投递者是个**侵入性转向**，responder 重新绑定（§5.2）是真设计工作，不是机械改写。
- 单 turn 互斥语义、`TurnInProgress` 的对外契约要重新表述，可能影响 acp-bridge 的错误投影。
- 多了一个常驻 task 的生命周期要管（session drop 时如何收口 input loop + 在途后台任务）。

**拐点**
- 若只想要"后台任务能活过 turn、结果靠下次用户输入捎带回来"（**被动**回流），那么只做 §3.1 任务表
  即可，不必做 input loop —— 注入进 history、等下条 `session/prompt` 自然带出。成本骤降。
- 一旦要求"后台完成即主动续转"（**主动** re-invoke，Claude 行为），就必须做 §3.2 input loop，
  responder 重绑（§5.2）不可回避。**本提案已选主动续转，故两步都做。**

## 7. 决议

**已决议：主动续转**（对齐 Claude 的 agent 自驱行为，而非被动等用户开口）。三项关键选择：

- **续转模式**：主动 re-invoke —— 后台任务完成即竞争一个新 turn，§3.2 input loop 不可省。
- **§5.1 注入语义**：新增 `IngestSource::Background`，走正派 ingest 语义（否决"标注 user 块"）。
- **§5.2 responder**：排队（带 responder 投递进 channel，loop 起 turn 时持有并 respond），不返回
  `TurnInProgress`。

落地顺序（两步都做，但分阶段验证）：

1. **阶段一（已落地）**：§3.1 后台任务表 + `ToolContext` spawner + 独立 cancel token。后台任务
   活过发起它的 turn、结果走**被动**回流（`run_turn` 起 turn 前 drain 已完成结果、prepend 进
   prompt）。`spawn_agent` 加 `run_in_background` 开关。e2e 验证 spawn 不阻塞 + 结果下轮回流。
   落点：`crates/agent/src/session/background.rs`、`ToolContext::background`、`spawn_agent.rs`、
   `DefaultSession::run_turn` 的 drain 段。
2. **阶段二（已落地）**：拆成 2a / 2b。
   - **2a（agent crate）**：session driver（`DefaultSession::drive`，持 `Weak` 自引、`session_cancel`
     退出、`turn_freed` 活性）+ `run_turn_core`（用户 turn / 自主 turn 共用核心）+ `IngestSource::Background`。
     后台任务完成经 `BackgroundTasks` 的 `Notify` 唤醒 driver，driver 抢 turn slot 起自主续转 turn；
     撞 `TurnInProgress` 则等 `turn_freed` 重试（用户输入与后台结果竞争同一 slot 的落点）。
   - **2b（ACP crate）**：`run_prompt_turn` 简化为"跑 turn + 排队重试 `TurnInProgress` + respond"；
     新增 `spawn_session_pump`——session/new · load 时起的**持久 event pump**，订阅一次、跨所有 turn
     转发 `session/update`（含自主 turn）。原先"每 prompt 订阅"改为"每 session 订阅"，避免双发。
   - **验证**：`crates/agent/tests/e2e_turn.rs::run_in_background_result_actively_reflows`（driver 自主
     续转）+ `crates/acp/tests/background_reflow.rs`（wire 级：客户端发一次 prompt 即收到自主 turn 的
     `session/update`）。`run_in_background` 不可用上下文 fail loud；同步等待仍是默认。

转正后本文相应章节移入 `docs/internal/`（task 表 → `session.md`；input loop → 新开或并入
`turn-loop.md`）。
