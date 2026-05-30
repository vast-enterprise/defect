# Defect TODO

按职责域划分：

- **outbound** —— defect 主动调用外部服务（LLM、MCP server、observability sink）
- **inbound** —— 外部调用 defect（ACP 客户端→agent）
- **internal** —— agent 自身的核心逻辑与机制

优先级：

- **P0** ：v0 跑通必需，缺它整条链路无法闭环
- **P1** ：v0 完整体验需要，缺它能跑但不好用
- **P2** ：演进/优化，可放到后续版本

完成情况：`🕒 未开始` / `🚧 进行中` / `✅ 已完成` / `⛔ 阻塞`

> v0 主线（任务表里曾经 `✅` 的全部条目）已迁到 `TODO.ARCHIVE.MD`。本文件只留**未完成 / 已写设计但未落地**的工作。

---

## 进行中 / 待落地（有文档，未完成）

| 域       | 功能点                                                         | 文档                                                                    | 优先级 | 完成情况 |
| -------- | -------------------------------------------------------------- | ----------------------------------------------------------------------- | ------ | -------- |
| internal | Skill 条件激活：`always: true` body 注入 + `triggers.globs/keywords` 匹配（字段 v0 已显式占位，"解析但不消费"，待 v1 接入 SkillRouterHook / SkillManifestHook 消费） | `docs/internal/skills.md` §4.3 / §5.1 / §9 | P2 | 🚧 进行中（占位已落地） |
| inbound  | ACP slash commands（`session/commands` 暴露 + 触发桥接）       | `docs/inbound/acp-slash-commands.md`                                    | P1     | 🕒 未开始 |
| internal | 上下文 microcompact（Phase 2：原地把旧 tool_result 替换成占位、不调 LLM；现有结构化摘要压缩已落地） | `docs/internal/turn-loop.md` §4 | P2 | 🕒 未开始 |
| internal | Workspace Checkpoint / Undo（git 隐藏引用做 turn 级文件快照，可回滚、不污染用户分支） | `docs/internal/workspace-checkpoints.md` | P2 | 🕒 未开始（设计已写） |
| internal | LSP 语义理解（诊断注入、跳转定义、查找引用；P1 可用 hook 跑 linter 过渡） | `docs/report/lsp-integration-survey.md`（调研报告，待转设计） | P2 | 🕒 未开始 |
| internal | 用 `schemars` 从 args struct 生成 Tool 参数 schema（替代手写 `json!`） | `docs/proposals/tool-schema-via-schemars.md` | P2 | 🕒 未开始（提案） |
| internal | 任务编排（task arrange）                                       | `docs/proposals/task-arrange.md`（仅占位标题，内容待写）                | P2     | 🕒 未开始 |
| internal | Hook async/observe 事件接入（8 个 observe 事件三层全缺，详见下方说明） | `docs/internal/hooks.md` §7.2 / §11 | P2 | 🕒 未开始（仅 enum 占位） |

---

## 远期目标

| 域       | 功能点                                                         | 文档                          | 完成情况 |
| -------- | -------------------------------------------------------------- | ----------------------------- | -------- |
| outbound | tracing OTLP exporter（导出到外部 collector；走 tracing-opentelemetry Layer，权衡见 langfuse 文档 §8） | `docs/internal/observability-langfuse.md` §8 | 🕒 未开始 |
| outbound | ObsContext 透传：工具内 embedding/后端调用挂回 langfuse trace | `docs/internal/observability-langfuse.md` §5 | 🕒 未开始 |
| outbound | Provider/protocol polyfills（azure-openai、openai-responses、gemini-generate-content、bedrock-converse、ollama-native、vertex-anthropic、OpenAI 兼容预设集） | `docs/proposals/provider-protocol-polyfills.md` | 🕒 未开始（backlog，等真实场景） |
| outbound | Hosted web_search（各 provider wire codec；现 capabilities 一律返回 false） | `docs/internal/capabilities.md` §143 | 🕒 未开始 |
| outbound | HTTP 层增强：分阶段超时（connect/header/body）、retry & timeout & proxy 的 wiremock e2e、SSE 流中途 resume-from-last-event-id | `docs/outbound/http.md` §6 / §9 | 🕒 未开始 |
| internal | fs 工具 v1：大文件 window read（offset/limit 流式）、write 描述阶段精确 diff、edit 并发写冲突检测、模糊匹配、二进制/多模态、目录增删改 | `docs/internal/tools-fs.md` §3.1/§4.3/§5.4/§9 | 🕒 未开始 |
| internal | bash 工具 v1：进程组清理（setpgid + kill tree）、大输出 spill-to-disk、argv 模式、stdin 支持 | `docs/internal/tools-bash.md` §8 | 🕒 未开始 |
| internal | 沙箱演进：OS 级隔离后端（landlock/seatbelt/seccomp，pluggable）、per-session policy override | `docs/internal/sandbox-policy.md` | 🕒 未开始 |
| internal | Storage 演进：jsonl → 带索引的后端（sqlite 等）；尾部残缺行自动截断 | `crates/storage/src/lib.rs` | 🕒 未开始 |
| inbound  | ACP Content 音频输入支持                                       | `docs/inbound/acp-prompt.md`  | 🕒 未开始 |
| inbound  | ACP Content 二进制资源支持（BlobResourceContents）              | `docs/inbound/acp-prompt.md`  | 🕒 未开始 |

---

## 备注：Hook async/observe 事件的真实缺口

文档曾写「Async 观察事件 enum 打桩、运行时不触发」，实测缺口比这句话更宽——
8 个 observe 事件（`SessionEnd` / `TurnStart` / `TurnEnd` / `PreLlmCall` /
`PostLlmCall` / `PreCompact` / `PostCompact` / `PermissionAsk`）**三层全缺**：

1. **emit**：全代码库零 `HookEvent::TurnStart` 等构造点（主循环只 emit 5 件 Sync 套）。
2. **引擎**：`DefaultHookEngine::observe()`（`crates/agent/src/hooks.rs`）是空函数，
   连 AgentEvent fan-out 框架都没接；`HookCapability::Observe` 目前无代码路径用到。
3. **config**：`HooksSection` / `HookEventTag`（`crates/config/src/hooks.rs`）只有
   5 个 Sync 字段/变体，根本没有 async 事件的桶。

落地需要：① config 加 async 事件桶 ② 引擎 `observe()` 接 [event-model](docs/internal/event-model.md)
的 AgentEvent 流 fan-out ③ 主循环（或 AgentEvent 投影层）emit。

**附带必修**：`HooksSection` 没有 `deny_unknown_fields`——用户现在写
`[[hooks.turn_start]]` 之类**超前/拼错的事件键会被 serde 静默丢弃**，hook 永不触发
却不报错。这违反「要约束就 hard fail 别静默忽略」原则，应补 `deny_unknown_fields`
（与 config 其余层、skill manifest 的做法对齐）。

---

## 维护规则

- 每个功能点动手前先把对应 `docs/...md` 写出来（即便只是几行设计意图）
- 完成情况字段在 PR 合入时同步更新；进入 `进行中` 必须有指派人或 PR 链接备注
- 功能点**做完后**从本文件移到 `TODO.ARCHIVE.MD`，不在本文件留 `✅` 行
- 新增功能点直接在对应表中追加，不另起小节；优先级与完成情况是表里的字段，不另开列表
