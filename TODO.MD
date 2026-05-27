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

---

## 任务表

| 域       | 功能点                                                         | 文档                                                                    | 优先级 | 完成情况 |
| -------- | -------------------------------------------------------------- | ----------------------------------------------------------------------- | ------ | -------- |
| 基础     | workspace 骨架与 9 个 crate 拆分                               | `docs/architecture.md`                                                  | P0     | ✅ 已完成 |
| internal | Event 枚举与事件流模型                                         | `docs/internal/event-model.md`                                          | P0     | ✅ 已完成 |
| internal | LlmProvider trait（流式 chunk 模型）                           | `docs/internal/llm-trait.md`                                            | P0     | ✅ 已完成 |
| internal | Tool trait（参数 schema、流式输出、权限钩子）                  | `docs/internal/tool-trait.md`                                           | P0     | ✅ 已完成 |
| internal | Session 状态容器（消息历史、tool 注册表、配置）                | `docs/internal/session.md`                                              | P0     | ✅ 已完成 |
| internal | Turn 主循环（LLM 调用 ↔ 工具执行编排）                         | `docs/internal/turn-loop.md`                                            | P0     | ✅ 已完成 |
| internal | 错误类型分层（agent / provider / tool / sandbox）              | `docs/internal/errors.md`                                               | P0     | ✅ 已完成 |
| internal | LLM thinking 多轮 round-trip（reasoning_content 回放）         | `docs/internal/thinking-roundtrip.md`                                   | P0     | ✅ 已完成 |
| internal | 配置加载与合并（用户/项目/CLI 覆盖）                           | `docs/internal/config.md`                                               | P1     | 🚧 进行中 |
| internal | System prompt 组装（`base_prompt` / `prompt` / `AGENTS.md` / session overlay） | `docs/internal/config.md`                                               | P1     | ✅ 已完成 |
| internal | 入口配置接入默认 MCP servers（`mcp.enabled_servers` / `mcp.servers.*`） | `docs/internal/config.md`                                               | P1     | ✅ 已完成 |
| internal | 沙箱策略决策器（read-only / auto / full + 路径白名单）         | `docs/internal/sandbox-policy.md`                                       | P0     | ✅ 已完成 |
| internal | 内置工具：fs（read / write / edit）                            | `docs/internal/tools-fs.md`                                             | P0     | ✅ 已完成 |
| internal | 内置工具：bash 执行                                            | `docs/internal/tools-bash.md`                                           | P0     | ✅ 已完成 |
| internal | 内置工具：grep / glob 检索                                     | `docs/internal/tools-search.md`                                         | P1     | 🕒 未开始 |
| internal | 会话持久化（jsonl，append-only，可 resume）                    | `docs/internal/storage-jsonl.md`                                        | P1     | ✅ 已完成 |
| outbound | LLM 模块两层架构（协议层 + 厂商层）总览                        | `docs/outbound/llm.md`                                                  | P0     | ✅ 已完成 |
| outbound | Anthropic provider（messages API + 流式）                      | `docs/outbound/llm-anthropic.md`                                        | P0     | ✅ 已完成 |
| outbound | OpenAI 兼容 provider（chat completions + 流式）                | `docs/outbound/llm-openai.md`                                           | P0     | ✅ 已完成 |
| outbound | LLM provider 集测（wiremock 离线 e2e，两家）                   | `docs/outbound/llm-anthropic.md` §10 / `docs/outbound/llm-openai.md` §9 | P1     | ✅ 已完成 |
| outbound | HTTP 客户端基础设施（超时/重试/代理/UA）                       | `docs/outbound/http.md`                                                 | P1     | ✅ 已完成 |
| outbound | tracing 接入（结构化日志 + EnvFilter）                         | `docs/outbound/tracing.md`                                              | P0     | ✅ 已完成 |
| outbound | MCP client（stdio + sse 传输）                                 | `docs/outbound/mcp-client.md`                                           | P1     | ✅ 已完成 |
| outbound | MCP 工具适配层（包装为内部 Tool trait）                        | `docs/outbound/mcp-adapter.md`                                          | P1     | ✅ 已完成 |
| inbound  | ACP 桥接层总体设计（薄壳形态、方法分发、AgentEvent 投影）      | `docs/inbound/acp-bridge.md`                                            | P0     | ✅ 已完成 |
| inbound  | ACP 握手与能力协商（initialize / authenticate）                | `docs/inbound/acp-handshake.md`                                         | P0     | ✅ 已完成 |
| inbound  | ACP 会话生命周期（new / load）                                 | `docs/inbound/acp-session.md`                                           | P0     | ✅ 已完成 |
| inbound  | ACP prompt/turn 桥接细化（字段语义、image 处理）               | `docs/inbound/acp-prompt.md`                                            | P0     | ✅ 已完成 |
| inbound  | ACP Content 兼容性补全（Image / Resource / EmbeddedResource 等） | `docs/inbound/acp-prompt.md`                                            | P1     | ✅ 已完成 |
| inbound  | ACP 权限请求桥（requestPermission）                            | `docs/inbound/acp-permission.md`                                        | P0     | ✅ 已完成 |
| inbound  | ACP 文件系统委托（通过客户端读写）                             | `docs/inbound/acp-fs.md`                                                | P1     | ✅ 已完成 |
| inbound  | ACP shell 委托（通过客户端跑终端命令）                         | `docs/inbound/acp-shell.md`                                             | P2     | ✅ 已完成 |
| inbound  | ACP 取消/中断（cancel turn 边界 case）                         | `docs/inbound/acp-cancel.md`                                            | P1     | ✅ 已完成 |
| inbound  | ACP 暴露 session 模型候选并支持 `session/set_model`            | `docs/inbound/acp-session.md`                                           | P1     | 🚧 进行中 |
| 基础     | CLI 入口（参数解析、组装、跑 ACP server）                      | `docs/cli.md`                                                           | P0     | ✅ 已完成 |
| 基础     | 端到端冒烟用例（CLI ←→ 假 ACP client ←→ 假 LLM）               | `docs/testing/e2e.md`                                                   | P1     | ✅ 已完成 |
| 基础     | CLI MCP 冒烟用例（真实 defect 二进制 + 假 OpenAI + stdio MCP） | `docs/testing/e2e.md`                                                   | P1     | ✅ 已完成 |

---

## 远期目标

| 域       | 功能点                                                         | 文档                          | 完成情况 |
| -------- | -------------------------------------------------------------- | ----------------------------- | -------- |
| outbound | LLM provider 真端点 smoke（examples/，手动跑）                 | —                             | ✅ 已完成 |
| outbound | tracing OTLP exporter（导出到外部 collector）                  | `docs/outbound/tracing-otlp.md` | 🕒 未开始 |
| inbound  | ACP Content 音频输入支持                                       | `docs/inbound/acp-prompt.md`  | 🕒 未开始 |
| inbound  | ACP Content 二进制资源支持（BlobResourceContents）              | `docs/inbound/acp-prompt.md`  | 🕒 未开始 |
| internal | Hook 系统（session / turn / tool 生命周期钩子，shell 命令执行，见 `docs/proposals/search-capability-and-fetch-tool.md` 讨论） | `docs/internal/hooks.md` | 🕒 未开始 |
| internal | Skill 加载机制（glob 条件匹配 + always 的 markdown skill 注入到 system prompt，附 `skill` 工具供模型按需加载） | `docs/internal/skills.md` | 🕒 未开始 |
| internal | LSP 语义理解（诊断注入、跳转定义、查找引用；P1 可用 hook 跑 linter 过渡） | `docs/internal/lsp.md` | 🕒 未开始 |
| internal | Checkpoint / Undo（工具执行前自动快照，支持回滚；可复用 git stash 或文件级快照） | `docs/internal/checkpoint.md` | 🕒 未开始 |

---

## 维护规则

- 每个功能点动手前先把对应 `docs/...md` 写出来（即便只是几行设计意图）
- 完成情况字段在 PR 合入时同步更新；进入 `进行中` 必须有指派人或 PR 链接备注
- 新增功能点直接在表中追加，不另起小节；优先级与完成情况是表里的字段，不另开列表
