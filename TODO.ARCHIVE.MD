# Defect TODO —— 已完成归档

从 `TODO.MD` 归档的已完成功能点。保留原表结构，便于回溯当时的职责域 / 优先级划分与对应设计文档。

域 / 优先级含义见 `TODO.MD` 顶部说明。

---

## 已完成（v0 主线）

| 域       | 功能点                                                         | 文档                                                                    | 优先级 |
| -------- | -------------------------------------------------------------- | ----------------------------------------------------------------------- | ------ |
| 基础     | workspace 骨架与 9 个 crate 拆分                               | `docs/architecture.md`                                                  | P0     |
| internal | Event 枚举与事件流模型                                         | `docs/internal/event-model.md`                                          | P0     |
| internal | LlmProvider trait（流式 chunk 模型）                           | `docs/internal/llm-trait.md`                                            | P0     |
| internal | Tool trait（参数 schema、流式输出、权限钩子）                  | `docs/internal/tool-trait.md`                                           | P0     |
| internal | Session 状态容器（消息历史、tool 注册表、配置）                | `docs/internal/session.md`                                              | P0     |
| internal | Turn 主循环（LLM 调用 ↔ 工具执行编排）                         | `docs/internal/turn-loop.md`                                            | P0     |
| internal | 错误类型分层（agent / provider / tool / sandbox）              | `docs/internal/errors.md`                                               | P0     |
| internal | LLM thinking 多轮 round-trip（reasoning_content 回放）         | `docs/internal/thinking-roundtrip.md`                                   | P0     |
| internal | 配置加载与合并（用户/项目/CLI 覆盖）                           | `docs/internal/config.md`                                               | P1     |
| internal | System prompt 组装（`base_prompt` / `prompt` / `AGENTS.md` / session overlay） | `docs/internal/config.md`                                               | P1     |
| internal | 入口配置接入默认 MCP servers（`mcp.enabled_servers` / `mcp.servers.*`） | `docs/internal/config.md`                                               | P1     |
| internal | 沙箱策略决策器（read-only / auto / full + 路径白名单）         | `docs/internal/sandbox-policy.md`                                       | P0     |
| internal | 内置工具：fs（read / write / edit）                            | `docs/internal/tools-fs.md`                                             | P0     |
| internal | 内置工具：bash 执行                                            | `docs/internal/tools-bash.md`                                           | P0     |
| internal | 内置工具：fetch（HTTP GET + html→markdown 渲染）               | `docs/internal/tools-fetch.md`                                          | P1     |
| internal | 内置工具：grep / glob 检索（统一 `search` tool）               | `docs/internal/tools-search.md`                                         | P1     |
| internal | 会话持久化（jsonl，append-only，可 resume）                    | `docs/internal/storage-jsonl.md`                                        | P1     |
| internal | Hook 系统（生命周期事件 + Builtin/Command/Prompt handler）     | `docs/internal/hooks.md`                                                | P1     |
| outbound | LLM 模块两层架构（协议层 + 厂商层）总览                        | `docs/outbound/llm.md`                                                  | P0     |
| outbound | Anthropic provider（messages API + 流式）                      | `docs/outbound/llm-anthropic.md`                                        | P0     |
| outbound | OpenAI 兼容 provider（chat completions + 流式）                | `docs/outbound/llm-openai.md`                                           | P0     |
| outbound | LLM provider 集测（wiremock 离线 e2e，两家）                   | `docs/outbound/llm-anthropic.md` §10 / `docs/outbound/llm-openai.md` §9 | P1     |
| outbound | HTTP 客户端基础设施（超时/重试/代理/UA）                       | `docs/outbound/http.md`                                                 | P1     |
| outbound | tracing 接入（结构化日志 + EnvFilter）                         | `docs/outbound/tracing.md`                                              | P0     |
| outbound | 可观测性独立成 `defect-obs` + Langfuse 上报（SessionObserver→AgentEvent→ingestion） | `docs/internal/observability-langfuse.md` | P1 |
| outbound | MCP client（stdio + sse 传输）                                 | `docs/outbound/mcp-client.md`                                           | P1     |
| outbound | MCP 工具适配层（包装为内部 Tool trait）                        | `docs/outbound/mcp-adapter.md`                                          | P1     |
| outbound | LLM provider 真端点 smoke（examples/，手动跑）                 | —                                                                       | 远期   |
| inbound  | ACP 桥接层总体设计（薄壳形态、方法分发、AgentEvent 投影）      | `docs/inbound/acp-bridge.md`                                            | P0     |
| inbound  | ACP 握手与能力协商（initialize / authenticate）                | `docs/inbound/acp-handshake.md`                                         | P0     |
| inbound  | ACP 会话生命周期（new / load）                                 | `docs/inbound/acp-session.md`                                           | P0     |
| inbound  | ACP prompt/turn 桥接细化（字段语义、image 处理）               | `docs/inbound/acp-prompt.md`                                            | P0     |
| inbound  | ACP Content 兼容性补全（Image / Resource / EmbeddedResource 等） | `docs/inbound/acp-prompt.md`                                            | P1     |
| inbound  | ACP 权限请求桥（requestPermission）                            | `docs/inbound/acp-permission.md`                                        | P0     |
| inbound  | ACP 文件系统委托（通过客户端读写）                             | `docs/inbound/acp-fs.md`                                                | P1     |
| inbound  | ACP shell 委托（通过客户端跑终端命令）                         | `docs/inbound/acp-shell.md`                                             | P2     |
| inbound  | ACP 取消/中断（cancel turn 边界 case）                         | `docs/inbound/acp-cancel.md`                                            | P1     |
| inbound  | ACP 暴露 session 模型候选并支持 `session/set_model`            | `docs/inbound/acp-session.md`                                           | P1     |
| internal | 上下文压缩编排（结构化摘要 + tail 保留 + ratio 触发）          | `docs/internal/turn-loop.md` §4                                         | P1     |
| internal | Skill 加载机制（dir-per-skill 发现 + `skill` 工具按需加载 L2 body） | `docs/internal/skills.md`                                          | 远期   |
| internal | Subagent 系统（`spawn_agent` 工具 + profile 文件夹）           | `docs/internal/skills.md` 顶部实现注 / `crates/config/src/profiles.rs`  | 远期   |
| 基础     | CLI 入口（参数解析、组装、跑 ACP server）                      | `docs/cli.md`                                                           | P0     |
| 基础     | 端到端冒烟用例（CLI ←→ 假 ACP client ←→ 假 LLM）               | `docs/testing/e2e.md`                                                   | P1     |
| 基础     | CLI MCP 冒烟用例（真实 defect 二进制 + 假 OpenAI + stdio MCP） | `docs/testing/e2e.md`                                                   | P1     |
