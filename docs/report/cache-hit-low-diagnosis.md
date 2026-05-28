# 缓存低命中率诊断报告

本文面向当前仓库里的 agent / llm 调用链，回答一个具体问题：

> 为什么 prompt cache 命中率看起来偏低，以及应该优先怀疑哪些内容在相邻请求之间发生了变化？

## 1. 结论先行

当前实现里，拉低缓存命中率的主要因素不是 provider 层偷偷附加了时间戳、UUID、request_id 这类随机字段，而是**多轮对话历史本身会不断漂移**：

1. `messages` 每轮都回放完整历史，历史会持续增长。
2. assistant 的 `thinking` 会被回放到后续请求。
3. assistant 的 `tool_use.id` 会被写入历史；这一类 ID 天然高波动。
4. tool 的执行结果会作为 `tool_result` 完整回放；这通常是最大的动态输入源。
5. 只要进入 tool-use 多轮，后续请求的前缀就不再是“稳定指令 + 当前用户输入”，而是“稳定指令 + 不断累积的执行轨迹”。

稳定项相对少：

1. `system_prompt` 当前没有自动注入时间、session_id、request_id。
2. `session_start_append` 目前保留但未真正接入 system prompt。
3. `ToolRegistry` 的 schema 顺序有稳定化处理，不是主要抖动来源。

## 2. 代码路径定位

### 2.1 请求是怎么组出来的

`TurnRunner::build_request()` 每次发起 LLM 调用时都会重新构造：

- `system`
- `messages = history.snapshot()`
- `tools = self.tools.schemas()`

这意味着我们分析缓存时，不能只看“用户这一轮输入是否相同”，而必须看**完整历史**是否仍然有长前缀稳定。

### 2.2 历史里有哪些高波动块

assistant 回复会被写回历史，包含：

- `Text`
- `Thinking`
- `ToolUse { id, name, args }`

tool 执行结束后，结果会被写成 user 侧的：

- `ToolResult { tool_use_id, output, is_error }`

这四类内容里，`Thinking`、`ToolUse.id`、`ToolResult.output` 最容易让缓存前缀快速失稳。

## 3. 新增的诊断能力

本次补了一套轻量的“请求稳定性审计”：

- 位置：`crates/agent/src/session/turn/request_audit.rs`
- 接入点：`TurnRunner::build_request()`
- 状态保存：`DefaultSession` 级别，跨 turn 持续比较相邻请求

每次实际发给 provider 的请求，都会打一条 tracing 日志，target 为：

```text
defect::cache_audit
```

日志消息固定为：

```text
llm request cache audit
```

### 3.1 会记录哪些字段

日志会输出这些关键信号：

- `system_hash`
- `messages_hash`
- `tools_hash`
- `user_messages`
- `assistant_messages`
- `text_blocks`
- `thinking_blocks`
- `tool_use_blocks`
- `tool_result_blocks`
- `total_text_bytes`
- `total_tool_result_bytes`
- `tool_names`
- `changed`

其中 `changed` 是最重要的聚合字段，会直接告诉你相邻两次请求有哪些部分发生了变化，例如：

```text
changed=messages,assistant_message_count,thinking_blocks,tool_use_blocks,tool_result_blocks,text_bytes,tool_result_bytes
```

### 3.2 如何解读

如果你看到：

```text
changed=messages,tool_result_blocks,tool_result_bytes
```

优先怀疑：

- 工具输出正文在变化
- 工具输出体积太大
- 工具执行结果被完整回放，导致可缓存前缀被较早打断

如果你看到：

```text
changed=messages,thinking_blocks,text_bytes
```

优先怀疑：

- reasoning / thinking 回放在增长
- assistant 历史文本越来越长

如果你看到：

```text
changed=tools,tool_names
```

优先怀疑：

- per-session MCP 工具集在变化
- schema 描述或顺序不稳定

如果你看到：

```text
changed=system
```

优先怀疑：

- AGENTS.md / system overlay / provider overlay / model overlay 发生变化

## 4. 当前最值得优先优化的地方

按收益排序，建议先做下面几件事：

1. 控制 `tool_result` 回放体积。
2. 避免把无关的大段终端输出、文件全文、诊断噪音完整塞回 history。
3. 如果协议允许，尽量把稳定 instructions 和工具描述放到 prompt 最前面，把高波动内容推迟到后缀。
4. 对必须回放的 `thinking` / `tool_use` / `tool_result` 做最小化保留，而不是原样全量复读。

## 5. 推荐的排查顺序

建议按这个顺序排：

1. 先看 `changed` 是否出现 `system` / `tools`。
2. 若没有，再看是否主要是 `tool_result_bytes` 在涨。
3. 若也不是，再看 `thinking_blocks` / `text_bytes` 是否持续增长。
4. 如果 `messages_hash` 变了但细项只变 `tool_use_blocks`，说明主要是工具调用轨迹在打散前缀。
5. 最后再去怀疑 provider 侧缓存策略或上游缓存时效。

## 6. 备注

OpenAI-compatible 路径上的 `prompt_cache_key` 当前只锚定：

- model
- system
- thinking echo mode
- tool_choice
- tools

它故意不包含 turn-local `messages`。这对“给上游一个稳定缓存锚点”是有帮助的，但不能抵消完整 prompt bytes 因历史回放而产生的变化。
