//! Langfuse model / projector 的单元测试。
//!
//! model 测试锁定 wire 契约（字段名 camelCase、type 判别值、None 跳过）；
//! projector 测试锁定事件翻译（trace/generation/span 结构、usage 映射、id 配对）。

use agent_client_protocol_schema::{
    ContentBlock, StopReason, TextContent, ToolCallId, ToolCallStatus, ToolCallUpdateFields,
};
use defect_agent::event::{AgentEvent, LlmRequestSnapshot};
use defect_agent::llm::{Message, MessageContent, Role, Usage};
use serde_json::json;

use super::model::{
    EventKind, IngestionEvent, IngestionResponse, ObservationBody, ObservationLevel, TraceBody,
};
use super::projector::TraceProjector;

/// 确定性 id 生成器：`env-1`、`env-2`…，便于断言。
fn counter_ids() -> impl FnMut() -> String {
    let mut n = 0u32;
    move || {
        n += 1;
        format!("id-{n}")
    }
}

const NOW: &str = "2026-05-29T00:00:00Z";

fn text_block(s: &str) -> ContentBlock {
    ContentBlock::Text(TextContent::new(s.to_string()))
}

/// 一个最小请求快照：system + 一条 user 文本消息。
fn snapshot(system: Option<&str>, user: &str) -> std::sync::Arc<LlmRequestSnapshot> {
    std::sync::Arc::new(LlmRequestSnapshot {
        system: system.map(std::sync::Arc::from),
        messages: vec![Message {
            role: Role::User,
            content: std::sync::Arc::from([MessageContent::Text {
                text: user.to_string(),
            }]),
        }],
    })
}

#[test]
fn trace_create_envelope_shape() {
    let body = TraceBody {
        id: "trace-1".into(),
        name: Some("turn".into()),
        session_id: Some("sess-1".into()),
        input: Some(json!("hello")),
        timestamp: Some("2026-05-29T00:00:00Z".into()),
        ..Default::default()
    };
    let ev = IngestionEvent::trace(
        "env-1".into(),
        "2026-05-29T00:00:00Z".into(),
        EventKind::TraceCreate,
        &body,
    );
    let v = serde_json::to_value(&ev).unwrap();
    assert_eq!(v["id"], "env-1");
    assert_eq!(v["type"], "trace-create");
    assert_eq!(v["timestamp"], "2026-05-29T00:00:00Z");
    // body 字段 camelCase，sessionId 而非 session_id。
    assert_eq!(v["body"]["id"], "trace-1");
    assert_eq!(v["body"]["sessionId"], "sess-1");
    assert_eq!(v["body"]["name"], "turn");
    assert_eq!(v["body"]["input"], "hello");
    // 未设的字段不出现在 JSON 里。
    assert!(v["body"].get("output").is_none());
    assert!(v["body"].get("metadata").is_none());
}

#[test]
fn generation_body_usage_details_camel_case() {
    let mut usage = serde_json::Map::new();
    usage.insert("input".into(), json!(100));
    usage.insert("output".into(), json!(20));
    usage.insert("cache_read_input_tokens".into(), json!(8));

    let body = ObservationBody {
        id: "gen-1".into(),
        trace_id: "trace-1".into(),
        parent_observation_id: None,
        name: Some("llm_call".into()),
        model: Some("claude-opus-4-8".into()),
        usage_details: Some(usage),
        level: Some(ObservationLevel::Error),
        status_message: Some("boom".into()),
        ..Default::default()
    };
    let ev = IngestionEvent::observation(
        "env-2".into(),
        "2026-05-29T00:00:01Z".into(),
        EventKind::GenerationUpdate,
        &body,
    );
    let v = serde_json::to_value(&ev).unwrap();
    assert_eq!(v["type"], "generation-update");
    assert_eq!(v["body"]["traceId"], "trace-1");
    assert_eq!(v["body"]["model"], "claude-opus-4-8");
    assert_eq!(v["body"]["usageDetails"]["input"], 100);
    assert_eq!(v["body"]["usageDetails"]["cache_read_input_tokens"], 8);
    assert_eq!(v["body"]["level"], "ERROR");
    assert_eq!(v["body"]["statusMessage"], "boom");
    // parentObservationId 为 None 时不出现。
    assert!(v["body"].get("parentObservationId").is_none());
}

#[test]
fn span_with_parent_observation() {
    let body = ObservationBody {
        id: "span-1".into(),
        trace_id: "trace-1".into(),
        parent_observation_id: Some("gen-1".into()),
        name: Some("bash".into()),
        ..Default::default()
    };
    let ev = IngestionEvent::observation(
        "env-3".into(),
        "2026-05-29T00:00:02Z".into(),
        EventKind::SpanCreate,
        &body,
    );
    let v = serde_json::to_value(&ev).unwrap();
    assert_eq!(v["type"], "span-create");
    assert_eq!(v["body"]["parentObservationId"], "gen-1");
    // span 不带 model / usageDetails。
    assert!(v["body"].get("model").is_none());
    assert!(v["body"].get("usageDetails").is_none());
}

// ---- ingestion response 解析（207 误报修复回归） ----

#[test]
fn parses_207_all_success_as_no_errors() {
    // 这是真实 207 响应体（全部成功）——errors 为空，不应被当作错误。
    let body =
        r#"{"successes":[{"id":"be5dbe21-a204-407b-bf52-6ec031164650","status":201}],"errors":[]}"#;
    let parsed: IngestionResponse = serde_json::from_str(body).unwrap();
    assert_eq!(parsed.successes.len(), 1);
    assert_eq!(parsed.successes[0].status, 201);
    assert!(parsed.errors.is_empty(), "全成功的 207 不应有 errors");
}

#[test]
fn parses_207_with_partial_errors() {
    let body = r#"{"successes":[{"id":"a","status":201}],"errors":[{"id":"b","status":400,"message":"bad body"}]}"#;
    let parsed: IngestionResponse = serde_json::from_str(body).unwrap();
    assert_eq!(parsed.successes.len(), 1);
    assert_eq!(parsed.errors.len(), 1);
    assert_eq!(parsed.errors[0].status, 400);
    assert_eq!(parsed.errors[0].message.as_deref(), Some("bad body"));
}

// ---- projector ----

/// 把 projector 产出的 ingestion 事件序列化成 `Vec<Value>`，便于断言。
fn project_json(
    proj: &mut TraceProjector,
    event: AgentEvent,
    ids: &mut impl FnMut() -> String,
) -> Vec<serde_json::Value> {
    proj.project(event, NOW, ids)
        .iter()
        .map(|e| serde_json::to_value(e).unwrap())
        .collect()
}

#[test]
fn turn_started_emits_trace_create_with_session() {
    let mut proj = TraceProjector::new("sess-abc");
    let mut ids = counter_ids();
    let out = project_json(&mut proj, AgentEvent::TurnStarted, &mut ids);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0]["type"], "trace-create");
    // trace_id 是第一个分配的 id；信封 id 是第二个。
    assert_eq!(out[0]["body"]["id"], "id-1");
    assert_eq!(out[0]["id"], "id-2");
    assert_eq!(out[0]["body"]["sessionId"], "sess-abc");
    assert_eq!(out[0]["body"]["name"], "turn");
}

#[test]
fn llm_call_lifecycle_creates_step_then_generation() {
    let mut proj = TraceProjector::new("sess-1");
    let mut ids = counter_ids();
    project_json(&mut proj, AgentEvent::TurnStarted, &mut ids); // trace_id = id-1

    // LlmCallStarted → 先建 step span（容器），再建挂在 step 下的 generation。
    let created = project_json(
        &mut proj,
        AgentEvent::LlmCallStarted {
            model: "claude-opus-4-8".into(),
            attempt: 1,
            request: snapshot(Some("you are helpful"), "hi there"),
        },
        &mut ids,
    );
    // [0] = step span-create；[1] = generation-create（父 = step）。
    assert_eq!(created[0]["type"], "span-create");
    assert_eq!(created[0]["body"]["name"], "step");
    let step_id = created[0]["body"]["id"].as_str().unwrap().to_string();
    assert_eq!(step_id, "id-1-step-1");
    // 顶层 step 直接挂 trace（无 parentObservationId）。
    assert!(created[0]["body"].get("parentObservationId").is_none());

    assert_eq!(created[1]["type"], "generation-create");
    let gen_id = created[1]["body"]["id"].as_str().unwrap().to_string();
    assert_eq!(gen_id, "id-1-step-1-gen");
    assert_eq!(created[1]["body"]["parentObservationId"], step_id);
    assert_eq!(created[1]["body"]["traceId"], "id-1");
    assert_eq!(created[1]["body"]["model"], "claude-opus-4-8");
    assert_eq!(created[1]["body"]["metadata"]["attempt"], 1);
    // input 是 chat messages 数组：system 第一条，user 第二条。
    assert_eq!(created[1]["body"]["input"][0]["role"], "system");
    assert_eq!(created[1]["body"]["input"][0]["content"], "you are helpful");
    assert_eq!(created[1]["body"]["input"][1]["role"], "user");
    assert_eq!(created[1]["body"]["input"][1]["content"], "hi there");

    // output/thinking 流式到达。
    project_json(
        &mut proj,
        AgentEvent::AssistantThought {
            content: text_block("let me think"),
        },
        &mut ids,
    );
    project_json(
        &mut proj,
        AgentEvent::AssistantText {
            content: text_block("hello "),
        },
        &mut ids,
    );
    project_json(
        &mut proj,
        AgentEvent::AssistantText {
            content: text_block("world"),
        },
        &mut ids,
    );

    // LlmCallFinished → generation **当即收尾**（gen 时长 = 纯 LLM，不含工具）。
    let finished = project_json(
        &mut proj,
        AgentEvent::LlmCallFinished {
            model: "claude-opus-4-8".into(),
            attempt: 1,
            usage: Usage {
                input_tokens: Some(80),
                output_tokens: Some(12),
                ..Default::default()
            },
            error: None,
        },
        &mut ids,
    );
    assert_eq!(finished[0]["type"], "generation-update");
    assert_eq!(finished[0]["body"]["id"], gen_id);
    assert_eq!(finished[0]["body"]["name"], "llm_call");
    assert_eq!(finished[0]["body"]["output"], "hello world");
    // thinking 放 metadata.reasoning，不进 output。
    assert_eq!(finished[0]["body"]["metadata"]["reasoning"], "let me think");
    assert!(finished[0]["body"].get("level").is_none());
    // generation 的 usageDetails = 本次调用 usage（80/12）。
    assert_eq!(finished[0]["body"]["usageDetails"]["input"], 80);
    assert_eq!(finished[0]["body"]["usageDetails"]["output"], 12);

    // TurnEnded → 关闭 step span（[0]），再 trace 收尾（[1]，usage = turn 累计）。
    let ended = project_json(
        &mut proj,
        AgentEvent::TurnEnded {
            reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: Some(100),
                output_tokens: Some(20),
                ..Default::default()
            },
        },
        &mut ids,
    );
    assert_eq!(ended[0]["type"], "span-update");
    assert_eq!(ended[0]["body"]["id"], step_id);
    assert!(ended[0]["body"]["endTime"].is_string());
    assert_eq!(ended[1]["type"], "trace-create");
    assert_eq!(ended[1]["body"]["name"], "turn");
    assert_eq!(ended[1]["body"]["output"], "hello world");
    assert_eq!(ended[1]["body"]["metadata"]["usage"]["input"], 100);
}

#[test]
fn two_llm_calls_get_distinct_per_call_usage() {
    // 多轮（工具循环）场景：一个 turn 内两次 LLM 调用，各自 usage 不同——
    // 这是本次修复的核心（之前 generation 全无 usage / 只有 turn 总和）。
    let mut proj = TraceProjector::new("s");
    let mut ids = counter_ids();
    project_json(&mut proj, AgentEvent::TurnStarted, &mut ids);

    // 第 1 次调用：usage 50/10。LlmCallFinished 当即收尾 gen1。
    project_json(
        &mut proj,
        AgentEvent::LlmCallStarted {
            model: "m".into(),
            attempt: 1,
            request: snapshot(None, "first"),
        },
        &mut ids,
    );
    let gen1_flush = project_json(
        &mut proj,
        AgentEvent::LlmCallFinished {
            model: "m".into(),
            attempt: 1,
            usage: Usage {
                input_tokens: Some(50),
                output_tokens: Some(10),
                ..Default::default()
            },
            error: None,
        },
        &mut ids,
    );
    // gen1 在自己的 LlmCallFinished 收尾，带它自己的 50/10。
    assert_eq!(gen1_flush[0]["type"], "generation-update");
    assert_eq!(gen1_flush[0]["body"]["usageDetails"]["input"], 50);
    assert_eq!(gen1_flush[0]["body"]["usageDetails"]["output"], 10);

    // 第 2 次调用开始 → 关闭 step-1，开 step-2，建 gen2。
    let step2 = project_json(
        &mut proj,
        AgentEvent::LlmCallStarted {
            model: "m".into(),
            attempt: 1,
            request: snapshot(None, "second"),
        },
        &mut ids,
    );
    // [0] = step-1 span-update（关闭）；[1] = step-2 span-create；[2] = gen2 create。
    assert_eq!(step2[0]["type"], "span-update");
    assert_eq!(step2[0]["body"]["id"], "id-1-step-1");
    assert_eq!(step2[1]["type"], "span-create");
    assert_eq!(step2[1]["body"]["id"], "id-1-step-2");
    assert_eq!(step2[2]["type"], "generation-create");
    assert_eq!(step2[2]["body"]["id"], "id-1-step-2-gen");

    // 第 2 次调用 usage 200/40（明显不同于第 1 次）→ 当即收尾 gen2。
    let gen2_flush = project_json(
        &mut proj,
        AgentEvent::LlmCallFinished {
            model: "m".into(),
            attempt: 1,
            usage: Usage {
                input_tokens: Some(200),
                output_tokens: Some(40),
                ..Default::default()
            },
            error: None,
        },
        &mut ids,
    );
    // gen2 收尾带它自己的 200/40，不是 turn 累计 250/50。
    assert_eq!(gen2_flush[0]["type"], "generation-update");
    assert_eq!(gen2_flush[0]["body"]["usageDetails"]["input"], 200);
    assert_eq!(gen2_flush[0]["body"]["usageDetails"]["output"], 40);

    let ended = project_json(
        &mut proj,
        AgentEvent::TurnEnded {
            reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: Some(250),
                output_tokens: Some(50),
                ..Default::default()
            },
        },
        &mut ids,
    );
    // TurnEnded 关闭 step-2（[0]），trace 收尾带 turn 总和 250/50（[1]）。
    assert_eq!(ended[0]["type"], "span-update");
    assert_eq!(ended[0]["body"]["id"], "id-1-step-2");
    assert_eq!(ended[1]["body"]["metadata"]["usage"]["input"], 250);
}

#[test]
fn llm_error_sets_error_level_and_status() {
    let mut proj = TraceProjector::new("s");
    let mut ids = counter_ids();
    project_json(&mut proj, AgentEvent::TurnStarted, &mut ids);
    project_json(
        &mut proj,
        AgentEvent::LlmCallStarted {
            model: "m".into(),
            attempt: 2,
            request: snapshot(None, "go"),
        },
        &mut ids,
    );
    // error 记在 generation 上，LlmCallFinished 当即写出 level/statusMessage。
    let finished = project_json(
        &mut proj,
        AgentEvent::LlmCallFinished {
            model: "m".into(),
            attempt: 2,
            usage: Usage::default(),
            error: Some("rate limited".into()),
        },
        &mut ids,
    );
    assert_eq!(finished[0]["type"], "generation-update");
    assert_eq!(finished[0]["body"]["level"], "ERROR");
    assert_eq!(finished[0]["body"]["statusMessage"], "rate limited");
}

#[test]
fn tool_call_creates_and_updates_span_with_pairing() {
    let mut proj = TraceProjector::new("s");
    let mut ids = counter_ids();
    project_json(&mut proj, AgentEvent::TurnStarted, &mut ids); // trace = id-1
    // 工具恒在某次 llm_call 之后——先起一次 LLM 调用建出 step（容器）。
    project_json(
        &mut proj,
        AgentEvent::LlmCallStarted {
            model: "m".into(),
            attempt: 1,
            request: snapshot(None, "go"),
        },
        &mut ids,
    );

    let mut started_fields = ToolCallUpdateFields::default();
    started_fields.raw_input = Some(json!({ "cmd": "ls" }));
    let started = project_json(
        &mut proj,
        AgentEvent::ToolCallStarted {
            id: ToolCallId::new("call-7"),
            name: "bash".into(),
            fields: started_fields,
        },
        &mut ids,
    );
    assert_eq!(started[0]["type"], "span-create");
    let span_id = started[0]["body"]["id"].as_str().unwrap().to_string();
    assert_eq!(span_id, "id-1-tool-call-7");
    assert_eq!(started[0]["body"]["name"], "bash");
    assert_eq!(started[0]["body"]["input"]["cmd"], "ls");
    // 工具挂在当前 step 下（与 llm_call 互为兄弟）。
    assert_eq!(started[0]["body"]["parentObservationId"], "id-1-step-1");

    let mut done_fields = ToolCallUpdateFields::default();
    done_fields.status = Some(ToolCallStatus::Completed);
    done_fields.raw_output = Some(json!({ "stdout": "a\nb" }));
    let finished = project_json(
        &mut proj,
        AgentEvent::ToolCallFinished {
            id: ToolCallId::new("call-7"),
            fields: done_fields,
        },
        &mut ids,
    );
    assert_eq!(finished[0]["type"], "span-update");
    // 配对到同一 span id。
    assert_eq!(finished[0]["body"]["id"], span_id);
    assert_eq!(finished[0]["body"]["output"]["stdout"], "a\nb");
    assert!(finished[0]["body"].get("level").is_none());
}

#[test]
fn failed_tool_sets_error_level() {
    let mut proj = TraceProjector::new("s");
    let mut ids = counter_ids();
    project_json(&mut proj, AgentEvent::TurnStarted, &mut ids);
    project_json(
        &mut proj,
        AgentEvent::ToolCallStarted {
            id: ToolCallId::new("c1"),
            name: "bash".into(),
            fields: ToolCallUpdateFields::default(),
        },
        &mut ids,
    );
    let mut f = ToolCallUpdateFields::default();
    f.status = Some(ToolCallStatus::Failed);
    let finished = project_json(
        &mut proj,
        AgentEvent::ToolCallFinished {
            id: ToolCallId::new("c1"),
            fields: f,
        },
        &mut ids,
    );
    assert_eq!(finished[0]["body"]["level"], "ERROR");
}

#[test]
fn turn_ended_updates_trace_with_same_id() {
    let mut proj = TraceProjector::new("sess-x");
    let mut ids = counter_ids();

    // 真实顺序：主循环先发 UserPromptCommitted，再发 TurnStarted。
    let pre = project_json(
        &mut proj,
        AgentEvent::UserPromptCommitted {
            content: vec![text_block("do something")],
        },
        &mut ids,
    );
    // UserPromptCommitted 本身不产出 ingestion 事件（只暂存 input）。
    assert!(pre.is_empty());

    let started = project_json(&mut proj, AgentEvent::TurnStarted, &mut ids);
    let trace_id = started[0]["body"]["id"].as_str().unwrap().to_string();
    // input 在 trace-create 时就带上（不必等 TurnEnded）——这是本次回归点。
    assert_eq!(started[0]["body"]["input"], "do something");

    project_json(
        &mut proj,
        AgentEvent::LlmCallStarted {
            model: "m".into(),
            attempt: 1,
            request: snapshot(None, "go"),
        },
        &mut ids,
    );
    project_json(
        &mut proj,
        AgentEvent::AssistantText {
            content: text_block("done"),
        },
        &mut ids,
    );

    let ended = project_json(
        &mut proj,
        AgentEvent::TurnEnded {
            reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: Some(100),
                output_tokens: Some(20),
                ..Default::default()
            },
        },
        &mut ids,
    );
    // 本轮没单独发 LlmCallFinished：TurnEnded 兜底——先 flush generation-update（[0]），
    // 再关闭 step span（[1]），最后发 trace 更新（[2]）。
    assert_eq!(ended[0]["type"], "generation-update");
    assert_eq!(ended[1]["type"], "span-update");
    assert!(ended[1]["body"]["name"].is_null() || ended[1]["body"]["name"] == "step");
    assert_eq!(ended[2]["type"], "trace-create");
    // trace 用同一 trace_id 更新（合并 input/output/endTime）。
    assert_eq!(ended[2]["body"]["id"], trace_id);
    assert_eq!(ended[2]["body"]["name"], "turn");
    assert_eq!(ended[2]["body"]["sessionId"], "sess-x");
    assert_eq!(ended[2]["body"]["input"], "do something");
    assert_eq!(ended[2]["body"]["output"], "done");
    assert_eq!(ended[2]["body"]["metadata"]["stop_reason"], "end_turn");
    assert_eq!(ended[2]["body"]["metadata"]["usage"]["input"], 100);
}

#[test]
fn two_turns_get_distinct_trace_ids() {
    let mut proj = TraceProjector::new("s");
    let mut ids = counter_ids();
    let t1 = project_json(&mut proj, AgentEvent::TurnStarted, &mut ids);
    project_json(
        &mut proj,
        AgentEvent::TurnEnded {
            reason: StopReason::EndTurn,
            usage: Usage::default(),
        },
        &mut ids,
    );
    let t2 = project_json(&mut proj, AgentEvent::TurnStarted, &mut ids);
    assert_ne!(t1[0]["body"]["id"], t2[0]["body"]["id"]);
}

#[test]
fn events_before_turn_started_are_ignored() {
    let mut proj = TraceProjector::new("s");
    let mut ids = counter_ids();
    // 没有 TurnStarted 就来 LLM/工具事件：projector 不应 panic，返回空。
    let out = project_json(
        &mut proj,
        AgentEvent::LlmCallStarted {
            model: "m".into(),
            attempt: 1,
            request: snapshot(None, "go"),
        },
        &mut ids,
    );
    assert!(out.is_empty());
}

#[test]
fn context_compressed_emits_event_observation() {
    let mut proj = TraceProjector::new("s");
    let mut ids = counter_ids();
    project_json(&mut proj, AgentEvent::TurnStarted, &mut ids);
    let out = project_json(
        &mut proj,
        AgentEvent::ContextCompressed {
            tokens_before: 5000,
            tokens_after: 1200,
        },
        &mut ids,
    );
    assert_eq!(out[0]["type"], "event-create");
    assert_eq!(out[0]["body"]["name"], "context_compaction");
    assert_eq!(out[0]["body"]["metadata"]["tokens_before"], 5000);
    assert_eq!(out[0]["body"]["metadata"]["tokens_after"], 1200);
}

// ---- subagent 投影：前台嵌套 / 后台相邻（用户构想的两层 span）----

/// 前台 subagent：spawn_agent 工具 span 仍张开时，子事件到达 → 独立 subagent span
/// 挂在工具 span 下，子 generation 再挂在 subagent span 下。
#[test]
fn foreground_subagent_nests_under_open_tool_span() {
    let mut proj = TraceProjector::new("s");
    let mut ids = counter_ids();
    project_json(&mut proj, AgentEvent::TurnStarted, &mut ids); // trace = id-1

    // 父 spawn_agent 工具开始（span 张开，登记锚点）。
    let started = project_json(
        &mut proj,
        AgentEvent::ToolCallStarted {
            id: ToolCallId::new("sa-1"),
            name: "spawn_agent".into(),
            fields: ToolCallUpdateFields::default(),
        },
        &mut ids,
    );
    let tool_span_id = started[0]["body"]["id"].as_str().unwrap().to_string();
    assert_eq!(tool_span_id, "id-1-tool-sa-1");

    // 子 turn 第一个事件：LlmCallStarted → 建 subagent span + step span + 子 generation。
    let out = project_json(
        &mut proj,
        AgentEvent::Subagent {
            ancestor_path: vec![ToolCallId::new("sa-1")],
            agent_type: "reviewer".into(),
            inner: Box::new(AgentEvent::LlmCallStarted {
                model: "m".into(),
                attempt: 1,
                request: snapshot(Some("sub system"), "do it"),
            }),
        },
        &mut ids,
    );
    // 三个 observation：subagent span-create + step span-create + generation-create。
    assert_eq!(out[0]["type"], "span-create");
    let subagent_span_id = out[0]["body"]["id"].as_str().unwrap().to_string();
    assert_eq!(subagent_span_id, "id-1-sub-sa-1");
    // subagent span 挂在父工具 span 下（嵌套）。
    assert_eq!(out[0]["body"]["parentObservationId"], tool_span_id);
    assert!(
        out[0]["body"]["name"]
            .as_str()
            .unwrap()
            .contains("reviewer")
    );

    // 子 turn 的 step 挂在 subagent span 下。
    assert_eq!(out[1]["type"], "span-create");
    assert_eq!(out[1]["body"]["name"], "step");
    let sub_step_id = out[1]["body"]["id"].as_str().unwrap().to_string();
    assert_eq!(sub_step_id, "id-1-sub-sa-1-step-1");
    assert_eq!(out[1]["body"]["parentObservationId"], subagent_span_id);

    assert_eq!(out[2]["type"], "generation-create");
    // 子 generation 挂在子 step 下。
    assert_eq!(out[2]["body"]["parentObservationId"], sub_step_id);
    assert_eq!(out[2]["body"]["traceId"], "id-1");
    // 子 generation 的 input 必须还原成 chat messages（system + user）。
    assert_eq!(out[2]["body"]["input"][0]["role"], "system");
    assert_eq!(out[2]["body"]["input"][0]["content"], "sub system");
    assert_eq!(out[2]["body"]["input"][1]["role"], "user");
    assert_eq!(out[2]["body"]["input"][1]["content"], "do it");
}

/// 后台 subagent：spawn_agent 工具 span 先正常关闭、发起 turn 也 TurnEnded，**之后**
/// 子事件才到达——仍能经 session 级锚点把 subagent span 挂回原工具 span 下（相邻而非嵌套）。
#[test]
fn background_subagent_attaches_after_tool_and_turn_closed() {
    let mut proj = TraceProjector::new("s");
    let mut ids = counter_ids();
    project_json(&mut proj, AgentEvent::TurnStarted, &mut ids); // trace = id-1

    // spawn_agent 工具开始 + **立即结束**（后台：返回"已启动"）。
    project_json(
        &mut proj,
        AgentEvent::ToolCallStarted {
            id: ToolCallId::new("sa-9"),
            name: "spawn_agent".into(),
            fields: ToolCallUpdateFields::default(),
        },
        &mut ids,
    );
    let mut done = ToolCallUpdateFields::default();
    done.status = Some(ToolCallStatus::Completed);
    let fin = project_json(
        &mut proj,
        AgentEvent::ToolCallFinished {
            id: ToolCallId::new("sa-9"),
            fields: done,
        },
        &mut ids,
    );
    assert_eq!(fin[0]["type"], "span-update"); // 工具 span 正常关闭

    // 发起 turn 结束。
    project_json(
        &mut proj,
        AgentEvent::TurnEnded {
            reason: StopReason::EndTurn,
            usage: Usage::default(),
        },
        &mut ids,
    );

    // **现在**后台子事件才到达（工具 span 与发起 turn 都已收尾）。
    let out = project_json(
        &mut proj,
        AgentEvent::Subagent {
            ancestor_path: vec![ToolCallId::new("sa-9")],
            agent_type: "worker".into(),
            inner: Box::new(AgentEvent::LlmCallStarted {
                model: "m".into(),
                attempt: 1,
                request: snapshot(None, "bg work"),
            }),
        },
        &mut ids,
    );
    // 仍然建出 subagent span，挂在原工具 span 下（锚点跨 turn 存活）。
    assert_eq!(out[0]["type"], "span-create");
    assert_eq!(out[0]["body"]["id"], "id-1-sub-sa-9");
    assert_eq!(out[0]["body"]["parentObservationId"], "id-1-tool-sa-9");
    // 复用原 trace（即便该 trace 的 turn 已 TurnEnded）。
    assert_eq!(out[0]["body"]["traceId"], "id-1");
    // step span 挂在 subagent span 下。
    assert_eq!(out[1]["type"], "span-create");
    assert_eq!(out[1]["body"]["name"], "step");
    assert_eq!(out[1]["body"]["parentObservationId"], "id-1-sub-sa-9");
    assert_eq!(out[2]["type"], "generation-create");
    assert_eq!(
        out[2]["body"]["parentObservationId"],
        "id-1-sub-sa-9-step-1"
    );
    let gen_id = out[2]["body"]["id"].as_str().unwrap().to_string();

    // 子 turn 的流式输出：output 正文 + thinking。
    let sub_event = |inner: AgentEvent| AgentEvent::Subagent {
        ancestor_path: vec![ToolCallId::new("sa-9")],
        agent_type: "worker".into(),
        inner: Box::new(inner),
    };
    project_json(
        &mut proj,
        sub_event(AgentEvent::AssistantText {
            content: text_block("bg answer"),
        }),
        &mut ids,
    );
    project_json(
        &mut proj,
        sub_event(AgentEvent::AssistantThought {
            content: text_block("bg reasoning"),
        }),
        &mut ids,
    );
    // 本次调用的 usage（流 drain 后到达）→ LlmCallFinished 当即 flush 子 generation。
    let gen_flush = project_json(
        &mut proj,
        sub_event(AgentEvent::LlmCallFinished {
            model: "m".into(),
            attempt: 1,
            usage: Usage {
                input_tokens: Some(11),
                output_tokens: Some(7),
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            },
            error: None,
        }),
        &mut ids,
    );
    // 子 generation-update：output / reasoning / usageDetails 都写到位（与父 turn 同形）。
    let gen_update = gen_flush
        .iter()
        .find(|e| e["type"] == "generation-update" && e["body"]["id"] == gen_id)
        .expect("subagent generation-update present");
    assert_eq!(gen_update["body"]["output"], "bg answer");
    assert_eq!(gen_update["body"]["metadata"]["reasoning"], "bg reasoning");
    assert_eq!(gen_update["body"]["usageDetails"]["input"], 11);
    assert_eq!(gen_update["body"]["usageDetails"]["output"], 7);

    // 子 turn 结束 → 关闭子 step span + 关闭 subagent span。
    let closed = project_json(
        &mut proj,
        sub_event(AgentEvent::TurnEnded {
            reason: StopReason::EndTurn,
            usage: Usage::default(),
        }),
        &mut ids,
    );
    // 子 step 收尾。
    assert!(closed.iter().any(|e| e["type"] == "span-update"
        && e["body"]["id"] == "id-1-sub-sa-9-step-1"
        && e["body"]["endTime"].is_string()));
    // subagent span 收尾（end_time）。
    assert!(closed.iter().any(|e| e["type"] == "span-update"
        && e["body"]["id"] == "id-1-sub-sa-9"
        && e["body"]["endTime"].is_string()));
}

/// 递归 subagent（深度 2）：A 派生 B（path=[A]），B 又派生 C（path=[A,B]）。
/// projector 用 ancestor_path 确定性派生层级，C 的 step/gen 挂在 C 的 subagent span 下，
/// C 的 subagent span 挂在 B 内那次 spawn_agent 工具 span 下。
#[test]
fn recursive_subagent_depth_two_nests_correctly() {
    let mut proj = TraceProjector::new("s");
    let mut ids = counter_ids();
    project_json(&mut proj, AgentEvent::TurnStarted, &mut ids); // trace = id-1

    // 顶层 spawn_agent 工具 A（锚定 trace）。
    project_json(
        &mut proj,
        AgentEvent::ToolCallStarted {
            id: ToolCallId::new("A"),
            name: "spawn_agent".into(),
            fields: ToolCallUpdateFields::default(),
        },
        &mut ids,
    );

    // 子 agent B 内：先一次 llm_call（建 B 的 subagent span + step + gen），
    // 再在 B 里调一次 spawn_agent 工具 "B"（这是 C 的发起 tool span，挂在 B 的 step 下）。
    project_json(
        &mut proj,
        AgentEvent::Subagent {
            ancestor_path: vec![ToolCallId::new("A")],
            agent_type: "coordinator".into(),
            inner: Box::new(AgentEvent::LlmCallStarted {
                model: "m".into(),
                attempt: 1,
                request: snapshot(None, "coordinate"),
            }),
        },
        &mut ids,
    );
    let b_spawn = project_json(
        &mut proj,
        AgentEvent::Subagent {
            ancestor_path: vec![ToolCallId::new("A")],
            agent_type: "coordinator".into(),
            inner: Box::new(AgentEvent::ToolCallStarted {
                id: ToolCallId::new("B"),
                name: "spawn_agent".into(),
                fields: ToolCallUpdateFields::default(),
            }),
        },
        &mut ids,
    );
    // B 内的 spawn_agent 工具 span：id = {B scope}-tool-B，挂在 B 的 step 下。
    assert_eq!(b_spawn[0]["type"], "span-create");
    assert_eq!(b_spawn[0]["body"]["id"], "id-1-sub-A-tool-B");
    assert_eq!(
        b_spawn[0]["body"]["parentObservationId"],
        "id-1-sub-A-step-1"
    );

    // 孙 agent C 的事件：path=[A,B]。建 C 的 subagent span + step + gen。
    let c = project_json(
        &mut proj,
        AgentEvent::Subagent {
            ancestor_path: vec![ToolCallId::new("A"), ToolCallId::new("B")],
            agent_type: "worker".into(),
            inner: Box::new(AgentEvent::LlmCallStarted {
                model: "m".into(),
                attempt: 1,
                request: snapshot(None, "work"),
            }),
        },
        &mut ids,
    );
    // C 的 subagent span：id = {trace}-sub-A-sub-B，父 = B 内的 spawn_agent 工具 span。
    assert_eq!(c[0]["type"], "span-create");
    assert_eq!(c[0]["body"]["id"], "id-1-sub-A-sub-B");
    assert_eq!(c[0]["body"]["parentObservationId"], "id-1-sub-A-tool-B");
    assert!(c[0]["body"]["name"].as_str().unwrap().contains("worker"));
    // C 的 step 挂在 C 的 subagent span 下。
    assert_eq!(c[1]["type"], "span-create");
    assert_eq!(c[1]["body"]["name"], "step");
    assert_eq!(c[1]["body"]["id"], "id-1-sub-A-sub-B-step-1");
    assert_eq!(c[1]["body"]["parentObservationId"], "id-1-sub-A-sub-B");
    // C 的 generation 挂在 C 的 step 下。
    assert_eq!(c[2]["type"], "generation-create");
    assert_eq!(c[2]["body"]["id"], "id-1-sub-A-sub-B-step-1-gen");
    assert_eq!(
        c[2]["body"]["parentObservationId"],
        "id-1-sub-A-sub-B-step-1"
    );
}
