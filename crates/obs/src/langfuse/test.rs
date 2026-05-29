//! Langfuse model / projector 的单元测试。
//!
//! model 测试锁定 wire 契约（字段名 camelCase、type 判别值、None 跳过）；
//! projector 测试锁定事件翻译（trace/generation/span 结构、usage 映射、id 配对）。

use agent_client_protocol::schema::{
    ContentBlock, StopReason, TextContent, ToolCallId, ToolCallStatus, ToolCallUpdateFields,
};
use defect_agent::event::AgentEvent;
use defect_agent::llm::Usage;
use serde_json::json;

use super::model::{EventKind, IngestionEvent, ObservationBody, ObservationLevel, TraceBody};
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
fn llm_call_lifecycle_creates_then_updates_generation() {
    let mut proj = TraceProjector::new("sess-1");
    let mut ids = counter_ids();
    project_json(&mut proj, AgentEvent::TurnStarted, &mut ids); // trace_id = id-1

    let created = project_json(
        &mut proj,
        AgentEvent::LlmCallStarted {
            model: "claude-opus-4-8".into(),
            attempt: 1,
        },
        &mut ids,
    );
    assert_eq!(created[0]["type"], "generation-create");
    let gen_id = created[0]["body"]["id"].as_str().unwrap().to_string();
    // generation id 派生自 trace_id：{trace}-gen-1
    assert_eq!(gen_id, "id-1-gen-1");
    assert_eq!(created[0]["body"]["traceId"], "id-1");
    assert_eq!(created[0]["body"]["model"], "claude-opus-4-8");
    assert_eq!(created[0]["body"]["metadata"]["attempt"], 1);

    // 助手输出累积。
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

    let finished = project_json(
        &mut proj,
        AgentEvent::LlmCallFinished {
            model: "claude-opus-4-8".into(),
            attempt: 1,
            usage: Usage {
                input_tokens: Some(100),
                output_tokens: Some(20),
                cache_read_input_tokens: Some(8),
                cache_creation_input_tokens: None,
            },
            error: None,
        },
        &mut ids,
    );
    assert_eq!(finished[0]["type"], "generation-update");
    // 同一 generation id（合并 endTime/output/usage）。
    assert_eq!(finished[0]["body"]["id"], gen_id);
    assert_eq!(finished[0]["body"]["output"], "hello world");
    assert_eq!(finished[0]["body"]["usageDetails"]["input"], 100);
    assert_eq!(finished[0]["body"]["usageDetails"]["output"], 20);
    assert_eq!(finished[0]["body"]["usageDetails"]["cache_read_input_tokens"], 8);
    // None 的 cache_creation 不上报。
    assert!(
        finished[0]["body"]["usageDetails"]
            .get("cache_creation_input_tokens")
            .is_none()
    );
    // 无 error → 无 level。
    assert!(finished[0]["body"].get("level").is_none());
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
        },
        &mut ids,
    );
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
    assert_eq!(finished[0]["body"]["level"], "ERROR");
    assert_eq!(finished[0]["body"]["statusMessage"], "rate limited");
    // usage 全 None → 不带 usageDetails。
    assert!(finished[0]["body"].get("usageDetails").is_none());
}

#[test]
fn tool_call_creates_and_updates_span_with_pairing() {
    let mut proj = TraceProjector::new("s");
    let mut ids = counter_ids();
    project_json(&mut proj, AgentEvent::TurnStarted, &mut ids); // trace = id-1

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
    let started = project_json(&mut proj, AgentEvent::TurnStarted, &mut ids);
    let trace_id = started[0]["body"]["id"].as_str().unwrap().to_string();

    project_json(
        &mut proj,
        AgentEvent::UserPromptCommitted {
            content: vec![text_block("do something")],
        },
        &mut ids,
    );
    project_json(
        &mut proj,
        AgentEvent::LlmCallStarted {
            model: "m".into(),
            attempt: 1,
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
    // 用同一 trace_id 更新（合并 input/output/endTime）。
    assert_eq!(ended[0]["body"]["id"], trace_id);
    assert_eq!(ended[0]["body"]["sessionId"], "sess-x");
    assert_eq!(ended[0]["body"]["input"], "do something");
    assert_eq!(ended[0]["body"]["output"], "done");
    assert_eq!(ended[0]["body"]["metadata"]["stop_reason"], "end_turn");
    assert_eq!(ended[0]["body"]["metadata"]["usage"]["input"], 100);
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
