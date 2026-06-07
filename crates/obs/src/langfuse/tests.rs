//! Unit tests for the Langfuse model and projector.
//!
//! Model tests lock down the wire contract (camelCase field names, type discriminant
//! values, `None` skipping);
//! projector tests lock down event translation (trace/generation/span structure, usage
//! mapping, id pairing).

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

/// Deterministic ID generator: `id-1`, `id-2`, …, for easy assertion.
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

/// A minimal request snapshot: system prompt + one user text message.
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
    // The `body` field uses camelCase: `sessionId` instead of `session_id`.
    assert_eq!(v["body"]["id"], "trace-1");
    assert_eq!(v["body"]["sessionId"], "sess-1");
    assert_eq!(v["body"]["name"], "turn");
    assert_eq!(v["body"]["input"], "hello");
    // Unset fields are omitted from the JSON output.
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
    // parentObservationId is omitted when it is None.
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
    // Span has no model / usageDetails.
    assert!(v["body"].get("model").is_none());
    assert!(v["body"].get("usageDetails").is_none());
}

// ---- ingestion response parsing (207 false-positive fix regression) ----

#[test]
fn parses_207_all_success_as_no_errors() {
    // This is a real 207 response body (all success) — errors is empty and should not be
    // treated as an error.
    let body =
        r#"{"successes":[{"id":"be5dbe21-a204-407b-bf52-6ec031164650","status":201}],"errors":[]}"#;
    let parsed: IngestionResponse = serde_json::from_str(body).unwrap();
    assert_eq!(parsed.successes.len(), 1);
    assert_eq!(parsed.successes[0].status, 201);
    assert!(
        parsed.errors.is_empty(),
        "a fully successful 207 should have no errors"
    );
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

/// Serializes the ingestion events produced by the projector into a `Vec<Value>` for
/// easier assertion.
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
    // The trace_id is the first allocated id; the envelope id is the second.
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

    // LlmCallStarted first creates a step span (the container), then creates a generation
    // attached under that step.
    let created = project_json(
        &mut proj,
        AgentEvent::LlmCallStarted {
            model: "claude-opus-4-8".into(),
            attempt: 1,
            request: snapshot(Some("you are helpful"), "hi there"),
        },
        &mut ids,
    );
    // [0] = step span-create; [1] = generation-create (parent = step).
    assert_eq!(created[0]["type"], "span-create");
    assert_eq!(created[0]["body"]["name"], "step");
    let step_id = created[0]["body"]["id"].as_str().unwrap().to_string();
    assert_eq!(step_id, "id-1-step-1");
    // The top-level step is attached directly to the trace (no `parentObservationId`).
    assert!(created[0]["body"].get("parentObservationId").is_none());

    assert_eq!(created[1]["type"], "generation-create");
    let gen_id = created[1]["body"]["id"].as_str().unwrap().to_string();
    assert_eq!(gen_id, "id-1-step-1-gen");
    assert_eq!(created[1]["body"]["parentObservationId"], step_id);
    assert_eq!(created[1]["body"]["traceId"], "id-1");
    assert_eq!(created[1]["body"]["model"], "claude-opus-4-8");
    assert_eq!(created[1]["body"]["metadata"]["attempt"], 1);
    // input is a chat messages array: system first, user second.
    assert_eq!(created[1]["body"]["input"][0]["role"], "system");
    assert_eq!(created[1]["body"]["input"][0]["content"], "you are helpful");
    assert_eq!(created[1]["body"]["input"][1]["role"], "user");
    assert_eq!(created[1]["body"]["input"][1]["content"], "hi there");

    // output/thinking arrives in streaming fashion.
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

    // LlmCallFinished → generation ends immediately (gen duration = pure LLM, no tools).
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
    // The `thinking` field goes into `metadata.reasoning`, not into `output`.
    assert_eq!(finished[0]["body"]["metadata"]["reasoning"], "let me think");
    assert!(finished[0]["body"].get("level").is_none());
    // generation's usageDetails = this call's usage (80/12).
    assert_eq!(finished[0]["body"]["usageDetails"]["input"], 80);
    assert_eq!(finished[0]["body"]["usageDetails"]["output"], 12);

    // TurnEnded closes the step span ([0]), then finalizes the trace ([1], usage =
    // cumulative turn usage).
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
    // Multi-turn (tool loop) scenario: two LLM calls within one turn, each with distinct
    // usage — this is the core of the fix (previously generations had no usage, only the
    // turn total).
    let mut proj = TraceProjector::new("s");
    let mut ids = counter_ids();
    project_json(&mut proj, AgentEvent::TurnStarted, &mut ids);

    // First call: usage 50/10. `LlmCallFinished` immediately closes gen1.
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
    // gen1 finishes with its own LlmCallFinished, carrying its own 50/10.
    assert_eq!(gen1_flush[0]["type"], "generation-update");
    assert_eq!(gen1_flush[0]["body"]["usageDetails"]["input"], 50);
    assert_eq!(gen1_flush[0]["body"]["usageDetails"]["output"], 10);

    // Second call starts → close step-1, open step-2, create gen2.
    let step2 = project_json(
        &mut proj,
        AgentEvent::LlmCallStarted {
            model: "m".into(),
            attempt: 1,
            request: snapshot(None, "second"),
        },
        &mut ids,
    );
    // [0] = step-1 span-update (closed); [1] = step-2 span-create; [2] = gen2 create.
    assert_eq!(step2[0]["type"], "span-update");
    assert_eq!(step2[0]["body"]["id"], "id-1-step-1");
    assert_eq!(step2[1]["type"], "span-create");
    assert_eq!(step2[1]["body"]["id"], "id-1-step-2");
    assert_eq!(step2[2]["type"], "generation-create");
    assert_eq!(step2[2]["body"]["id"], "id-1-step-2-gen");

    // Second call with usage 200/40 (clearly different from the first) → immediately
    // finalize gen2.
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
    // gen2 flush carries its own 200/40, not the turn cumulative 250/50.
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
    // TurnEnded closes step-2 ([0]); the trace ends with a turn total of 250/50 ([1]).
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
    // The error is recorded on the generation; `LlmCallFinished` immediately writes out
    // the level/statusMessage.
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
    // A tool always follows an `llm_call` — first start an LLM call to create the step
    // (container).
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
    // The tool is attached under the current step (sibling to `llm_call`).
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
    // Matches the same span id.
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

    // Real order: the main loop emits `UserPromptCommitted` first, then `TurnStarted`.
    let pre = project_json(
        &mut proj,
        AgentEvent::UserPromptCommitted {
            content: vec![text_block("do something")],
        },
        &mut ids,
    );
    // UserPromptCommitted itself does not produce an ingestion event; it only buffers the
    // input.
    assert!(pre.is_empty());

    let started = project_json(&mut proj, AgentEvent::TurnStarted, &mut ids);
    let trace_id = started[0]["body"]["id"].as_str().unwrap().to_string();
    // input is attached at trace-create time (no need to wait for TurnEnded) — this is
    // the regression point.
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
    // This turn did not emit a separate LlmCallFinished; TurnEnded serves as a fallback —
    // first flush the generation-update ([0]), then close the step span ([1]), and
    // finally send the trace update ([2]).
    assert_eq!(ended[0]["type"], "generation-update");
    assert_eq!(ended[1]["type"], "span-update");
    assert!(ended[1]["body"]["name"].is_null() || ended[1]["body"]["name"] == "step");
    assert_eq!(ended[2]["type"], "trace-create");
    // The trace is updated with the same `trace_id` (merging input, output, and endTime).
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
    // LLM/tool events without a preceding TurnStarted: the projector should not panic and
    // should return empty.
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

// ---- subagent projection: foreground nested / background adjacent (user's envisioned
// two-layer span) ----

/// Foreground subagent: when the `spawn_agent` tool span is still open, an incoming child
/// event creates a separate subagent span nested under the tool span, and the child
/// generation is then nested under that subagent span.
#[test]
fn foreground_subagent_nests_under_open_tool_span() {
    let mut proj = TraceProjector::new("s");
    let mut ids = counter_ids();
    project_json(&mut proj, AgentEvent::TurnStarted, &mut ids); // trace = id-1

    // Parent spawn_agent tool starts (span opens, anchor registered).
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

    // First event of the sub-turn: `LlmCallStarted` → creates subagent span + step span +
    // child generation.
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
    // Three observations: subagent span-create, step span-create, and generation-create.
    assert_eq!(out[0]["type"], "span-create");
    let subagent_span_id = out[0]["body"]["id"].as_str().unwrap().to_string();
    assert_eq!(subagent_span_id, "id-1-sub-sa-1");
    // The subagent span is nested under the parent tool span.
    assert_eq!(out[0]["body"]["parentObservationId"], tool_span_id);
    assert!(
        out[0]["body"]["name"]
            .as_str()
            .unwrap()
            .contains("reviewer")
    );

    // The child turn's step is nested under the subagent span.
    assert_eq!(out[1]["type"], "span-create");
    assert_eq!(out[1]["body"]["name"], "step");
    let sub_step_id = out[1]["body"]["id"].as_str().unwrap().to_string();
    assert_eq!(sub_step_id, "id-1-sub-sa-1-step-1");
    assert_eq!(out[1]["body"]["parentObservationId"], subagent_span_id);

    assert_eq!(out[2]["type"], "generation-create");
    // The child generation is attached under the child step.
    assert_eq!(out[2]["body"]["parentObservationId"], sub_step_id);
    assert_eq!(out[2]["body"]["traceId"], "id-1");
    // The child generation's input must be restored to chat messages (system + user).
    assert_eq!(out[2]["body"]["input"][0]["role"], "system");
    assert_eq!(out[2]["body"]["input"][0]["content"], "sub system");
    assert_eq!(out[2]["body"]["input"][1]["role"], "user");
    assert_eq!(out[2]["body"]["input"][1]["content"], "do it");
}

/// Background subagent: the `spawn_agent` tool span closes normally, the initiating turn
/// also ends with `TurnEnded`, and **only then** do the child events arrive — the
/// subagent span can still be attached under the original tool span via the session-level
/// anchor (as a sibling, not nested).
#[test]
fn background_subagent_attaches_after_tool_and_turn_closed() {
    let mut proj = TraceProjector::new("s");
    let mut ids = counter_ids();
    project_json(&mut proj, AgentEvent::TurnStarted, &mut ids); // trace = id-1

    // spawn_agent tool starts and finishes immediately (background: returns "started").
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
    assert_eq!(fin[0]["type"], "span-update"); // Tool span closed normally

    // Initiate turn end.
    project_json(
        &mut proj,
        AgentEvent::TurnEnded {
            reason: StopReason::EndTurn,
            usage: Usage::default(),
        },
        &mut ids,
    );

    // Now the background sub-event arrives (both the tool span and the initiating turn
    // have been finalized).
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
    // Still creates a subagent span, attached under the original tool span (anchor
    // survives across turns).
    assert_eq!(out[0]["type"], "span-create");
    assert_eq!(out[0]["body"]["id"], "id-1-sub-sa-9");
    assert_eq!(out[0]["body"]["parentObservationId"], "id-1-tool-sa-9");
    // Reuse the original trace even though its turn has already ended.
    assert_eq!(out[0]["body"]["traceId"], "id-1");
    // The step span is attached under the subagent span.
    assert_eq!(out[1]["type"], "span-create");
    assert_eq!(out[1]["body"]["name"], "step");
    assert_eq!(out[1]["body"]["parentObservationId"], "id-1-sub-sa-9");
    assert_eq!(out[2]["type"], "generation-create");
    assert_eq!(
        out[2]["body"]["parentObservationId"],
        "id-1-sub-sa-9-step-1"
    );
    let gen_id = out[2]["body"]["id"].as_str().unwrap().to_string();

    // Streaming output for the sub-turn: output body + thinking.
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
    // The usage from this call (arriving after the stream is drained) triggers
    // `LlmCallFinished` to immediately flush the child generation.
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
    // Sub generation-update: output, reasoning, and usageDetails are all populated (same
    // shape as the parent turn).
    let gen_update = gen_flush
        .iter()
        .find(|e| e["type"] == "generation-update" && e["body"]["id"] == gen_id)
        .expect("subagent generation-update present");
    assert_eq!(gen_update["body"]["output"], "bg answer");
    assert_eq!(gen_update["body"]["metadata"]["reasoning"], "bg reasoning");
    assert_eq!(gen_update["body"]["usageDetails"]["input"], 11);
    assert_eq!(gen_update["body"]["usageDetails"]["output"], 7);

    // Sub-turn ends → close child step span + close subagent span.
    let closed = project_json(
        &mut proj,
        sub_event(AgentEvent::TurnEnded {
            reason: StopReason::EndTurn,
            usage: Usage::default(),
        }),
        &mut ids,
    );
    // Finalize the sub-step.
    assert!(closed.iter().any(|e| e["type"] == "span-update"
        && e["body"]["id"] == "id-1-sub-sa-9-step-1"
        && e["body"]["endTime"].is_string()));
    // Subagent span finalization (end_time).
    assert!(closed.iter().any(|e| e["type"] == "span-update"
        && e["body"]["id"] == "id-1-sub-sa-9"
        && e["body"]["endTime"].is_string()));
}

/// Recursive subagent (depth 2): A spawns B (path=[A]), then B spawns C (path=[A,B]).
/// The projector uses `ancestor_path` to deterministically derive the hierarchy: C's
/// step/gen is nested under C's subagent span,
/// and C's subagent span is nested under the `spawn_agent` tool span inside B.
#[test]
fn recursive_subagent_depth_two_nests_correctly() {
    let mut proj = TraceProjector::new("s");
    let mut ids = counter_ids();
    project_json(&mut proj, AgentEvent::TurnStarted, &mut ids); // trace = id-1

    // Top-level spawn_agent tool A (anchors the trace).
    project_json(
        &mut proj,
        AgentEvent::ToolCallStarted {
            id: ToolCallId::new("A"),
            name: "spawn_agent".into(),
            fields: ToolCallUpdateFields::default(),
        },
        &mut ids,
    );

    // Inside sub-agent B: first, one `llm_call` (creates B's subagent span + step + gen),
    // then a `spawn_agent` tool call "B" inside B (this is C's initiating tool span,
    // attached under B's step).
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
    // The `spawn_agent` tool span inside B: id = `{B scope}-tool-B`, parented under B's
    // step.
    assert_eq!(b_spawn[0]["type"], "span-create");
    assert_eq!(b_spawn[0]["body"]["id"], "id-1-sub-A-tool-B");
    assert_eq!(
        b_spawn[0]["body"]["parentObservationId"],
        "id-1-sub-A-step-1"
    );

    // Subagent C event: path=[A,B]. Create C's subagent span + step + gen.
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
    // C's subagent span: id = {trace}-sub-A-sub-B, parent = the spawn_agent tool span
    // inside B.
    assert_eq!(c[0]["type"], "span-create");
    assert_eq!(c[0]["body"]["id"], "id-1-sub-A-sub-B");
    assert_eq!(c[0]["body"]["parentObservationId"], "id-1-sub-A-tool-B");
    assert!(c[0]["body"]["name"].as_str().unwrap().contains("worker"));
    // C's step is nested under C's subagent span.
    assert_eq!(c[1]["type"], "span-create");
    assert_eq!(c[1]["body"]["name"], "step");
    assert_eq!(c[1]["body"]["id"], "id-1-sub-A-sub-B-step-1");
    assert_eq!(c[1]["body"]["parentObservationId"], "id-1-sub-A-sub-B");
    // C's generation is nested under C's step.
    assert_eq!(c[2]["type"], "generation-create");
    assert_eq!(c[2]["body"]["id"], "id-1-sub-A-sub-B-step-1-gen");
    assert_eq!(
        c[2]["body"]["parentObservationId"],
        "id-1-sub-A-sub-B-step-1"
    );
}
