//! `HookStep` 的信封往返单测：构造 step → to_envelope → 喂 mock verdict → 断言控制流 + 字段改动。

use super::*;
use serde_json::json;

// ----- before turn-end -----

fn turn_end(voluntary: bool) -> BeforeTurnEnd {
    BeforeTurnEnd {
        stop_reason: AcpStopReason::EndTurn,
        continues_so_far: 0,
        voluntary,
        feedback: Vec::new(),
    }
}

#[test]
fn turn_end_envelope_carries_state() {
    let env = turn_end(true).to_envelope();
    assert_eq!(env["stop_reason"], "end_turn");
    assert_eq!(env["voluntary"], true);
    assert_eq!(env["continues_so_far"], 0);
}

#[test]
fn turn_end_null_verdict_proceeds() {
    let mut step = turn_end(true);
    let ctrl = step.apply_verdict(&json!({})).expect("verdict");
    assert_eq!(ctrl, HookControl::Proceed);
    assert!(step.feedback.is_empty());
}

#[test]
fn turn_end_continue_injects_feedback() {
    let mut step = turn_end(true);
    let ctrl = step
        .apply_verdict(&json!({
            "control": "continue",
            "additional_context": ["tests not run yet, keep going"],
        }))
        .expect("verdict");
    assert_eq!(ctrl, HookControl::Continue);
    assert_eq!(step.feedback.len(), 1);
}

#[test]
fn turn_end_veto_means_continue() {
    // command hook exit 2 → {"control":"veto"}；turn-end 把 veto 解读为续命。
    let mut step = turn_end(true);
    let ctrl = step
        .apply_verdict(&json!({"control": "veto", "additional_context": ["just test failed"]}))
        .expect("verdict");
    assert_eq!(ctrl, HookControl::Continue);
    assert_eq!(step.feedback.len(), 1);
}

#[test]
fn turn_end_break_with_reason() {
    let mut step = turn_end(true);
    let ctrl = step
        .apply_verdict(&json!({"control": "break", "stop_reason": "refusal"}))
        .expect("verdict");
    assert_eq!(
        ctrl,
        HookControl::Break {
            reason: AcpStopReason::Refusal
        }
    );
}

#[test]
fn unknown_control_errors() {
    let mut step = turn_end(true);
    let err = step.apply_verdict(&json!({"control": "explode"}));
    assert!(matches!(err, Err(VerdictError::UnknownControl(_))));
}

// ----- before ToolApply -----

fn tool_apply() -> BeforeToolApply {
    BeforeToolApply {
        tool_name: "bash".to_string(),
        safety: crate::tool::SafetyClass::ReadOnly,
        args: json!({"command": "ls"}),
        result: None,
    }
}

#[test]
fn tool_apply_envelope_exposes_args_and_safety() {
    let env = tool_apply().to_envelope();
    assert_eq!(env["tool"], "bash");
    assert_eq!(env["args"]["command"], "ls");
    assert_eq!(env["safety"], "read_only");
}

#[test]
fn after_tool_apply_envelope_exposes_output() {
    let step = AfterToolApply {
        tool_name: "bash".to_string(),
        is_error: false,
        output: ToolResultBody::Text {
            text: "hello stdout".to_string(),
        },
        additional_context: Vec::new(),
    };
    let env = step.to_envelope();
    assert_eq!(env["tool"], "bash");
    assert_eq!(env["output"], "hello stdout");
    assert_eq!(env["is_error"], false);
}

#[test]
fn tool_apply_patches_args() {
    let mut step = tool_apply();
    let ctrl = step
        .apply_verdict(&json!({"args": {"command": "ls -la"}}))
        .expect("verdict");
    assert_eq!(ctrl, HookControl::Proceed);
    assert_eq!(step.args["command"], "ls -la");
    assert!(step.result.is_none());
}

#[test]
fn tool_apply_short_circuit_fills_result() {
    let mut step = tool_apply();
    let ctrl = step
        .apply_verdict(&json!({
            "result": {"kind": "text", "text": "blocked by hook"},
            "is_error": true,
        }))
        .expect("verdict");
    // 拦工具 ≠ 结束 turn：control 仍 Proceed，turn 继续；result 被填上。
    assert_eq!(ctrl, HookControl::Proceed);
    let r = step.result.expect("synthetic result");
    assert!(r.is_error);
    assert_eq!(
        r.body,
        ToolResultBody::Text {
            text: "blocked by hook".to_string()
        }
    );
}

#[test]
fn tool_apply_break_ends_turn() {
    let mut step = tool_apply();
    let ctrl = step
        .apply_verdict(&json!({"control": "break"}))
        .expect("verdict");
    assert_eq!(
        ctrl,
        HookControl::Break {
            reason: AcpStopReason::EndTurn
        }
    );
}

#[test]
fn tool_apply_malformed_result_errors() {
    let mut step = tool_apply();
    let err = step.apply_verdict(&json!({"result": {"kind": "bogus"}}));
    assert!(matches!(
        err,
        Err(VerdictError::Malformed {
            field: "result",
            ..
        })
    ));
}

// ----- after Generate -----

#[test]
fn after_generate_envelope_and_observe_only() {
    let mut step = AfterGenerate {
        model: "claude".to_string(),
        usage: Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
        },
        stop: AcpStopReason::EndTurn,
        error: None,
    };
    let env = step.to_envelope();
    assert_eq!(env["model"], "claude");
    assert_eq!(env["usage"]["input_tokens"], 10);
    assert_eq!(env["stop_reason"], "end_turn");

    // 观察型：control 仍可 break，但无"填产出"。
    let ctrl = step.apply_verdict(&json!({})).expect("verdict");
    assert_eq!(ctrl, HookControl::Proceed);
}

// ----- 作用域 / 变更型 step -----

#[test]
fn session_enter_injects_and_breaks() {
    let mut step = AfterSessionEnter {
        cwd: "/repo".to_string(),
        source: SessionSource::New,
        additional_context: Vec::new(),
    };
    assert_eq!(step.to_envelope()["source"], "new");
    let ctrl = step
        .apply_verdict(&json!({"additional_context": ["use rustfmt"], "control": "break"}))
        .expect("verdict");
    assert_eq!(
        ctrl,
        HookControl::Break {
            reason: AcpStopReason::EndTurn
        }
    );
    assert_eq!(step.additional_context.len(), 1);
}

#[test]
fn before_ingest_rewrites_input() {
    let mut step = BeforeIngest {
        source: IngestSource::User,
        input: vec![ContentBlock::from("original")],
    };
    assert_eq!(step.to_envelope()["source"], "user");
    let ctrl = step
        .apply_verdict(&json!({"input": ["rewritten", "extra"]}))
        .expect("verdict");
    assert_eq!(ctrl, HookControl::Proceed);
    assert_eq!(step.input.len(), 2);
}

#[test]
fn before_compact_can_skip() {
    let mut step = BeforeCompact {
        token_estimate: 9000,
        threshold: 8000,
    };
    assert_eq!(step.to_envelope()["threshold"], 8000);
    let ctrl = step
        .apply_verdict(&json!({"control": "skip"}))
        .expect("verdict");
    assert_eq!(ctrl, HookControl::Skip);
}

#[test]
fn before_compact_veto_means_skip() {
    // command hook exit 2 → veto；compact 把 veto 解读为跳过本次压缩。
    let mut step = BeforeCompact {
        token_estimate: 9000,
        threshold: 8000,
    };
    let ctrl = step
        .apply_verdict(&json!({"control": "veto"}))
        .expect("verdict");
    assert_eq!(ctrl, HookControl::Skip);
}

#[test]
fn tool_apply_veto_means_break() {
    // 默认 step（如 ToolApply）把 veto 解读为 Break。
    let mut step = tool_apply();
    let ctrl = step
        .apply_verdict(&json!({"control": "veto"}))
        .expect("verdict");
    assert_eq!(
        ctrl,
        HookControl::Break {
            reason: AcpStopReason::EndTurn
        }
    );
}

#[test]
fn before_generate_short_circuits() {
    let mut step = BeforeGenerate {
        model: "claude".to_string(),
        message_count: 3,
        attempt: 1,
        assistant_text: None,
    };
    let ctrl = step
        .apply_verdict(&json!({"assistant": "synthetic reply", "model": "haiku"}))
        .expect("verdict");
    assert_eq!(ctrl, HookControl::Proceed);
    assert_eq!(step.assistant_text.as_deref(), Some("synthetic reply"));
    assert_eq!(step.model, "haiku");
}

#[test]
fn before_permission_stub_records_resolved() {
    let mut step = BeforePermission {
        tool: "bash".to_string(),
        decision: "ask".to_string(),
        resolved: None,
    };
    let ctrl = step
        .apply_verdict(&json!({"resolved": true}))
        .expect("verdict");
    assert_eq!(ctrl, HookControl::Proceed);
    assert_eq!(step.resolved, Some(true));
}

// ----- pipeline 合并语义 -----

#[test]
fn pipeline_accumulates_data_then_early_exits_on_control() {
    let mut step = tool_apply();
    // 三个 verdict：① 改 args ② 再改 args（看到①的结果）③ break。
    let verdicts = vec![
        json!({"args": {"command": "step1"}}),
        json!({"args": {"command": "step2"}}),
        json!({"control": "break"}),
    ];
    let ctrl = run_step_pipeline(&mut step, verdicts, |_| None);
    assert_eq!(
        ctrl,
        HookControl::Break {
            reason: AcpStopReason::EndTurn
        }
    );
    // 数据轴累积到最后一次改动。
    assert_eq!(step.args["command"], "step2");
}

#[test]
fn pipeline_stops_at_first_control() {
    let mut step = turn_end(true);
    // 第一个 continue 即早退，第二个 verdict 不应被应用。
    let verdicts = vec![
        json!({"control": "continue", "additional_context": ["first"]}),
        json!({"control": "break"}),
    ];
    let ctrl = run_step_pipeline(&mut step, verdicts, |_| None);
    assert_eq!(ctrl, HookControl::Continue);
    assert_eq!(step.feedback.len(), 1); // 只应用了第一个
}

#[test]
fn pipeline_error_handler_can_skip_or_block() {
    let mut step = tool_apply();
    // 第一个 verdict 形态错误；on_error 选择跳过(None)，继续到第二个。
    let verdicts = vec![
        json!({"result": {"kind": "bogus"}}),
        json!({"control": "break"}),
    ];
    let ctrl = run_step_pipeline(&mut step, verdicts, |_| None);
    assert_eq!(
        ctrl,
        HookControl::Break {
            reason: AcpStopReason::EndTurn
        }
    );

    // on_error 选择早退(Some)：错误等价 break。
    let mut step2 = tool_apply();
    let ctrl2 = run_step_pipeline(
        &mut step2,
        vec![json!({"result": {"kind": "bogus"}})],
        |_| {
            Some(HookControl::Break {
                reason: AcpStopReason::Refusal,
            })
        },
    );
    assert_eq!(
        ctrl2,
        HookControl::Break {
            reason: AcpStopReason::Refusal
        }
    );
}

#[test]
fn after_tool_batch_envelope_lists_results() {
    let step = AfterToolBatch {
        results: vec![
            ToolBatchEntry {
                tool_name: "bash".to_string(),
                is_error: false,
            },
            ToolBatchEntry {
                tool_name: "edit".to_string(),
                is_error: true,
            },
        ],
        additional_context: Vec::new(),
    };
    let env = step.to_envelope();
    assert_eq!(env["results"][0]["tool"], "bash");
    assert_eq!(env["results"][1]["is_error"], true);
}
