//! 目标驱动循环的共享状态。
//!
//! `--goal` 模式下，agent 多轮自主跑直到目标达成。机制：
//! - `goal_done` 工具（[`crate::tool::GoalDoneTool`]）—— AI 认为目标达成时调用，
//!   设 [`GoalState::reached`]。
//! - `goal-gate` hook（[`crate::hooks::builtin::GoalGate`]）—— turn 自愿停止时
//!   （`before_turn_end`）读 [`GoalState::is_reached`]：已达成则放行结束，否则
//!   续命（注入"继续工作"反馈）回循环顶再转一轮。
//!
//! 两者跨 turn 阶段共享同一份 `Arc<GoalState>`：工具在某轮写，hook 在之后某轮读。
//!
//! ## 为何是具名结构而非通用状态袋
//!
//! 照 [`crate::session::DefaultSession`] 既有规律（`background` / `compaction_slot`
//! 等都是职责明确的具名结构，不塞万能 `HashMap<String, Value>`）。v0 只有两字段，
//! 但是结构体——以后加 `summary` / `reached_at` / 子目标列表等往里加字段，
//! `ToolContext` / builtin 持的是 `Arc<GoalState>`，字段增减不破接口。

use std::sync::atomic::{AtomicBool, Ordering};

/// 一次目标驱动循环的共享状态。
#[derive(Debug)]
pub struct GoalState {
    /// `--goal` 传入的目标描述。注入 `goal-gate` 续命反馈，让模型每轮都看到目标。
    objective: String,
    /// 目标是否已达成。`goal_done` 工具置位；`goal-gate` hook 读取。
    reached: AtomicBool,
}

impl GoalState {
    #[must_use]
    pub fn new(objective: impl Into<String>) -> Self {
        Self {
            objective: objective.into(),
            reached: AtomicBool::new(false),
        }
    }

    /// 目标描述。
    #[must_use]
    pub fn objective(&self) -> &str {
        &self.objective
    }

    /// 标记目标已达成（`goal_done` 工具调用）。
    pub fn mark_reached(&self) {
        self.reached.store(true, Ordering::SeqCst);
    }

    /// 目标是否已达成（`goal-gate` hook 判定用）。
    #[must_use]
    pub fn is_reached(&self) -> bool {
        self.reached.load(Ordering::SeqCst)
    }
}
