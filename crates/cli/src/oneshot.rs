//! 单轮无人值守模式 —— `defect --message <prompt>`。
//!
//! 定位：CI / 脚本里"跑一个 prompt、产出结果、按成败退出"。与交互 REPL、ACP
//! server 三者**平级**，只共享 [`AgentCore`] 内核：
//!
//! - **不**复用 REPL 的 [`crate::repl`] 渲染（那套带 ANSI / 行编辑，绑死在
//!   `repl` feature 的 crossterm/owo-colors 上）。本模块自带极简、无 ANSI 的
//!   事件投影，可在 `--no-default-features --features oneshot` 下编出不含 TUI
//!   依赖的精简 CI 二进制。
//! - 走进程内直连 `AgentCore`（像 REPL），不走 wire——CI 跑的是自己的 agent，
//!   不需要 ACP 的跨进程通用性。
//!
//! ## 输出约定：stdout = agent 内容，stderr = 框架日志
//!
//! agent 的**全部内容**（助手正文 / 思考 / 工具调用）按事件顺序打到 **stdout**
//! 一条流；框架级诊断（被拒警告、turn error、goal 未达成）走 `tracing`——而
//! `tracing` 由 `defect_obs::init_tracing` 统一写 **stderr**。于是 `2>/dev/null`
//! 干净滤掉框架噪音、保留 agent 完整工作记录；两条流不再共用光标、不再黏连。
//!
//! ## 退出码（CI 判断成败的命脉）
//!
//! 优先级从高到低：`TurnError` > `Refusal` > `MaxTokens`/`MaxTurnRequests` >
//! `Cancelled` > 无人值守被拒(`denied`) > `EndTurn`(0)。见 [`ExitOutcome`]。
//!
//! ## 非交互权限
//!
//! 调用方（`bin/cli.rs`）负责把 session 的 policy 包一层
//! [`defect_agent::policy::NonInteractivePolicy`]，使 `Ask` 降级为 `Deny`、
//! 不在无 TTY 环境挂死等输入。本模块监听事件流里的 `PolicyDecision::Deny`：
//! 一旦发生，经 `tracing` 打警告（→ stderr）并置 `denied` 标志，turn 即便正常
//! `EndTurn` 也用非 0 退出码——fail loud，让 CI 知道"有操作被拒、本次结果不可信"。

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use agent_client_protocol_schema::{
    ContentBlock, SessionId, StopReason, TextContent, ToolCallId,
};
use defect_agent::event::AgentEvent;
use defect_agent::policy::PolicyDecision;
use defect_agent::session::{AgentCore, TurnError};
use futures::{FutureExt, StreamExt};
use tokio::io::{AsyncWriteExt, Stdout};

use crate::args::OutputFormat;
use crate::session_open::open_session;

/// 跑一个单轮 prompt 并返回进程退出码。
///
/// `track_denied = true` 时（调用方包了 `NonInteractivePolicy`）才把
/// `PolicyDecision::Deny` 视作"无人值守缺口"参与退出码——`deny-all` 等用户
/// 明知会拒的模式下应传 `false`，避免误报非 0。
///
/// # Errors
///
/// session 开启失败、stdin 读取失败、stdout/stderr 写入失败。
/// `goal` 为 `Some` 时（`--goal` 模式）：turn 结束后若目标未达成（续命耗尽仍没调
/// `goal_done`），用 Exhausted 退出码——避免 CI 把"跑满轮数仍未达成"误判成功。
pub async fn run(
    agent: Arc<dyn AgentCore>,
    cwd: PathBuf,
    message: String,
    format: OutputFormat,
    resume: Option<SessionId>,
    track_denied: bool,
    goal: Option<Arc<defect_agent::session::GoalState>>,
) -> anyhow::Result<ExitCode> {
    let prompt = resolve_prompt(message).await?;

    let mut out = tokio::io::stdout();

    let session = open_session(&agent, &cwd, resume).await?;

    // 循环外订阅一次，跨本轮排空（含 driver 自主续转 turn）——与交互 REPL /
    // ACP event pump 同构。turn future 只给最终 StopReason，本轮内容经事件流推。
    let mut events = session.subscribe();
    let mut sink = EventSink::new(format, track_denied);

    let prompt_blocks = vec![ContentBlock::Text(TextContent::new(prompt))];
    let turn = session.run_turn(prompt_blocks);
    tokio::pin!(turn);

    let result = loop {
        tokio::select! {
            // biased + 先排事件：turn future 与事件流可能同时就绪（turn 飞快结束、
            // 尾部事件已在 buffer）。先 poll events 保证不漏渲染。
            biased;
            ev = events.next() => {
                if let Some(ev) = ev {
                    sink.emit(&mut out, ev).await?;
                }
            }
            r = &mut turn => break r,
        }
    };

    // turn 已结束，buffer 里可能还有刚 send、未被 poll 的尾部事件——立即就绪的全排掉。
    while let Some(Some(ev)) = events.next().now_or_never() {
        sink.emit(&mut out, ev).await?;
    }

    // goal 模式：turn 正常结束但目标未达成（续命耗尽仍没调 goal_done）→ Exhausted。
    let goal_unreached = goal.as_ref().is_some_and(|g| !g.is_reached());
    if goal_unreached {
        tracing::warn!(
            "goal not reached: the agent stopped (or ran out of turns) without calling `goal_done`"
        );
    }
    let outcome = ExitOutcome::from(&result, sink.denied, goal_unreached);
    sink.finish(&mut out, &result, &outcome).await?;
    out.flush().await?;
    Ok(outcome.code())
}

/// 解析 prompt 来源：`-` 或在 stdin 被管道时从 stdin 读，否则用字面值。
async fn resolve_prompt(message: String) -> anyhow::Result<String> {
    use std::io::IsTerminal;

    let from_stdin = message == "-" || (message.is_empty() && !std::io::stdin().is_terminal());
    if from_stdin {
        use tokio::io::AsyncReadExt;
        let mut buf = String::new();
        tokio::io::stdin().read_to_string(&mut buf).await?;
        Ok(buf)
    } else {
        Ok(message)
    }
}

/// turn 结果到进程退出码的归约。
enum ExitOutcome {
    Success,
    Denied,
    Cancelled,
    MaxTokens,
    Refusal,
    Error,
    /// goal 模式：turn 正常结束但目标未达成（续命耗尽 / 模型放弃）。
    GoalUnreached,
}

impl ExitOutcome {
    fn from(
        result: &Result<StopReason, TurnError>,
        denied: bool,
        goal_unreached: bool,
    ) -> Self {
        match result {
            Err(_) => Self::Error,
            Ok(StopReason::Refusal) => Self::Refusal,
            Ok(StopReason::MaxTokens) | Ok(StopReason::MaxTurnRequests) => Self::MaxTokens,
            Ok(StopReason::Cancelled) => Self::Cancelled,
            // EndTurn（及未来新增的成功类终态）：被拒过 > 目标未达成 > 成功。
            Ok(_) if denied => Self::Denied,
            Ok(_) if goal_unreached => Self::GoalUnreached,
            Ok(_) => Self::Success,
        }
    }

    /// 数值退出码（0 = 成功）。
    fn raw(&self) -> u8 {
        match self {
            Self::Success => 0,
            Self::Error => 1,
            Self::MaxTokens => 2,
            Self::Refusal => 3,
            Self::Denied => 4,
            Self::Cancelled => 5,
            Self::GoalUnreached => 6,
        }
    }

    fn code(&self) -> ExitCode {
        ExitCode::from(self.raw())
    }
}

/// 事件投影器：把 [`AgentEvent`] 流按 [`OutputFormat`] 写到 stdout/stderr。
struct EventSink {
    format: OutputFormat,
    track_denied: bool,
    /// 是否发生过无人值守被拒。
    denied: bool,
    /// `ToolCallId → 工具名`，用于在 `PolicyDecision::Deny` 时报出是哪个工具。
    tool_names: HashMap<ToolCallId, String>,
    /// text 格式下：stdout 上是否还停在一行中间（最后写的不是 `\n`）。goal 模式下
    /// 多轮助手输出之间靠它补换行，避免「上一段尾巴 + 下一段开头」黏成一行。
    mid_line: bool,
}

impl EventSink {
    fn new(format: OutputFormat, track_denied: bool) -> Self {
        Self {
            format,
            track_denied,
            denied: false,
            tool_names: HashMap::new(),
            mid_line: false,
        }
    }

    async fn emit(&mut self, out: &mut Stdout, event: AgentEvent) -> anyhow::Result<()> {
        // 记录工具名（任何格式都要，用于被拒报告）。
        if let AgentEvent::ToolCallStarted { id, name, fields } = &event {
            let label = fields.title.clone().unwrap_or_else(|| name.clone());
            self.tool_names.insert(id.clone(), label);
        }

        // 无人值守被拒：框架级诊断走 tracing（→ stderr）+ 置标志（fail loud）。
        if self.track_denied
            && let AgentEvent::PolicyDecision {
                id,
                decision: PolicyDecision::Deny,
            } = &event
        {
            self.denied = true;
            let tool = self
                .tool_names
                .get(id)
                .map(String::as_str)
                .unwrap_or("<unknown>");
            tracing::warn!(
                tool = %tool,
                "tool denied: no operator present to approve (non-interactive)"
            );
        }

        match self.format {
            OutputFormat::Json => self.emit_json(out, &event).await,
            OutputFormat::Text => self.emit_text(out, &event).await,
            OutputFormat::Quiet => Ok(()),
        }
    }

    /// NDJSON：每个事件一行。`AgentEvent` 已 derive Serialize（`LlmCallStarted.request`
    /// 是 `#[serde(skip)]`，不会进 JSON）。
    async fn emit_json(&self, out: &mut Stdout, event: &AgentEvent) -> anyhow::Result<()> {
        let line = serde_json::to_string(event)?;
        write_raw(out, &line).await?;
        write_raw(out, "\n").await
    }

    /// 纯文本：**agent 的全部内容**（正文 / 思考 / 工具）都到 stdout，按事件顺序
    /// 一条流——框架级日志（tracing）走 stderr，两者井水不犯河水（见模块头 §输出约定）。
    ///
    /// 边界换行：助手正文常无尾换行，而紧随其后可能是思考 / 工具行或下一轮生成。
    /// 在切到「非正文行」（思考 / 工具）或开新生成段前，若 stdout 还停在行中间就补
    /// 一个 `\n`，让每段各占整行、不黏连。
    async fn emit_text(&mut self, out: &mut Stdout, event: &AgentEvent) -> anyhow::Result<()> {
        match event {
            // 新一次 LLM 生成开始：上一段助手正文若没换行，先补一个再开新段。
            AgentEvent::LlmCallStarted { .. } | AgentEvent::TurnEnded { .. } => {
                self.break_line(out).await?;
            }
            AgentEvent::AssistantText { content } => {
                if let Some(text) = block_text(content)
                    && !text.is_empty()
                {
                    write_raw(out, &text).await?;
                    out.flush().await?;
                    self.mid_line = !text.ends_with('\n');
                }
            }
            AgentEvent::AssistantThought { content } => {
                if let Some(text) = block_text(content) {
                    self.break_line(out).await?;
                    write(out, &format!("[thinking] {text}\n")).await?;
                    out.flush().await?;
                }
            }
            AgentEvent::ToolCallStarted { name, fields, .. } => {
                self.break_line(out).await?;
                let title = fields.title.clone().unwrap_or_else(|| name.clone());
                write(out, &format!("[tool] {title}\n")).await?;
                out.flush().await?;
            }
            _ => {}
        }
        Ok(())
    }

    /// 若 stdout 还停在行中间，补一个 `\n` 收尾并 flush。用于在「非正文行」（思考 /
    /// 工具）或新生成段前，让上一段助手正文独占整行——同一条流里也要按行分隔。
    async fn break_line(&mut self, out: &mut Stdout) -> anyhow::Result<()> {
        if self.mid_line {
            write_raw(out, "\n").await?;
            out.flush().await?;
            self.mid_line = false;
        }
        Ok(())
    }

    /// turn 结束后的收尾输出。框架级诊断（turn error）走 tracing（→ stderr）。
    async fn finish(
        &self,
        out: &mut Stdout,
        result: &Result<StopReason, TurnError>,
        outcome: &ExitOutcome,
    ) -> anyhow::Result<()> {
        if let Err(e) = result {
            tracing::error!(error = %e, "turn error");
        }
        match self.format {
            OutputFormat::Text => {
                // 助手正文流式无尾随换行时补一个，避免和后续 shell 提示符黏在一起；
                // 已在行首（mid_line=false）则不补，免得多出一行空白。
                if self.mid_line {
                    write_raw(out, "\n").await?;
                }
            }
            OutputFormat::Json => {
                // 末行汇总：最终状态 + 退出码语义。
                let summary = serde_json::json!({
                    "type": "oneshot_result",
                    "stop_reason": result.as_ref().ok().map(|r| format!("{r:?}")),
                    "error": result.as_ref().err().map(|e| e.to_string()),
                    "denied": self.denied,
                    "exit_code": outcome.raw(),
                });
                write_raw(out, &summary.to_string()).await?;
                write_raw(out, "\n").await?;
            }
            OutputFormat::Quiet => {}
        }
        Ok(())
    }
}

/// 从 [`ContentBlock`] 取文本；非文本块返回 `None`。
fn block_text(block: &ContentBlock) -> Option<String> {
    match block {
        ContentBlock::Text(t) => Some(t.text.clone()),
        _ => None,
    }
}

/// 写一段字符串到任意 async writer。
async fn write<W>(out: &mut W, s: &str) -> anyhow::Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    out.write_all(s.as_bytes()).await?;
    Ok(())
}

/// `write` 的别名，语义上强调"不加任何修饰原样写"。
async fn write_raw<W>(out: &mut W, s: &str) -> anyhow::Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    write(out, s).await
}
