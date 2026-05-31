//! Hook step-context：typestate + 信封。
//!
//! 设计详见 `docs/internal/hook-step-context.md`。
//!
//! ## 一句话
//!
//! 每个挂载点对应一个独立的 **step 类型**（typestate）。同一份 step state 被两种 hook 消费——
//! 内部 Rust hook 直接拿强类型 + 改字段；用户配置 hook 经 [`HookStep::to_envelope`] 的 JSON 信封
//! 看世界、经 [`HookStep::apply_verdict`] 把输出 JSON 应用回来。**能力对等、可见面一致**，只是表达
//! 媒介不同。
//!
//! ## 两条公理
//!
//! 1. **typestate**：不用一个带 variant 字段的大 enum，每个挂载点一个具体 struct。可见面编译期锁死；
//!    `Option` 的有/无编码"已产出 / 将产出"——将产出的 `Option` 被填上 = short-circuit。
//! 2. **调用型 vs 变更型不对称**：调用型（Generate / ToolApply / Permission）填 `Option` 跳过；
//!    变更型（Compact / Ingest）退化成 veto / rewrite，不走"填 Option"。
//!
//! ## 落地范围（第 1 步）
//!
//! 本模块交付**类型 + 信封 + 单测**，**不接任何挂载点**（call site 接入是后续 PR）。当前实现了基础
//! 设施（[`HookControl`] / [`HookStep`] / 信封通用约定）+ 3 个代表性 step：[`BeforeTurnEnd`]（控制
//! 分叉）、[`BeforeToolApply`]（调用型 short-circuit）、[`AfterGenerate`]（观察型）。其余 10 个 step
//! 是同形态的机械填充。

use agent_client_protocol_schema::{ContentBlock, StopReason as AcpStopReason};
use serde_json::{Value, json};

use crate::llm::{ToolResultBody, Usage};
use crate::tool::SafetyClass;

/// 全部挂载点的 `event_name`（snake_case）——配置层校验事件名、CLI 装配分桶的唯一真相源。
///
/// 顺序无意义；新增 step 时在此追加一行，配置层即自动认这个新事件名（无需改 config crate）。
pub const ALL_EVENT_NAMES: &[&str] = &[
    "after_session_enter",
    "after_turn_enter",
    "before_ingest",
    "after_ingest",
    "before_compact",
    "after_compact",
    "before_generate",
    "after_generate",
    "before_permission",
    "after_permission",
    "before_tool_apply",
    "after_tool_apply",
    "after_tool_batch",
    "before_turn_end",
];

/// 某个事件名是否是已知挂载点。配置层用它 fail-fast 掉拼错的事件键。
#[must_use]
pub fn is_known_event(name: &str) -> bool {
    ALL_EVENT_NAMES.contains(&name)
}

// ---------------------------------------------------------------------------
// 控制流
// ---------------------------------------------------------------------------

/// 一个 hook 对**控制流**的指示（轴二）。数据注入（轴一）走 step 的 `&mut` 字段，不在这里。
///
/// 位置决定哪些变体有意义：`Break` 任何 step 可用；`Continue` 仅 [`BeforeTurnEnd`]；`Skip` 仅
/// `before Compact`。引擎对越权变体降级 + warn（见 `apply_verdict` 的校验）。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum HookControl {
    /// 不干预控制流——step 带着 ctx 上已发生的数据改动正常往下走。对应信封 `control: null`。
    #[default]
    Proceed,
    /// 结束当前 turn，带最终停止原因。任何 step 可用。
    Break { reason: AcpStopReason },
    /// 不结束、回循环顶再转一轮。仅 [`BeforeTurnEnd`] 有意义（且须先注入，见设计 §4）。
    Continue,
    /// 跳过本 step 的真实调用。仅 `before Compact`（veto 压缩）有意义。
    Skip,
}

/// 解析信封 verdict 时的错误。
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum VerdictError {
    #[error("hook verdict `control` is not a known directive: {0:?}")]
    UnknownControl(String),

    #[error("hook verdict field `{field}` is malformed: {reason}")]
    Malformed { field: &'static str, reason: String },
}

// ---------------------------------------------------------------------------
// HookStep trait
// ---------------------------------------------------------------------------

/// 一个挂载点的 step state。两种 hook 共同消费：
/// - 内部 Rust hook：直接拿 `&mut Self`，改字段 = 注入，自行返回 [`HookControl`]。
/// - 用户配置 hook：[`Self::to_envelope`] → JSON 喂 stdin/模板；handler 输出 JSON →
///   [`Self::apply_verdict`] 应用回 step（数据改动）并解析出 [`HookControl`]。
pub trait HookStep: Send {
    /// 事件名（snake_case）。信封头与 matcher 用。
    fn event_name(&self) -> &'static str;

    /// 投影成**输入信封**——喂 command stdin / prompt 模板。含通用头 + step 专属字段。
    fn to_envelope(&self) -> Value;

    /// 把 handler 的**输出 verdict**（JSON）应用回本 step：解析通用 `control` /
    /// `additional_context`，再处理 step 专属的"填产出"字段。返回控制指示。
    ///
    /// # Errors
    ///
    /// verdict 的 `control` 是未知值、或专属字段形态错误时返回 [`VerdictError`]。
    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError>;
}

// ---------------------------------------------------------------------------
// 信封通用约定
// ---------------------------------------------------------------------------

/// 解析 verdict 里的通用 `control` 字段。`null` / 缺省 → [`HookControl::Proceed`]。
///
/// `break` 可带 `stop_reason`（缺省 `end_turn`）。校验在调用方做（哪个 step 允许哪个 control）。
fn parse_control(verdict: &Value) -> Result<HookControl, VerdictError> {
    // 默认：`veto` 解读为 `Break`（多数 step 的否决语义）。turn-end / compact 用
    // `parse_control_veto` 覆盖成自己的语义。
    parse_control_veto(verdict, HookControl::Break {
        reason: AcpStopReason::EndTurn,
    })
}

/// 同 [`parse_control`]，但把抽象的 `"veto"` 控制（command hook exit 2 产生）解读为 `veto_as`——
/// 让每个 step 按自己的否决语义翻译（turn-end→Continue、compact→Skip、其余→Break）。
fn parse_control_veto(verdict: &Value, veto_as: HookControl) -> Result<HookControl, VerdictError> {
    let Some(ctrl) = verdict.get("control") else {
        return Ok(HookControl::Proceed);
    };
    match ctrl {
        Value::Null => Ok(HookControl::Proceed),
        Value::String(s) => match s.as_str() {
            "proceed" => Ok(HookControl::Proceed),
            "continue" => Ok(HookControl::Continue),
            "skip" => Ok(HookControl::Skip),
            "veto" => Ok(veto_as),
            "break" => {
                let reason = verdict
                    .get("stop_reason")
                    .and_then(Value::as_str)
                    .map_or(AcpStopReason::EndTurn, parse_stop_reason);
                Ok(HookControl::Break { reason })
            }
            other => Err(VerdictError::UnknownControl(other.to_string())),
        },
        other => Err(VerdictError::UnknownControl(other.to_string())),
    }
}

/// 解析 verdict 里的 `additional_context`：接受字符串数组（用户 hook 最自然的形态），
/// 每条转成一个文本 [`ContentBlock`]。缺省 → 空。
fn parse_additional_context(verdict: &Value) -> Result<Vec<ContentBlock>, VerdictError> {
    let Some(v) = verdict.get("additional_context") else {
        return Ok(Vec::new());
    };
    match v {
        Value::Null => Ok(Vec::new()),
        Value::Array(items) => items
            .iter()
            .map(|item| {
                item.as_str().map(ContentBlock::from).ok_or_else(|| {
                    VerdictError::Malformed {
                        field: "additional_context",
                        reason: "each entry must be a string".to_string(),
                    }
                })
            })
            .collect(),
        _ => Err(VerdictError::Malformed {
            field: "additional_context",
            reason: "must be an array of strings".to_string(),
        }),
    }
}

/// [`AcpStopReason`] → snake_case 字符串（信封用）。
fn stop_reason_str(reason: AcpStopReason) -> &'static str {
    match reason {
        AcpStopReason::EndTurn => "end_turn",
        AcpStopReason::MaxTokens => "max_tokens",
        AcpStopReason::MaxTurnRequests => "max_turn_requests",
        AcpStopReason::Refusal => "refusal",
        AcpStopReason::Cancelled => "cancelled",
        _ => "end_turn",
    }
}

/// [`ToolResultBody`] → 信封 JSON。Text/Json 直接放；多模态 Content 退化成文本摘要
/// （图片块标注占位），让 hook 信封保持紧凑可读。
fn tool_result_body_to_json(body: &ToolResultBody) -> Value {
    match body {
        ToolResultBody::Text { text } => Value::String(text.clone()),
        ToolResultBody::Json { value } => value.clone(),
        ToolResultBody::Content { blocks } => {
            use crate::llm::ToolResultContent;
            let text: String = blocks
                .iter()
                .map(|b| match b {
                    ToolResultContent::Text { text } => text.clone(),
                    ToolResultContent::Image { mime, .. } => format!("[image: {mime}]"),
                })
                .collect::<Vec<_>>()
                .join("\n");
            Value::String(text)
        }
    }
}

/// [`SafetyClass`] → snake_case 字符串（信封用，与引擎侧 `parse_safety` 对称）。
fn safety_str(s: SafetyClass) -> &'static str {
    match s {
        SafetyClass::ReadOnly => "read_only",
        SafetyClass::Mutating => "mutating",
        SafetyClass::Destructive => "destructive",
        SafetyClass::Network => "network",
    }
}

/// snake_case 字符串 → [`AcpStopReason`]。未知值回退 `EndTurn`。
fn parse_stop_reason(s: &str) -> AcpStopReason {
    match s {
        "max_tokens" => AcpStopReason::MaxTokens,
        "max_turn_requests" => AcpStopReason::MaxTurnRequests,
        "refusal" => AcpStopReason::Refusal,
        "cancelled" => AcpStopReason::Cancelled,
        _ => AcpStopReason::EndTurn,
    }
}

// ---------------------------------------------------------------------------
// 代表 step 1：before turn-end（控制分叉点，默认 Break）
// ---------------------------------------------------------------------------

/// `before turn-end`：turn 唯一的自愿出口判定。**默认 `Break`**——"什么都不干预"= 放它停。
///
/// hook 返回 [`HookControl::Continue`] = 续命：把 [`Self::feedback`] 注入 history（落地时作为
/// user 消息 append），不结束、回循环顶再转一轮。`continue` 仅在 [`Self::voluntary`] 时生效——
/// 被动停止（Refusal / MaxTokens / Cancelled / MaxTurnRequests）忽略 continue，否则 hook 能绕过
/// request cap 无限续命。
#[derive(Debug, Clone)]
pub struct BeforeTurnEnd {
    /// 到达本判定的停止原因。
    pub stop_reason: AcpStopReason,
    /// 本 turn 已被 hook 续命几次（hook 自行判断收手；循环内另有硬上限兜底）。
    pub continues_so_far: u32,
    /// 是否自愿停止（LLM 说 EndTurn / 空 tool_use）。仅自愿时 `Continue` 生效。
    pub voluntary: bool,
    /// 续命时要注入 history 的反馈。`apply_verdict` 把 verdict 的 `additional_context` 填进来；
    /// 内部 Rust hook 直接 push。落地时由循环作为 user 消息 append。
    pub feedback: Vec<ContentBlock>,
}

impl HookStep for BeforeTurnEnd {
    fn event_name(&self) -> &'static str {
        "before_turn_end"
    }

    fn to_envelope(&self) -> Value {
        json!({
            "stop_reason": stop_reason_str(self.stop_reason),
            "continues_so_far": self.continues_so_far,
            "voluntary": self.voluntary,
        })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        // turn-end 的"否决（veto）"= 续命（Continue）：command hook exit 2 在这里意味着"别停"。
        let control = parse_control_veto(verdict, HookControl::Continue)?;
        let ctx = parse_additional_context(verdict)?;
        // 在 turn-end，additional_context 即续命反馈。
        self.feedback.extend(ctx);
        Ok(control)
    }
}

// ---------------------------------------------------------------------------
// 代表 step 2：before ToolApply（调用型，short-circuit = 填 result）
// ---------------------------------------------------------------------------

/// 一个被 hook 合成的工具结果——填上 [`BeforeToolApply::result`] 即"拦掉这个工具"。
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticToolResult {
    pub body: ToolResultBody,
    pub is_error: bool,
}

/// `before ToolApply`（每工具）：调用型变换的入口。
///
/// 两种干预正交：
/// - **改 args**（数据轴）：改写传给工具的参数。
/// - **填 result**（short-circuit）：`Some` = 不真跑工具、用这个合成输出当结果，**turn 继续**。
///   这与 `Break`（结束整个 turn）控制流完全不同——别把"拦一个工具"和"结束 turn"混为一谈。
#[derive(Debug, Clone)]
pub struct BeforeToolApply {
    pub tool_name: String,
    /// 工具的 safety 等级——进信封供 matcher 的 safety 过滤用。
    pub safety: SafetyClass,
    /// 可改的工具参数。
    pub args: Value,
    /// 将产出的结果。`None` = 真去跑工具；`Some` = short-circuit。
    pub result: Option<SyntheticToolResult>,
}

impl HookStep for BeforeToolApply {
    fn event_name(&self) -> &'static str {
        "before_tool_apply"
    }

    fn to_envelope(&self) -> Value {
        json!({
            "tool": self.tool_name,
            "safety": safety_str(self.safety),
            "args": self.args,
        })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        let control = parse_control(verdict)?;

        // 数据轴：改 args。
        if let Some(new_args) = verdict.get("args") {
            self.args = new_args.clone();
        }

        // short-circuit：填 result。verdict `result` 形如 ToolResultBody + 可选 is_error。
        if let Some(r) = verdict.get("result").filter(|r| !r.is_null()) {
            let body: ToolResultBody =
                serde_json::from_value(r.clone()).map_err(|e| VerdictError::Malformed {
                    field: "result",
                    reason: e.to_string(),
                })?;
            let is_error = verdict
                .get("is_error")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            self.result = Some(SyntheticToolResult { body, is_error });
        }

        Ok(control)
    }
}

// ---------------------------------------------------------------------------
// 代表 step 3：after Generate（观察型，已产出非 Option）
// ---------------------------------------------------------------------------

/// `after Generate`：LLM 调用已返回。**观察型**——usage / stop / error 都已产出（非 Option），
/// 没有"填产出"的余地；要影响下一轮走 [`BeforeTurnEnd`]。仅 `Break` 与观察有意义。
#[derive(Debug, Clone)]
pub struct AfterGenerate {
    pub model: String,
    pub usage: Usage,
    pub stop: AcpStopReason,
    pub error: Option<String>,
}

impl HookStep for AfterGenerate {
    fn event_name(&self) -> &'static str {
        "after_generate"
    }

    fn to_envelope(&self) -> Value {
        json!({
            "model": self.model,
            "usage": self.usage,
            "stop_reason": stop_reason_str(self.stop),
            "error": self.error,
        })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        // 观察型：无可填产出，只接受控制（通常仅 break）；additional_context 此处无落点，忽略。
        parse_control(verdict)
    }
}

// ---------------------------------------------------------------------------
// 作用域 step：after session enter / after turn enter（无产出，可注入 / 可 break）
// ---------------------------------------------------------------------------

/// session 来源：新建 or resume。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionSource {
    New,
    Resume,
}

/// `after session enter`：session 作用域已进入。可注入 system 后缀 / `Break` 拒开。
#[derive(Debug, Clone)]
pub struct AfterSessionEnter {
    pub cwd: String,
    pub source: SessionSource,
    /// 注入到 system prompt 的后缀（`apply_verdict` 从 additional_context 填）。
    pub additional_context: Vec<ContentBlock>,
}

impl HookStep for AfterSessionEnter {
    fn event_name(&self) -> &'static str {
        "after_session_enter"
    }

    fn to_envelope(&self) -> Value {
        json!({
            "cwd": self.cwd,
            "source": match self.source { SessionSource::New => "new", SessionSource::Resume => "resume" },
        })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        self.additional_context.extend(parse_additional_context(verdict)?);
        parse_control(verdict)
    }
}

/// `after turn enter`：turn 作用域已进入、但本轮输入尚未摄入。可注入 / `Break` 拒该 turn。
#[derive(Debug, Clone)]
pub struct AfterTurnEnter {
    pub is_subagent: bool,
    pub agent_type: Option<String>,
    pub additional_context: Vec<ContentBlock>,
}

impl HookStep for AfterTurnEnter {
    fn event_name(&self) -> &'static str {
        "after_turn_enter"
    }

    fn to_envelope(&self) -> Value {
        json!({
            "is_subagent": self.is_subagent,
            "agent_type": self.agent_type,
        })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        self.additional_context.extend(parse_additional_context(verdict)?);
        parse_control(verdict)
    }
}

// ---------------------------------------------------------------------------
// Ingest step：before / after（变更型：rewrite 输入 / veto）
// ---------------------------------------------------------------------------

/// 本轮待摄入输入的来源。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngestSource {
    /// 首轮：用户 prompt。
    User,
    /// 续命轮：before turn-end 注入的反馈。
    Continuation,
    /// 后台任务回流：`run_in_background` 子任务完成后由 session driver 起的自主续转 turn，
    /// 其输入是延迟的工具结果而非用户发言。详见 `docs/proposals/task-arrange.md` §5.1。
    Background,
}

/// `before Ingest`：摄入本轮输入之前。可改写整条待摄入输入 / `Break` 拒该 turn。
///
/// 变更型——short-circuit 是 `Break`（拒掉），不是"填结果"（无可分离产出）。空摄入轮 `input` 为空。
#[derive(Debug, Clone)]
pub struct BeforeIngest {
    pub source: IngestSource,
    /// 可改写的待摄入输入。
    pub input: Vec<ContentBlock>,
}

impl HookStep for BeforeIngest {
    fn event_name(&self) -> &'static str {
        "before_ingest"
    }

    fn to_envelope(&self) -> Value {
        // 暴露输入文本（拼接 Text 块）让 hook 能看到/据此改写；非文本块不进信封但仍在 step 上。
        let text: String = self
            .input
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        json!({
            "source": match self.source {
                IngestSource::User => "user",
                IngestSource::Continuation => "continuation",
                IngestSource::Background => "background",
            },
            "input": text,
            "input_len": self.input.len(),
        })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        // rewrite：verdict 的 `input` 可为字符串（整条替换成一个文本块）或字符串数组。
        if let Some(v) = verdict.get("input").filter(|v| !v.is_null()) {
            self.input = match v {
                Value::String(s) => vec![ContentBlock::from(s.as_str())],
                _ => parse_block_array(v, "input")?,
            };
        }
        parse_control(verdict)
    }
}

/// `after Ingest`：输入已并入 history。仅可注入。
#[derive(Debug, Clone)]
pub struct AfterIngest {
    pub committed_len: usize,
    pub additional_context: Vec<ContentBlock>,
}

impl HookStep for AfterIngest {
    fn event_name(&self) -> &'static str {
        "after_ingest"
    }

    fn to_envelope(&self) -> Value {
        json!({ "committed_len": self.committed_len })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        self.additional_context.extend(parse_additional_context(verdict)?);
        parse_control(verdict)
    }
}

// ---------------------------------------------------------------------------
// Compact step：before（veto only）/ after（观察）
// ---------------------------------------------------------------------------

/// `before Compact`：压缩之前。变更型——short-circuit = `Skip`（veto 本次压缩），无"填结果"。
#[derive(Debug, Clone)]
pub struct BeforeCompact {
    pub token_estimate: u64,
    pub threshold: u64,
}

impl HookStep for BeforeCompact {
    fn event_name(&self) -> &'static str {
        "before_compact"
    }

    fn to_envelope(&self) -> Value {
        json!({ "token_estimate": self.token_estimate, "threshold": self.threshold })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        // compact 的"否决（veto）"= 跳过本次压缩（Skip）。
        parse_control_veto(verdict, HookControl::Skip)
    }
}

/// `after Compact`：压缩完成。仅可注入 / 观察。
#[derive(Debug, Clone)]
pub struct AfterCompact {
    pub tokens_before: u64,
    pub tokens_after: u64,
    pub additional_context: Vec<ContentBlock>,
}

impl HookStep for AfterCompact {
    fn event_name(&self) -> &'static str {
        "after_compact"
    }

    fn to_envelope(&self) -> Value {
        json!({ "tokens_before": self.tokens_before, "tokens_after": self.tokens_after })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        self.additional_context.extend(parse_additional_context(verdict)?);
        parse_control(verdict)
    }
}

// ---------------------------------------------------------------------------
// Generate step：before（改 request / short-circuit）
// ---------------------------------------------------------------------------

/// `before Generate`：LLM 调用之前。调用型——可改 request 字段、或填 `assistant_text` short-circuit
/// （用一条合成回复跳过真实 LLM 调用）。
#[derive(Debug, Clone)]
pub struct BeforeGenerate {
    pub model: String,
    pub message_count: usize,
    pub attempt: u32,
    /// short-circuit：`Some` = 不调 LLM、用这段合成 assistant 文本当回复。落地时建成 Message。
    pub assistant_text: Option<String>,
}

impl HookStep for BeforeGenerate {
    fn event_name(&self) -> &'static str {
        "before_generate"
    }

    fn to_envelope(&self) -> Value {
        json!({
            "model": self.model,
            "message_count": self.message_count,
            "attempt": self.attempt,
        })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        if let Some(m) = verdict.get("model").and_then(Value::as_str) {
            self.model = m.to_string();
        }
        if let Some(a) = verdict.get("assistant").and_then(Value::as_str) {
            self.assistant_text = Some(a.to_string());
        }
        parse_control(verdict)
    }
}

// ---------------------------------------------------------------------------
// Permission step：before（代答；v0 仅打桩）/ after（观察）
// ---------------------------------------------------------------------------

/// `before Permission`：向用户请求授权之前。v0 仅打桩 observe——`resolved` 代答能力先不接
/// （policy 仍是放行权威，见 hooks.md §7.3）。桩留好，未来开。
#[derive(Debug, Clone)]
pub struct BeforePermission {
    pub tool: String,
    /// 当前 policy 决策（"allow" / "deny" / "ask"）。
    pub decision: String,
    /// 代答结果。v0 不消费。
    pub resolved: Option<bool>,
}

impl HookStep for BeforePermission {
    fn event_name(&self) -> &'static str {
        "before_permission"
    }

    fn to_envelope(&self) -> Value {
        json!({ "tool": self.tool, "decision": self.decision })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        // v0：仅接受 control（通常 break）；resolved 代答桩留着但不在此消费。
        if let Some(r) = verdict.get("resolved").and_then(Value::as_bool) {
            self.resolved = Some(r);
        }
        parse_control(verdict)
    }
}

/// `after Permission`：授权结果已定。观察型。
#[derive(Debug, Clone)]
pub struct AfterPermission {
    pub tool: String,
    pub granted: bool,
}

impl HookStep for AfterPermission {
    fn event_name(&self) -> &'static str {
        "after_permission"
    }

    fn to_envelope(&self) -> Value {
        json!({ "tool": self.tool, "granted": self.granted })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        parse_control(verdict)
    }
}

// ---------------------------------------------------------------------------
// ToolApply step：after（每工具）/ after ToolBatch（整批）
// ---------------------------------------------------------------------------

/// `after ToolApply`（每工具）：工具已产出结果。可注入（拼进 tool_result）/ `Break`。
#[derive(Debug, Clone)]
pub struct AfterToolApply {
    pub tool_name: String,
    pub is_error: bool,
    /// 工具产出的结果体（已产出，非 Option）——进信封供 hook 看到工具输出内容。
    pub output: ToolResultBody,
    pub additional_context: Vec<ContentBlock>,
}

impl HookStep for AfterToolApply {
    fn event_name(&self) -> &'static str {
        "after_tool_apply"
    }

    fn to_envelope(&self) -> Value {
        json!({
            "tool": self.tool_name,
            "is_error": self.is_error,
            "output": tool_result_body_to_json(&self.output),
        })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        self.additional_context.extend(parse_additional_context(verdict)?);
        parse_control(verdict)
    }
}

/// 一批并行工具结果的摘要项（信封用）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolBatchEntry {
    pub tool_name: String,
    pub is_error: bool,
}

/// `after ToolBatch`：一整批并行工具结束。可注入 / `Break`（graceful，见 proposal §7）。
#[derive(Debug, Clone)]
pub struct AfterToolBatch {
    pub results: Vec<ToolBatchEntry>,
    pub additional_context: Vec<ContentBlock>,
}

impl HookStep for AfterToolBatch {
    fn event_name(&self) -> &'static str {
        "after_tool_batch"
    }

    fn to_envelope(&self) -> Value {
        json!({
            "results": self.results.iter().map(|e| json!({
                "tool": e.tool_name,
                "is_error": e.is_error,
            })).collect::<Vec<_>>(),
        })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        self.additional_context.extend(parse_additional_context(verdict)?);
        parse_control(verdict)
    }
}

// ---------------------------------------------------------------------------
// Pipeline：多个 verdict 在一个 step 上的合并语义
// ---------------------------------------------------------------------------

/// 把一串 handler verdict 按声明顺序应用到同一个 step 上，合并出最终 [`HookControl`]。
///
/// 这是 step 层面的 pipeline 语义（对齐现有 `merge_outcome`）：
/// - **数据轴累积**：每个 verdict 的字段改动（改 args / 注入 / 填 result…）依次落到同一个 `&mut step`，
///   后一个 handler 看到的是前者改写后的状态。
/// - **控制轴早退**：任一 verdict 给出非 [`HookControl::Proceed`] 的指示即**停止 pipeline**并返回它
///   ——`Break` / `Continue` / `Skip` 都意味着"走向已定"，后续 handler 不应再覆盖。
/// - **错误处理**：某个 verdict 解析失败时由 `on_error` 决定降级（返回 `Some(control)` 早退 / `None`
///   跳过该 verdict 继续）——把"允许 block 的事件错误等价 block"这类策略留给调用方，本函数不写死。
pub fn run_step_pipeline<S, I, F>(
    step: &mut S,
    verdicts: I,
    mut on_error: F,
) -> HookControl
where
    S: HookStep + ?Sized,
    I: IntoIterator<Item = Value>,
    F: FnMut(VerdictError) -> Option<HookControl>,
{
    for verdict in verdicts {
        match step.apply_verdict(&verdict) {
            Ok(HookControl::Proceed) => {}
            Ok(control) => return control,
            Err(err) => {
                if let Some(control) = on_error(err) {
                    return control;
                }
            }
        }
    }
    HookControl::Proceed
}

/// 解析 verdict 里的 ContentBlock 数组（字符串数组 → 文本块）。
fn parse_block_array(v: &Value, field: &'static str) -> Result<Vec<ContentBlock>, VerdictError> {
    match v {
        Value::Array(items) => items
            .iter()
            .map(|item| {
                item.as_str().map(ContentBlock::from).ok_or(VerdictError::Malformed {
                    field,
                    reason: "each entry must be a string".to_string(),
                })
            })
            .collect(),
        _ => Err(VerdictError::Malformed {
            field,
            reason: "must be an array of strings".to_string(),
        }),
    }
}

#[cfg(test)]
#[path = "step/test.rs"]
mod test;
