//! Sandbox policy：工具调用的"放行 / 拒绝 / 询问用户"决策。
//!
//! 设计详见 `docs/internal/sandbox-policy.md`。
//!
//! ## 与主循环的接口
//!
//! [`SandboxPolicy::classify`] 是一次纯决策；返回 [`PolicyDecision`]：
//! - `Allow` / `Deny`：直接进入相应分支
//! - `Ask(Ask)`：主循环把 `Ask::options` 装进 ACP `RequestPermissionRequest`
//!   等用户回执，回执到达后调用 [`SandboxPolicy::record`] 让 policy 有机会
//!   更新内部"已授权"表
//!
//! ## 与 OS 级 sandbox 的边界
//!
//! 本模块**只做决策**——OS 级隔离（landlock / seatbelt / 子进程权限降级）
//! 是另一个 trait（未来的 `ToolSandbox`）。本模块的产出是"要不要执行"，
//! 与"执行时给多大权限"正交，参见 `docs/internal/sandbox-policy.md` §8。

use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, Mutex};

use agent_client_protocol_schema::{PermissionOptionId, PermissionOptionKind};
use serde::{Deserialize, Serialize};

use crate::tool::SafetyClass;

/// 决策结果。
///
/// `Ask::options` 必须由 policy 自己组装（含文案、wire id、`allows`）。
/// 主循环不再为 [`PermissionOptionKind::AllowOnce`] / `RejectOnce` 等
/// 推断"是不是放行"——那是 policy 的语义。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PolicyDecision {
    /// 直接放行，不打扰用户。
    Allow,
    /// 直接拒绝；主循环把"denied by policy"当作 tool_result 喂回 LLM。
    Deny,
    /// 需要用户确认。主循环触发 ACP `session/request_permission`，
    /// 等用户在 [`Ask::options`] 里选一项。
    Ask(Ask),
}

/// `Ask` 的选项装填载荷。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ask {
    /// 给客户端展示的选项列表。**空向量等价于 [`PolicyDecision::Deny`]**。
    pub options: Vec<AskOption>,
}

/// 一项给用户挑选的权限选项。
///
/// `kind` 是 ACP 的 UI 提示；`allows` 才是策略层面的"放行 / 拒绝"判定。
/// 二者通常一致（`AllowOnce` / `AllowAlways` → `allows = true`），但解耦
/// 让未来出现"AllowReadOnly"这类部分允许选项时不破坏现有形状。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AskOption {
    pub id: PermissionOptionId,
    pub name: String,
    pub kind: PermissionOptionKind,
    /// 用户选了这一项之后该执行（`true`）还是拒绝（`false`）。
    pub allows: bool,
}

/// 主循环回写给 policy 的"用户应答"。
///
/// `Selected::allows` 由 policy 在 [`SandboxPolicy::classify`] 阶段填进
/// [`AskOption`]；主循环按 `option_id` 查表后再回喂——避免 policy 二次
/// 解析自己刚发出去的选项 id。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordedOutcome {
    Selected {
        option_id: PermissionOptionId,
        allows: bool,
    },
    /// 用户取消了 turn。policy 不更新授权表，但可以做审计。
    Cancelled,
}

/// `classify` / `record` 共用的上下文。
#[non_exhaustive]
pub struct PolicyCtx<'a> {
    pub tool_name: &'a str,
    pub safety_hint: SafetyClass,
    pub args: &'a serde_json::Value,
    /// 当前 session 的工作目录。路径白名单策略要用；不需要的实现可以忽略。
    pub cwd: &'a Path,
}

impl<'a> PolicyCtx<'a> {
    pub fn new(
        tool_name: &'a str,
        safety_hint: SafetyClass,
        args: &'a serde_json::Value,
        cwd: &'a Path,
    ) -> Self {
        Self {
            tool_name,
            safety_hint,
            args,
            cwd,
        }
    }
}

/// 工具调用的决策器。
///
/// 实现要求纯函数式：`classify` 不做 IO、不持久化；持久化"已授权"表
/// 走 [`Self::record`]。
pub trait SandboxPolicy: Send + Sync {
    fn classify(&self, ctx: PolicyCtx<'_>) -> PolicyDecision;

    /// 用户应答 `Ask` 之后的回写钩子。
    ///
    /// 主循环在收到 [`crate::event::PermissionResolution::Selected`] 之后、
    /// 把工具入队执行 / 拒绝 *之前* 调用一次。`outcome.allows()` 已经从
    /// [`AskOption::allows`] 查好。
    fn record(&self, ctx: PolicyCtx<'_>, outcome: RecordedOutcome);
}

// ---------------------------------------------------------------------------
// 内置策略
// ---------------------------------------------------------------------------

/// 一切 `Allow`。等价 v0 早期 stub；测试 / dev mode 用。
pub struct OpenPolicy;

impl SandboxPolicy for OpenPolicy {
    fn classify(&self, _ctx: PolicyCtx<'_>) -> PolicyDecision {
        PolicyDecision::Allow
    }
    fn record(&self, _ctx: PolicyCtx<'_>, _outcome: RecordedOutcome) {}
}

/// 只放行 `ReadOnly`，其余一律 `Deny`。
pub struct ReadOnlyPolicy;

impl SandboxPolicy for ReadOnlyPolicy {
    fn classify(&self, ctx: PolicyCtx<'_>) -> PolicyDecision {
        match ctx.safety_hint {
            SafetyClass::ReadOnly => PolicyDecision::Allow,
            _ => PolicyDecision::Deny,
        }
    }
    fn record(&self, _ctx: PolicyCtx<'_>, _outcome: RecordedOutcome) {}
}

/// 一切 `Deny`。冒烟测试用。
pub struct DenyAllPolicy;

impl SandboxPolicy for DenyAllPolicy {
    fn classify(&self, _ctx: PolicyCtx<'_>) -> PolicyDecision {
        PolicyDecision::Deny
    }
    fn record(&self, _ctx: PolicyCtx<'_>, _outcome: RecordedOutcome) {}
}

/// 默认策略：`ReadOnly` 直接 `Allow`，`Mutating` / `Destructive` / `Network`
/// 走 `Ask`。`AllowAlways` 在内部维护一份 tool_name 白名单，命中即直接 Allow。
pub struct AskWritesPolicy {
    always_allow: Mutex<HashSet<String>>,
}

impl AskWritesPolicy {
    pub fn new() -> Self {
        Self {
            always_allow: Mutex::new(HashSet::new()),
        }
    }
}

impl Default for AskWritesPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl SandboxPolicy for AskWritesPolicy {
    fn classify(&self, ctx: PolicyCtx<'_>) -> PolicyDecision {
        if matches!(ctx.safety_hint, SafetyClass::ReadOnly) {
            return PolicyDecision::Allow;
        }
        if let Ok(table) = self.always_allow.lock()
            && table.contains(ctx.tool_name)
        {
            return PolicyDecision::Allow;
        }
        PolicyDecision::Ask(default_ask_options(ctx.tool_name))
    }

    fn record(&self, ctx: PolicyCtx<'_>, outcome: RecordedOutcome) {
        let RecordedOutcome::Selected { option_id, allows } = outcome else {
            return;
        };
        if !allows {
            return;
        }
        if option_id.0.as_ref() != ALLOW_ALWAYS_ID {
            return;
        }
        if let Ok(mut table) = self.always_allow.lock() {
            table.insert(ctx.tool_name.to_string());
        }
    }
}

/// 把任意内层策略适配成"非交互"语义：内层返回 [`PolicyDecision::Ask`]
/// 时一律降级为 [`PolicyDecision::Deny`]，`Allow` / `Deny` 原样透传。
///
/// 用于 subagent（`spawn_agent`）的嵌套 turn——子 agent 没有人在场回答
/// 权限请求，若放任 `Ask` 进入主循环会在 [`crate::session::permissions::PermissionGate`]
/// 上永久挂死。包一层本策略即可保证子 turn **永不阻塞、永不提权**：
/// 子 agent 的实际授权恒 ≤ 它包装的父策略（父会 Ask 的，子直接 Deny）。
///
/// 这与"工具白名单裁剪"是两道独立闸门：白名单决定子 agent **看得到**哪些
/// 工具，本策略决定在这些工具上**运行时**放行到什么程度。详见
/// `project-subagent-design` 设计记录。
pub struct NonInteractivePolicy {
    inner: Arc<dyn SandboxPolicy>,
}

impl NonInteractivePolicy {
    pub fn new(inner: Arc<dyn SandboxPolicy>) -> Self {
        Self { inner }
    }
}

impl SandboxPolicy for NonInteractivePolicy {
    fn classify(&self, ctx: PolicyCtx<'_>) -> PolicyDecision {
        match self.inner.classify(ctx) {
            PolicyDecision::Ask(_) => PolicyDecision::Deny,
            other => other,
        }
    }

    // 本策略永不返回 Ask，主循环也就不会回喂 record——空实现即可。
    fn record(&self, _ctx: PolicyCtx<'_>, _outcome: RecordedOutcome) {}
}

// ---------------------------------------------------------------------------
// permission mode 目录
// ---------------------------------------------------------------------------

/// 一个可被 ACP 客户端选择的权限模式条目。
///
/// `defect-agent` 不认识上层的 `SandboxMode`（那是 `defect-config` 的概念，
/// 本 crate 是它的依赖底座、不能反向依赖）。装配方（CLI）把每个
/// [`crate::policy::SandboxPolicy`] 连同一个稳定的 `id`、展示用 `name` /
/// `description` 一起塞进来；本 crate 只在不透明条目上做"按 id 查表换 active
/// policy"，与 ACP `session/set_mode` 的 `SessionMode` 一一对应。
#[derive(Clone)]
pub struct PolicyMode {
    /// 稳定标识——ACP wire 上的 `mode_id`。约定用 kebab-case（如
    /// `ask-writes`），与 `SandboxMode::as_str()` 对齐。
    pub id: String,
    /// 给客户端展示的人类可读名字。
    pub name: String,
    /// 可选描述，客户端 UI 展示用。
    pub description: Option<String>,
    /// 该模式对应的决策策略。`set_mode` 命中本条目时整体换成它。
    pub policy: Arc<dyn SandboxPolicy>,
}

impl std::fmt::Debug for PolicyMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyMode")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

/// 一组互斥的权限模式 + 当前选中项。映射到 ACP 的 `SessionModeState`。
///
/// 由装配方（CLI）一次性构造，随 [`crate::session::AgentCore`] 流入每个
/// session。session 持有自己的副本（`current` 可独立切换），`set_mode` 在其上
/// 按 id 查表换出对应 policy。
#[derive(Debug, Clone)]
pub struct ModeCatalog {
    modes: Vec<PolicyMode>,
    current: String,
}

impl ModeCatalog {
    /// 构造目录。`current` 必须命中 `modes` 中某个 `id`，否则返回 `None`
    /// （装配错误应 fail loud，不静默回落）。空目录也返回 `None`。
    #[must_use]
    pub fn new(modes: Vec<PolicyMode>, current: impl Into<String>) -> Option<Self> {
        let current = current.into();
        if modes.is_empty() || !modes.iter().any(|m| m.id == current) {
            return None;
        }
        Some(Self { modes, current })
    }

    /// 当前选中模式的 id。
    #[must_use]
    pub fn current_id(&self) -> &str {
        &self.current
    }

    /// 当前选中模式对应的 policy。
    #[must_use]
    pub fn current_policy(&self) -> Arc<dyn SandboxPolicy> {
        self.modes
            .iter()
            .find(|m| m.id == self.current)
            .map(|m| m.policy.clone())
            // 不变量：`current` 恒命中某条目（构造时校验，set 时也校验）。
            .expect("ModeCatalog current id must always resolve to a mode")
    }

    /// 全部可选模式（顺序即装配顺序）。
    #[must_use]
    pub fn modes(&self) -> &[PolicyMode] {
        &self.modes
    }

    /// 切换当前模式。`id` 未命中任一条目时返回 `false`，`current` 不变。
    pub fn set_current(&mut self, id: &str) -> bool {
        if self.modes.iter().any(|m| m.id == id) {
            self.current = id.to_string();
            true
        } else {
            false
        }
    }
}

const ALLOW_ONCE_ID: &str = "allow_once";
const ALLOW_ALWAYS_ID: &str = "allow_always";
const REJECT_ONCE_ID: &str = "reject_once";

/// 默认的 `Ask` 选项三件套：Allow once / Allow always / Reject once。
///
/// `RejectAlways` v0 不放——v0 没有"持久化拒绝"的需求；用户拒绝一次
/// 重新调起时还会再问。
fn default_ask_options(tool_name: &str) -> Ask {
    let options = vec![
        AskOption {
            id: PermissionOptionId::new(ALLOW_ONCE_ID),
            name: format!("Allow `{tool_name}` once"),
            kind: PermissionOptionKind::AllowOnce,
            allows: true,
        },
        AskOption {
            id: PermissionOptionId::new(ALLOW_ALWAYS_ID),
            name: format!("Allow `{tool_name}` always"),
            kind: PermissionOptionKind::AllowAlways,
            allows: true,
        },
        AskOption {
            id: PermissionOptionId::new(REJECT_ONCE_ID),
            name: "Reject".to_string(),
            kind: PermissionOptionKind::RejectOnce,
            allows: false,
        },
    ];
    Ask { options }
}

#[cfg(test)]
mod test;
