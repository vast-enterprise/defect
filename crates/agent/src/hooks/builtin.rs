//! Builtin hook handlers.
//!
//! 进程内 Rust handler——零外部依赖，CLI 装配时按 [`BuiltinRegistry`] 按名查表
//! 实例化，挂进 `DefaultHookEngine` 的 [`super::HandlerTable`]。
//!
//! 详见 `docs/internal/hooks.md` §4.1 / §10。

use std::collections::BTreeMap;
use std::sync::Arc;

use agent_client_protocol::schema::ContentBlock;
use futures::future::BoxFuture;
use serde_json::{Map, Value};

use super::{HookCapability, HookCtx, HookError, HookEvent, HookHandler, HookOutcome, HookPatch};
use crate::tool::SkillEntry;

/// Builtin handler 的注册表：name → 工厂闭包。
///
/// CLI 装配 `DefaultHookEngine` 时把 `HookHandlerSpec::Builtin { name }` 喂给
/// [`Self::lookup`]，配置加载期未知名直接 fail-fast——避免用户在 turn 跑到
/// 一半才发现拼错（见 hooks.md §4.1）。
///
/// 工厂签名是 `Fn() -> Arc<dyn HookHandler>`：handler 没有 per-config 参数，
/// 多个 `[[hooks.*]]` 引用同名 builtin 共享同一份 `Arc`。后续若有 builtin 需要
/// 配置参数，再把 `name` 升级成结构化 enum，registry 改成 `match` 分发。
pub struct BuiltinRegistry {
    factories: BTreeMap<String, Box<dyn Fn() -> Arc<dyn HookHandler> + Send + Sync>>,
}

impl BuiltinRegistry {
    /// v0 默认 registry：`tracing-audit` + `redact-secrets`。
    pub fn defaults() -> Self {
        let mut reg = Self {
            factories: BTreeMap::new(),
        };
        reg.register("tracing-audit", || Arc::new(TracingAuditHook));
        reg.register("redact-secrets", || Arc::new(RedactSecretsHook));
        reg
    }

    /// 注册一条 builtin。重复 name 直接覆盖——测试用例可以 stub 出测试 builtin
    /// 替换默认行为。
    pub fn register<F>(&mut self, name: &str, factory: F)
    where
        F: Fn() -> Arc<dyn HookHandler> + Send + Sync + 'static,
    {
        self.factories.insert(name.to_string(), Box::new(factory));
    }

    /// 按名查 handler。`None` = 配置层应当 fail-fast 报错。
    pub fn lookup(&self, name: &str) -> Option<Arc<dyn HookHandler>> {
        self.factories.get(name).map(|f| f())
    }

    /// 列出已注册的 builtin name——`defect hooks list` CLI 用。
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.factories.keys().map(String::as_str)
    }
}

impl Default for BuiltinRegistry {
    fn default() -> Self {
        Self::defaults()
    }
}

// ---------------------------------------------------------------------------
// tracing-audit
// ---------------------------------------------------------------------------

/// 把 `Post*ToolUse` 事件转成结构化 tracing 记录。
///
/// 适合挂在 `[[hooks.post_tool_use]]` / `[[hooks.post_tool_use_failure]]` 上做
/// 审计 trail；其他事件上挂会被 [`HookHandler::handle`] 直接 `Pass`。
pub struct TracingAuditHook;

impl HookHandler for TracingAuditHook {
    fn capability(&self) -> HookCapability {
        HookCapability::Intercept
    }

    fn handle<'a>(
        &'a self,
        event: &'a HookEvent<'a>,
        _ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<HookOutcome, HookError>> {
        Box::pin(async move {
            match event {
                HookEvent::PostToolUse { id, name, .. } => {
                    tracing::info!(
                        target: "defect_agent::hooks::audit",
                        tool = %name,
                        tool_call_id = %id.0,
                        outcome = "ok",
                        "tool call completed",
                    );
                }
                HookEvent::PostToolUseFailure { id, name, error } => {
                    tracing::info!(
                        target: "defect_agent::hooks::audit",
                        tool = %name,
                        tool_call_id = %id.0,
                        outcome = "error",
                        error = %error,
                        "tool call failed",
                    );
                }
                _ => {
                    // 其他事件挂这条 builtin 不报错，仅 silent pass——hook 配置
                    // 写错也别炸。
                }
            }
            Ok(HookOutcome::default())
        })
    }
}

// ---------------------------------------------------------------------------
// redact-secrets
// ---------------------------------------------------------------------------

/// `PreToolUse` 上对 args 里的疑似敏感字段做就地替换。
///
/// 命中名（不区分大小写包含子串）：`password` / `secret` / `token` / `api_key`
/// / `apikey` / `authorization`。命中后该字段值被替换为 `"***"`，patch 进 args。
///
/// 仅在 args 是 `Object` 时操作；其他形态（数组、字符串）不动——args 形态由
/// 工具自身定义，深度递归改写有可能破坏工具语义。
///
/// 不处理 `bash` 的 `command` 字符串里嵌入的 `password=xxx` 这类——那需要
/// shell 词法分析，超出 builtin 的稳定承诺。
pub struct RedactSecretsHook;

const SECRET_KEY_NEEDLES: &[&str] = &[
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
];

impl HookHandler for RedactSecretsHook {
    fn capability(&self) -> HookCapability {
        HookCapability::Intercept
    }

    fn handle<'a>(
        &'a self,
        event: &'a HookEvent<'a>,
        _ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<HookOutcome, HookError>> {
        let HookEvent::PreToolUse { args, .. } = event else {
            return Box::pin(async { Ok(HookOutcome::default()) });
        };
        let Some(obj) = args.as_object() else {
            return Box::pin(async { Ok(HookOutcome::default()) });
        };
        let redacted = redact_object(obj);
        Box::pin(async move {
            if redacted.changed {
                Ok(HookOutcome {
                    patch: Some(HookPatch::ToolArgs(Value::Object(redacted.value))),
                    ..Default::default()
                })
            } else {
                Ok(HookOutcome::default())
            }
        })
    }
}

struct Redacted {
    value: Map<String, Value>,
    changed: bool,
}

fn redact_object(obj: &Map<String, Value>) -> Redacted {
    let mut out = Map::with_capacity(obj.len());
    let mut changed = false;
    for (key, value) in obj {
        if key_is_secret(key) {
            out.insert(key.clone(), Value::String("***".to_string()));
            changed = true;
        } else {
            out.insert(key.clone(), value.clone());
        }
    }
    Redacted {
        value: out,
        changed,
    }
}

fn key_is_secret(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    SECRET_KEY_NEEDLES
        .iter()
        .any(|needle| lower.contains(needle))
}

// ---------------------------------------------------------------------------
// skill-manifest
// ---------------------------------------------------------------------------

/// `SessionStart` 上把可用 skill 的 L1 清单（`name + description`）拼进 system
/// prompt suffix——让模型一开机就知道有哪些 skill 可按需用 `skill` 工具加载。
///
/// 这是 progressive disclosure 的 L1 注入点（设计见 `docs/internal/skills.md`
/// §6.1）。注意 `skill` 工具自身的 description 已经内嵌同一份 catalog（见
/// [`crate::tool::SkillTool`]），所以本 hook 是**可选增强**：装配方挂上它能让
/// 清单同时出现在 system prompt 里（对不把 tool description 计入注意力预算的
/// 客户端更稳）。两条路径同源（同一个 skill 索引），不会发散。
///
/// 与其它 builtin 不同，本 handler 持有 skill 索引，**不能**用
/// [`BuiltinRegistry::defaults`] 的无参工厂构造——CLI 装配期用捕获索引的闭包
/// 注册（见 `defect_cli::hooks`）。
pub struct SkillManifestHook {
    skills: Arc<BTreeMap<String, SkillEntry>>,
}

impl SkillManifestHook {
    /// 用已加载的 skill 索引构造。`skills` 为空时调用方**不应**注册本 hook
    /// （清单会是空段，徒增 token）。
    pub fn new(skills: Arc<BTreeMap<String, SkillEntry>>) -> Self {
        Self { skills }
    }
}

/// 渲染 L1 清单文本。空索引返回 `None`（不注入空段）。
fn render_skill_manifest(skills: &BTreeMap<String, SkillEntry>) -> Option<String> {
    if skills.is_empty() {
        return None;
    }
    let mut out = String::from(
        "## Available Skills\n\n\
         Load a skill's full instructions with the `skill` tool (by name) when the task matches:\n",
    );
    for (name, entry) in skills {
        out.push_str(&format!("- **{name}**: {}\n", entry.description));
    }
    Some(out)
}

impl HookHandler for SkillManifestHook {
    fn capability(&self) -> HookCapability {
        HookCapability::Intercept
    }

    fn handle<'a>(
        &'a self,
        event: &'a HookEvent<'a>,
        _ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<HookOutcome, HookError>> {
        Box::pin(async move {
            let HookEvent::SessionStart { .. } = event else {
                // 其他事件挂这条 builtin 不报错，仅 silent pass。
                return Ok(HookOutcome::default());
            };
            let Some(manifest) = render_skill_manifest(&self.skills) else {
                return Ok(HookOutcome::default());
            };
            Ok(HookOutcome {
                append: vec![ContentBlock::from(manifest)],
                ..Default::default()
            })
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::hooks::{HookEvent, SessionSource};
    use crate::tool::SafetyClass;
    use agent_client_protocol::schema::{ToolCallId, ToolCallUpdateFields};
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    fn ctx<'a>(
        session_id: &'a agent_client_protocol::schema::SessionId,
        cwd: &'a std::path::Path,
    ) -> HookCtx<'a> {
        HookCtx::new(session_id, cwd, CancellationToken::new())
    }

    #[test]
    fn registry_defaults_have_two_builtins() {
        let reg = BuiltinRegistry::defaults();
        let names: Vec<_> = reg.names().collect();
        assert!(names.contains(&"tracing-audit"));
        assert!(names.contains(&"redact-secrets"));
    }

    #[test]
    fn registry_lookup_unknown_returns_none() {
        let reg = BuiltinRegistry::defaults();
        assert!(reg.lookup("does-not-exist").is_none());
    }

    #[tokio::test]
    async fn tracing_audit_passes_through() {
        let h = TracingAuditHook;
        let session_id = agent_client_protocol::schema::SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let id = ToolCallId::new("c1");
        let fields = ToolCallUpdateFields::default();
        let ev = HookEvent::PostToolUse {
            id: &id,
            name: "bash",
            fields: &fields,
        };
        let outcome = h.handle(&ev, ctx(&session_id, cwd)).await.expect("ok");
        assert!(outcome.block.is_none());
        assert!(outcome.patch.is_none());
        assert!(outcome.append.is_empty());
    }

    #[tokio::test]
    async fn tracing_audit_silently_passes_other_events() {
        let h = TracingAuditHook;
        let session_id = agent_client_protocol::schema::SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let ev = HookEvent::SessionStart {
            source: SessionSource::New,
            cwd,
        };
        let outcome = h.handle(&ev, ctx(&session_id, cwd)).await.expect("ok");
        assert!(outcome.block.is_none());
    }

    #[tokio::test]
    async fn redact_replaces_password_field() {
        let h = RedactSecretsHook;
        let session_id = agent_client_protocol::schema::SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let id = ToolCallId::new("c1");
        let args = serde_json::json!({"password": "hunter2", "user": "alice"});
        let ev = HookEvent::PreToolUse {
            id: &id,
            name: "login",
            args: &args,
            safety: SafetyClass::Network,
        };
        let outcome = h.handle(&ev, ctx(&session_id, cwd)).await.expect("ok");
        let Some(HookPatch::ToolArgs(value)) = outcome.patch else {
            panic!("expected ToolArgs patch, got {:?}", outcome.patch);
        };
        assert_eq!(value["password"], Value::String("***".to_string()));
        assert_eq!(value["user"], Value::String("alice".to_string()));
    }

    #[tokio::test]
    async fn redact_no_op_when_no_secret_key() {
        let h = RedactSecretsHook;
        let session_id = agent_client_protocol::schema::SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let id = ToolCallId::new("c1");
        let args = serde_json::json!({"command": "echo hi"});
        let ev = HookEvent::PreToolUse {
            id: &id,
            name: "bash",
            args: &args,
            safety: SafetyClass::Destructive,
        };
        let outcome = h.handle(&ev, ctx(&session_id, cwd)).await.expect("ok");
        assert!(outcome.patch.is_none());
    }

    #[test]
    fn key_is_secret_matches_common_variants() {
        assert!(key_is_secret("password"));
        assert!(key_is_secret("Password"));
        assert!(key_is_secret("API_KEY"));
        assert!(key_is_secret("auth_token"));
        assert!(key_is_secret("authorization"));
        assert!(!key_is_secret("user"));
        assert!(!key_is_secret("command"));
    }

    fn skill_entry(description: &str) -> SkillEntry {
        SkillEntry {
            description: description.to_string(),
            body: "body".to_string(),
            dir: std::path::PathBuf::from("/skills/x"),
        }
    }

    #[tokio::test]
    async fn skill_manifest_injects_catalog_on_session_start() {
        let mut skills = BTreeMap::new();
        skills.insert("code-review".to_string(), skill_entry("review Rust diffs"));
        let h = SkillManifestHook::new(Arc::new(skills));
        let session_id = agent_client_protocol::schema::SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let ev = HookEvent::SessionStart {
            source: SessionSource::New,
            cwd,
        };
        let outcome = h.handle(&ev, ctx(&session_id, cwd)).await.expect("ok");
        assert_eq!(outcome.append.len(), 1);
        let ContentBlock::Text(t) = &outcome.append[0] else {
            panic!("expected text block");
        };
        assert!(t.text.contains("Available Skills"));
        assert!(t.text.contains("- **code-review**: review Rust diffs"));
    }

    #[tokio::test]
    async fn skill_manifest_empty_index_injects_nothing() {
        let h = SkillManifestHook::new(Arc::new(BTreeMap::new()));
        let session_id = agent_client_protocol::schema::SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let ev = HookEvent::SessionStart {
            source: SessionSource::New,
            cwd,
        };
        let outcome = h.handle(&ev, ctx(&session_id, cwd)).await.expect("ok");
        assert!(outcome.append.is_empty());
    }

    #[tokio::test]
    async fn skill_manifest_passes_other_events() {
        let mut skills = BTreeMap::new();
        skills.insert("x".to_string(), skill_entry("d"));
        let h = SkillManifestHook::new(Arc::new(skills));
        let session_id = agent_client_protocol::schema::SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let ev = HookEvent::UserPromptSubmit { content: &[] };
        let outcome = h.handle(&ev, ctx(&session_id, cwd)).await.expect("ok");
        assert!(outcome.append.is_empty());
    }

    // Make sure the unused-import warning doesn't fire on Arc when Arc isn't used.
    fn _arc_used() {
        let _: Arc<dyn HookHandler> = Arc::new(TracingAuditHook);
    }
}
