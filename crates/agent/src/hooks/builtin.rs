//! Builtin hook handlers.
//!
//! 进程内 Rust handler——零外部依赖，CLI 装配时按 [`BuiltinRegistry`] 按名查表
//! 实例化，挂进 `DefaultHookEngine` 的 [`super::HandlerTable`]。
//!
//! 详见 `docs/internal/hooks.md` §4.1 / §10。

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::future::BoxFuture;
use serde_json::{Map, Value};

use super::{HookCtx, HookError, StepHandler};
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
    /// name → `Arc<dyn StepHandler>` 工厂。
    step_factories: BTreeMap<String, Box<dyn Fn() -> Arc<dyn StepHandler> + Send + Sync>>,
}

impl BuiltinRegistry {
    /// v0 默认 registry：`tracing-audit` + `redact-secrets`。
    pub fn defaults() -> Self {
        let mut reg = Self {
            step_factories: BTreeMap::new(),
        };
        reg.register_step("tracing-audit", || Arc::new(TracingAuditHook));
        reg.register_step("redact-secrets", || Arc::new(RedactSecretsHook));
        reg
    }

    /// 注册一条 builtin 的 step handler 工厂。重复 name 直接覆盖——测试可 stub 替换默认行为。
    pub fn register_step<F>(&mut self, name: &str, factory: F)
    where
        F: Fn() -> Arc<dyn StepHandler> + Send + Sync + 'static,
    {
        self.step_factories
            .insert(name.to_string(), Box::new(factory));
    }

    /// 按名查 step handler。`None` = 配置层应当 fail-fast 报错。
    pub fn lookup_step(&self, name: &str) -> Option<Arc<dyn StepHandler>> {
        self.step_factories.get(name).map(|f| f())
    }

    /// 列出已注册的 builtin name——`defect hooks list` CLI 用。
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.step_factories.keys().map(String::as_str)
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

impl StepHandler for TracingAuditHook {
    /// Step 模型：吃 `after_tool_apply` 信封 `{tool, is_error}`，记一条结构化审计日志，不产 verdict。
    fn handle_step<'a>(
        &'a self,
        envelope: &'a Value,
        _ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<Option<Value>, HookError>> {
        Box::pin(async move {
            let tool = envelope.get("tool").and_then(Value::as_str).unwrap_or("?");
            let is_error = envelope
                .get("is_error")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            tracing::info!(
                target: "defect_agent::hooks::audit",
                tool = %tool,
                outcome = if is_error { "error" } else { "ok" },
                "tool call completed",
            );
            Ok(None)
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

impl StepHandler for RedactSecretsHook {
    /// Step 模型：吃 `before_tool_apply` 信封 `{tool, args}`，对 args 里疑似敏感字段就地脱敏，
    /// 命中则返回 `{args: <redacted>}` verdict（引擎 apply 回 step → 改 args）。
    fn handle_step<'a>(
        &'a self,
        envelope: &'a Value,
        _ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<Option<Value>, HookError>> {
        let verdict = envelope
            .get("args")
            .and_then(Value::as_object)
            .map(redact_object)
            .filter(|r| r.changed)
            .map(|r| serde_json::json!({ "args": Value::Object(r.value) }));
        Box::pin(async move { Ok(verdict) })
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

impl StepHandler for SkillManifestHook {
    /// Step 模型：在 `after_session_enter` 把 L1 skill 清单作为 `additional_context` 注入
    /// （引擎 apply 回 step → 拼进 system prompt 后缀）。
    fn handle_step<'a>(
        &'a self,
        _envelope: &'a Value,
        _ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<Option<Value>, HookError>> {
        let verdict = render_skill_manifest(&self.skills)
            .map(|manifest| serde_json::json!({ "additional_context": [manifest] }));
        Box::pin(async move { Ok(verdict) })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    fn ctx<'a>(
        session_id: &'a agent_client_protocol_schema::SessionId,
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
        assert!(reg.lookup_step("does-not-exist").is_none());
    }

    #[test]
    fn registry_step_factories_match_event_factories() {
        let reg = BuiltinRegistry::defaults();
        assert!(reg.lookup_step("tracing-audit").is_some());
        assert!(reg.lookup_step("redact-secrets").is_some());
        assert!(reg.lookup_step("does-not-exist").is_none());
    }

    #[tokio::test]
    async fn redact_secrets_step_redacts_args() {
        let h = RedactSecretsHook;
        let session_id = agent_client_protocol_schema::SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let envelope = serde_json::json!({
            "tool": "login",
            "args": {"user": "alice", "password": "hunter2"},
        });
        let verdict = h
            .handle_step(&envelope, ctx(&session_id, cwd))
            .await
            .expect("ok")
            .expect("verdict");
        assert_eq!(verdict["args"]["password"], "***");
        assert_eq!(verdict["args"]["user"], "alice");
    }

    #[tokio::test]
    async fn redact_secrets_step_no_secrets_no_verdict() {
        let h = RedactSecretsHook;
        let session_id = agent_client_protocol_schema::SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let envelope = serde_json::json!({"tool": "ls", "args": {"path": "/tmp"}});
        let verdict = h
            .handle_step(&envelope, ctx(&session_id, cwd))
            .await
            .expect("ok");
        assert!(verdict.is_none());
    }

    #[tokio::test]
    async fn skill_manifest_step_injects_context() {
        let mut skills = BTreeMap::new();
        skills.insert(
            "deploy".to_string(),
            SkillEntry {
                description: "deploy the app".to_string(),
                body: String::new(),
                dir: std::path::PathBuf::from("/skills/deploy"),
            },
        );
        let h = SkillManifestHook::new(Arc::new(skills));
        let session_id = agent_client_protocol_schema::SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let envelope = serde_json::json!({"cwd": "/", "source": "new"});
        let verdict = h
            .handle_step(&envelope, ctx(&session_id, cwd))
            .await
            .expect("ok")
            .expect("verdict");
        let ctx_arr = verdict["additional_context"].as_array().expect("array");
        assert_eq!(ctx_arr.len(), 1);
        assert!(ctx_arr[0].as_str().unwrap().contains("deploy"));
    }

}
