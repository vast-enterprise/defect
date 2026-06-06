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

/// 渲染 session 启动注入：L1 清单（所有 skill 的 name+description）+ 每个
/// `always` skill 的完整 body（always-on，直接进 system prompt）。空索引返回
/// `None`（不注入空段）。
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
    // always-on：把标了 `always: true` 的 skill body 直接拼进去——模型一开机
    // 就带着这些说明，无需再调 `skill` 工具加载（设计 §5.1）。
    for (name, entry) in skills {
        if entry.always {
            out.push_str(&format!("\n## Skill: {name}\n\n{}\n", entry.body));
        }
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

// ---------------------------------------------------------------------------
// skill-triggers
// ---------------------------------------------------------------------------

/// `before_ingest` 上按用户 prompt 自动激活相关 skill——命中即在 prompt 前面
/// 前插一条 **L1 提示**（"检测到 skill X 相关，需要时用 `skill` 工具加载"），
/// 而非整段 body（progressive disclosure：把"是否真加载"留给模型）。
///
/// 命中条件（任一即命中，设计见 `docs/internal/skills.md` §4.3）：
/// - **keyword**：skill 的 `triggers.keywords` 任一是 prompt 文本的大小写不敏感
///   子串；
/// - **glob**：从 prompt 文本里抽出的"路径样 token"任一被 `triggers.globs`
///   命中。
///
/// `always` skill 已在 session 启动整段注入，这里跳过——不重复提示。
///
/// 与 [`SkillManifestHook`] 一样持有 skill 索引，用捕获索引的闭包注册
/// （见 `defect_cli::hooks`）。
pub struct SkillTriggersHook {
    skills: Arc<BTreeMap<String, SkillEntry>>,
}

impl SkillTriggersHook {
    /// 用已加载的 skill 索引构造。`skills` 为空时调用方**不应**注册本 hook。
    pub fn new(skills: Arc<BTreeMap<String, SkillEntry>>) -> Self {
        Self { skills }
    }
}

/// 从 prompt 文本里抽"路径样 token"（best-effort，不做 NLP）。
///
/// 按空白切分，剥两侧引号 / 反引号 / 括号与尾部标点；token 满足任一即算路径：
/// (1) 含 `/`（如 `crates/agent/src/foo.rs`）；(2) 结尾是扩展名 `xxx.ext`
/// （如 `Cargo.toml` / `main.rs`）。剥前导 `./`。纯词（无 `/` 无扩展名）不算
/// 路径——交给 keyword 匹配。
fn extract_path_tokens(prompt: &str) -> Vec<String> {
    prompt
        .split_whitespace()
        .filter_map(|raw| {
            let trimmed = raw.trim_matches(|c: char| {
                c == '`' || c == '"' || c == '\'' || c == '(' || c == ')' || c == '[' || c == ']'
            });
            let trimmed = trimmed.trim_end_matches([',', '.', ':', ';']);
            let token = trimmed.strip_prefix("./").unwrap_or(trimmed);
            if token.is_empty() {
                return None;
            }
            if is_path_like(token) {
                Some(token.to_string())
            } else {
                None
            }
        })
        .collect()
}

/// token 是否"路径样"：含 `/`，或形如 `name.ext`（结尾点 + 1+ 字母数字）。
fn is_path_like(token: &str) -> bool {
    if token.contains('/') {
        return true;
    }
    // 结尾扩展名：最后一个 `.` 之后是 1+ 个字母数字，且点不在首位。
    match token.rsplit_once('.') {
        Some((stem, ext)) => {
            !stem.is_empty() && !ext.is_empty() && ext.chars().all(|c| c.is_ascii_alphanumeric())
        }
        None => false,
    }
}

/// 对单个 skill 判断是否被 prompt 激活：keyword 子串 OR glob 命中路径 token。
fn skill_triggered(entry: &SkillEntry, prompt_lower: &str, path_tokens: &[String]) -> bool {
    let keyword_hit = entry
        .triggers
        .keywords
        .iter()
        .any(|kw| !kw.is_empty() && prompt_lower.contains(&kw.to_ascii_lowercase()));
    if keyword_hit {
        return true;
    }
    match &entry.triggers.globs {
        Some(set) => path_tokens.iter().any(|t| set.is_match(t)),
        None => false,
    }
}

impl StepHandler for SkillTriggersHook {
    /// Step 模型：在 `before_ingest` 读 prompt 文本，命中的 skill 各前插一条 L1
    /// 提示（`prepend_input` verdict）。无命中返回 `None`。
    fn handle_step<'a>(
        &'a self,
        envelope: &'a Value,
        _ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<Option<Value>, HookError>> {
        let prompt = envelope.get("input").and_then(Value::as_str).unwrap_or("");
        let prompt_lower = prompt.to_ascii_lowercase();
        let path_tokens = extract_path_tokens(prompt);

        let hints: Vec<String> = self
            .skills
            .iter()
            .filter(|(_, e)| !e.always)
            .filter(|(_, e)| skill_triggered(e, &prompt_lower, &path_tokens))
            .map(|(name, _)| {
                format!(
                    "Detected skill `{name}` is relevant to the current task; \
                     load it with the `skill` tool when needed."
                )
            })
            .collect();

        let verdict = (!hints.is_empty()).then(|| serde_json::json!({ "prepend_input": hints }));
        Box::pin(async move { Ok(verdict) })
    }
}

// ---------------------------------------------------------------------------
// goal-gate
// ---------------------------------------------------------------------------

/// `--goal` 目标驱动循环的核心 hook，**同时挂两个事件**（据信封 `hook_event` 分流）：
///
/// - `after_session_enter`：把目标说明 + `goal_done` 使用契约作为 `additional_context`
///   注入 system prompt 后缀——**turn 1 就生效**。这样模型一开机就知道目标是什么、
///   完成后要主动调 `goal_done`，不必等第一次自愿停止才被告知（否则白白多耗一轮）。
/// - `before_turn_end`：turn 自愿停止时读 [`GoalState::is_reached`]：reached（模型调过
///   `goal_done`）→ `proceed` 放行结束；否则 → `continue` 续命 + 注入英文催促反馈。
///
/// 续命硬上限由 turn loop 的 [`crate::session::TurnConfig::max_hook_continues`] 兜底
/// （`--max-turns` 映射到它）——本 hook 只管"达成没"，不自己计数。
///
/// 与 [`SkillManifestHook`] 一样是有状态 builtin（持 `Arc<GoalState>`），不能用
/// [`BuiltinRegistry::defaults`] 的无参工厂构造——CLI 装配期按 `--goal` 用捕获
/// 状态的闭包注册到两个事件上（见 `defect_cli::hooks`）。
pub struct GoalGate {
    goal: Arc<crate::session::GoalState>,
}

impl GoalGate {
    pub fn new(goal: Arc<crate::session::GoalState>) -> Self {
        Self { goal }
    }

    /// turn 1 起注入 system prompt 的目标说明 + `goal_done` 契约。
    fn briefing(&self) -> String {
        format!(
            "## Goal\n\n\
             You are running in goal-driven mode. Your objective:\n\n{}\n\n\
             Work autonomously across as many turns as needed to achieve this goal. \
             When — and only when — the goal is genuinely and fully achieved, call the \
             `goal_done` tool to finish the run. Do not call it prematurely. If you stop \
             without calling `goal_done`, you will be prompted to keep working.",
            self.goal.objective()
        )
    }
}

impl StepHandler for GoalGate {
    /// Step 模型：按信封 `hook_event` 分流——
    /// - `after_session_enter` → 注入目标说明 + 契约（`additional_context`）；
    /// - `before_turn_end` → reached?proceed:continue+催促。
    fn handle_step<'a>(
        &'a self,
        envelope: &'a Value,
        _ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<Option<Value>, HookError>> {
        let event = envelope
            .get("hook_event")
            .and_then(Value::as_str)
            .unwrap_or("");
        let verdict = match event {
            "after_session_enter" => {
                serde_json::json!({ "additional_context": [self.briefing()] })
            }
            // before_turn_end（及兜底）：检测达成。
            _ if self.goal.is_reached() => serde_json::json!({ "control": "proceed" }),
            _ => serde_json::json!({
                "control": "continue",
                "additional_context": [format!(
                    "The goal \"{}\" is not yet complete. Keep working toward it. \
                     Once it is genuinely achieved, call the `goal_done` tool to finish.",
                    self.goal.objective()
                )],
            }),
        };
        Box::pin(async move { Ok(Some(verdict)) })
    }
}

#[cfg(test)]
mod tests {
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

    /// 造一个 skill：description/body/always/keywords/globs 可定制。
    fn skill(
        description: &str,
        body: &str,
        always: bool,
        keywords: &[&str],
        globs: &[&str],
    ) -> SkillEntry {
        let compiled = if globs.is_empty() {
            None
        } else {
            let mut b = globset::GlobSetBuilder::new();
            for g in globs {
                b.add(globset::Glob::new(g).expect("valid glob"));
            }
            Some(b.build().expect("glob set"))
        };
        SkillEntry {
            description: description.to_string(),
            body: body.to_string(),
            dir: std::path::PathBuf::from("/skills/x"),
            always,
            triggers: crate::tool::SkillTriggers {
                globs: compiled,
                keywords: keywords.iter().map(|s| s.to_string()).collect(),
            },
        }
    }

    #[tokio::test]
    async fn skill_manifest_step_injects_context() {
        let mut skills = BTreeMap::new();
        skills.insert(
            "deploy".to_string(),
            skill("deploy the app", "", false, &[], &[]),
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

    #[test]
    fn manifest_includes_always_on_body() {
        let mut skills = BTreeMap::new();
        skills.insert(
            "style".to_string(),
            skill("coding style", "ALWAYS USE TABS", true, &[], &[]),
        );
        skills.insert(
            "deploy".to_string(),
            skill("deploy", "deploy body", false, &[], &[]),
        );
        let out = render_skill_manifest(&skills).expect("some");
        // L1 清单含两者；always-on body 只拼 style 的。
        assert!(out.contains("**style**"));
        assert!(out.contains("**deploy**"));
        assert!(out.contains("ALWAYS USE TABS"));
        assert!(!out.contains("deploy body"));
    }

    fn triggers_envelope(prompt: &str) -> Value {
        serde_json::json!({ "source": "user", "input": prompt, "input_len": 1 })
    }

    #[tokio::test]
    async fn triggers_keyword_hit() {
        let mut skills = BTreeMap::new();
        skills.insert(
            "db".to_string(),
            skill("database", "", false, &["migration"], &[]),
        );
        let h = SkillTriggersHook::new(Arc::new(skills));
        let session_id = agent_client_protocol_schema::SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        // 大小写不敏感子串命中。
        let verdict = h
            .handle_step(
                &triggers_envelope("please run the MIGRATION now"),
                ctx(&session_id, cwd),
            )
            .await
            .expect("ok")
            .expect("verdict");
        let arr = verdict["prepend_input"].as_array().expect("array");
        assert_eq!(arr.len(), 1);
        assert!(arr[0].as_str().unwrap().contains("`db`"));
    }

    #[tokio::test]
    async fn triggers_glob_hit_on_path_token() {
        let mut skills = BTreeMap::new();
        skills.insert(
            "sql".to_string(),
            skill("sql files", "", false, &[], &["**/*.sql"]),
        );
        let h = SkillTriggersHook::new(Arc::new(skills));
        let session_id = agent_client_protocol_schema::SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let verdict = h
            .handle_step(
                &triggers_envelope("edit migrations/0001.sql to add a column"),
                ctx(&session_id, cwd),
            )
            .await
            .expect("ok")
            .expect("verdict");
        assert_eq!(verdict["prepend_input"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn triggers_no_hit_returns_none() {
        let mut skills = BTreeMap::new();
        skills.insert(
            "db".to_string(),
            skill("database", "", false, &["migration"], &["**/*.sql"]),
        );
        let h = SkillTriggersHook::new(Arc::new(skills));
        let session_id = agent_client_protocol_schema::SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let verdict = h
            .handle_step(
                &triggers_envelope("write some rust code"),
                ctx(&session_id, cwd),
            )
            .await
            .expect("ok");
        assert!(verdict.is_none());
    }

    #[tokio::test]
    async fn triggers_excludes_always_on_skill() {
        let mut skills = BTreeMap::new();
        // always 的 skill 即便 keyword 命中也不再提示（已整段注入）。
        skills.insert(
            "style".to_string(),
            skill("style", "body", true, &["rust"], &[]),
        );
        let h = SkillTriggersHook::new(Arc::new(skills));
        let session_id = agent_client_protocol_schema::SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let verdict = h
            .handle_step(&triggers_envelope("write rust"), ctx(&session_id, cwd))
            .await
            .expect("ok");
        assert!(verdict.is_none());
    }

    #[test]
    fn path_token_extraction() {
        let toks = extract_path_tokens("look at `crates/agent/src/foo.rs` and Cargo.toml please");
        assert!(toks.contains(&"crates/agent/src/foo.rs".to_string()));
        assert!(toks.contains(&"Cargo.toml".to_string()));
        // 纯词不算路径。
        assert!(!toks.contains(&"please".to_string()));
        assert!(!toks.contains(&"look".to_string()));
    }

    // ----- goal-gate -----

    #[tokio::test]
    async fn goal_gate_briefs_at_session_enter() {
        let goal = Arc::new(crate::session::GoalState::new("ship the feature"));
        let h = GoalGate::new(goal);
        let session_id = agent_client_protocol_schema::SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let envelope = serde_json::json!({ "hook_event": "after_session_enter" });
        let verdict = h
            .handle_step(&envelope, ctx(&session_id, cwd))
            .await
            .expect("ok")
            .expect("verdict");
        // 注入 system prompt 后缀，不带 control（不干预控制流）。
        assert!(verdict.get("control").is_none());
        let ctxs = verdict["additional_context"].as_array().expect("array");
        let briefing = ctxs[0].as_str().expect("str");
        assert!(briefing.contains("ship the feature"));
        assert!(briefing.contains("goal_done"));
    }

    #[tokio::test]
    async fn goal_gate_not_reached_continues_with_feedback() {
        let goal = Arc::new(crate::session::GoalState::new("ship the feature"));
        let h = GoalGate::new(goal);
        let session_id = agent_client_protocol_schema::SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let envelope = serde_json::json!({
            "hook_event": "before_turn_end",
            "stop_reason": "end_turn", "continues_so_far": 0, "voluntary": true,
        });
        let verdict = h
            .handle_step(&envelope, ctx(&session_id, cwd))
            .await
            .expect("ok")
            .expect("verdict");
        assert_eq!(verdict["control"], "continue");
        let ctxs = verdict["additional_context"].as_array().expect("array");
        assert_eq!(ctxs.len(), 1);
        assert!(
            ctxs[0]
                .as_str()
                .expect("str")
                .contains("ship the feature")
        );
    }

    #[tokio::test]
    async fn goal_gate_reached_proceeds() {
        let goal = Arc::new(crate::session::GoalState::new("ship the feature"));
        goal.mark_reached();
        let h = GoalGate::new(goal);
        let session_id = agent_client_protocol_schema::SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let envelope = serde_json::json!({
            "hook_event": "before_turn_end",
            "stop_reason": "end_turn", "continues_so_far": 1, "voluntary": true,
        });
        let verdict = h
            .handle_step(&envelope, ctx(&session_id, cwd))
            .await
            .expect("ok")
            .expect("verdict");
        assert_eq!(verdict["control"], "proceed");
    }
}
