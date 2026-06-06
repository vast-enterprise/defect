//! `skill`：把一个 skill 的完整说明加载进当前对话。
//!
//! Skill 是用户配置的可复用提示片段——一段 markdown body 加上同目录下可选的
//! `scripts/` / `refs/` 资源。模型按 progressive disclosure 分三层看到 skill
//! 内容（设计见 `docs/internal/skills.md`）：
//!
//! - **L1 清单**：所有 skill 的 `name + description`。本工具把它编进自己的
//!   `description`（与 `spawn_agent` 把 profile catalog 编进 description 同款），
//!   模型一开机就看得到有哪些 skill 可用。可选地也由
//!   `crate::hooks::builtin::SkillManifestHook` 注入 system prompt。
//! - **L2 body**：模型调本工具按 `name` 拉取的 `SKILL.md` 全文，作为 tool result
//!   进入对话——之后模型**在当前对话里**按说明干活（区别于 `spawn_agent` 派生
//!   隔离子会话）。
//! - **L3 附件**：body 里引用的 `scripts/*.sh` / `refs/*.md`，模型用普通 `bash`
//!   / `read_file` 工具按需读——tool result 里回填了 skill 目录绝对路径供拼接。
//!
//! 本工具是纯 [`Tool`] 实现，`safety_hint = ReadOnly`（只查内存里已加载的 skill
//! 索引、不写盘、不出网），与其它内置工具一视同仁。

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use agent_client_protocol_schema::{
    Content, ContentBlock, TextContent, ToolCallContent, ToolCallUpdateFields, ToolKind,
};
use futures::future::BoxFuture;
use serde::Deserialize;
use serde_json::json;

use crate::error::BoxError;
use crate::tool::{
    SafetyClass, Tool, ToolCallDescription, ToolContext, ToolError, ToolEvent, ToolSchema,
    ToolStream,
};

/// `skill` 工具的名字。
pub(crate) const SKILL_TOOL_NAME: &str = "skill";

/// skill 的自动激活触发条件（Agent Skills open-standard 的 `triggers` 子表）。
///
/// 定义在 agent 侧、由 `defect-config` 在解析期填充并复用（依赖方向：config →
/// agent，不能反向）。`globs` 在配置解析期就编译成 [`globset::GlobSet`]——坏
/// glob 当场 fail-fast，运行期不再 parse；为空时为 `None`。匹配逻辑见
/// `crate::hooks::builtin::SkillTriggersHook`。
#[derive(Debug, Clone, Default)]
pub struct SkillTriggers {
    /// 编译好的文件路径 glob 集合；`None` = 未配置 globs。
    pub globs: Option<globset::GlobSet>,
    /// prompt 关键字（大小写不敏感 substring 匹配）。
    pub keywords: Vec<String>,
}

/// 一个可被 `skill` 工具加载的 skill（agent 侧表示）。
///
/// `defect-config` 的 `SkillSpec` 是配置侧真相源；CLI 装配时投影成本结构再交给
/// 工具。两边分开是因为 `defect-config` 依赖 `defect-agent`——不能反向依赖成环
/// （与 [`crate::tool::SubagentProfile`] / `ProfileSpec` 同款边界）。
#[derive(Debug, Clone)]
pub struct SkillEntry {
    /// 选择期描述，进 L1 清单（工具 description 的 catalog）。
    pub description: String,
    /// `SKILL.md` 去 frontmatter 后的 body 全文——L2 加载时回给模型。
    pub body: String,
    /// skill 目录绝对路径——L2 tool result 里回填，供模型拼 `scripts/` /
    /// `refs/` 等资源的绝对路径喂给 `bash` / `read_file`。
    pub dir: PathBuf,
    /// `always: true` ⇒ body 在 session 启动直接拼进 system prompt（always-on，
    /// 见 `crate::hooks::builtin::SkillManifestHook`）。
    pub always: bool,
    /// 自动激活触发条件（按文件 glob / prompt 关键字），见 [`SkillTriggers`]。
    pub triggers: SkillTriggers,
}

/// `skill` 工具。挂在 `StaticToolRegistry` 上、随 `process_tools` 被所属
/// `AgentCore` 的各 session 共享一份（**不是**进程全局单例——一个进程里可以装配
/// 多个 `AgentCore`，各持自己的 skill 索引）。
pub struct SkillTool {
    schema: ToolSchema,
    skills: Arc<BTreeMap<String, SkillEntry>>,
}

impl SkillTool {
    /// 构造一个 `skill` 工具。`skills` 为空时调用方**不应**注册本工具
    /// （schema 的 `name` enum 会是空集，永远调用失败）——见 [`Self::has_skills`]。
    pub fn new(skills: Arc<BTreeMap<String, SkillEntry>>) -> Self {
        let schema = build_schema(&skills);
        Self { schema, skills }
    }

    /// 是否发现到任何 skill。装配方据此决定是否注册本工具。
    pub fn has_skills(skills: &BTreeMap<String, SkillEntry>) -> bool {
        !skills.is_empty()
    }
}

/// 动态构造 schema：`name` 是发现到的 skill 名的 enum（硬约束），工具
/// description 内嵌 `- <name>: <description>` 的 catalog（软引导，即 L1 清单）。
/// 两者缺一不可：光 enum 模型不知用途，光 catalog 模型可能填错名（与
/// [`crate::tool::SpawnAgentTool`] 的 `build_schema` 同款理由）。
fn build_schema(skills: &BTreeMap<String, SkillEntry>) -> ToolSchema {
    let names: Vec<&str> = skills.keys().map(String::as_str).collect();
    let catalog = skills
        .iter()
        .map(|(name, s)| format!("- {name}: {}", s.description))
        .collect::<Vec<_>>()
        .join("\n");
    let description = format!(
        "Load the full instructions for a specialized skill into the current conversation. \
         Use this when the task at hand matches one of the skills below; the loaded content may \
         contain detailed workflow guidance plus references to scripts / files in the skill's \
         directory that you can then read with `bash` / `read_file`. After loading, carry out the \
         task in this same conversation.\n\n\
         Available skills:\n{catalog}"
    );
    ToolSchema {
        name: SKILL_TOOL_NAME.to_string(),
        description,
        input_schema: json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": names,
                    "description": "Which skill to load. See the tool description for what each skill does."
                }
            },
            "required": ["name"]
        }),
    }
}

#[derive(Debug, Deserialize)]
struct SkillArgs {
    name: String,
}

impl Tool for SkillTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }

    fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
        // 只查内存里已加载的 skill 索引、把 body 文本喂回模型——不写盘、不出网。
        SafetyClass::ReadOnly
    }

    fn describe<'a>(
        &'a self,
        args: &'a serde_json::Value,
        _ctx: ToolContext<'a>,
    ) -> BoxFuture<'a, ToolCallDescription> {
        Box::pin(async move {
            let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("?");
            let mut fields = ToolCallUpdateFields::default();
            fields.title = Some(format!("Load skill `{name}`"));
            fields.kind = Some(ToolKind::Think);
            ToolCallDescription { fields }
        })
    }

    fn execute(&self, args: serde_json::Value, _ctx: ToolContext<'_>) -> ToolStream {
        let skills = self.skills.clone();
        let fut = async move {
            let parsed: SkillArgs = match serde_json::from_value(args) {
                Ok(v) => v,
                Err(err) => return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(err))),
            };

            let Some(skill) = skills.get(&parsed.name) else {
                return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(io_err(format!(
                    "unknown skill `{}`; available: {}",
                    parsed.name,
                    skills.keys().cloned().collect::<Vec<_>>().join(", ")
                )))));
            };

            let output = render_skill(&parsed.name, skill);
            let mut fields = ToolCallUpdateFields::default();
            fields.content = Some(vec![ToolCallContent::Content(Content::new(
                ContentBlock::Text(TextContent::new(output.clone())),
            ))]);
            // raw_output 给遥测（langfuse projector 只读 raw_output 作 observation output）。
            fields.raw_output = Some(serde_json::Value::String(output));
            ToolEvent::Completed(fields)
        };
        let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> =
            Box::pin(futures::stream::once(fut));
        s
    }
}

/// 拼 L2 加载的 tool result 文本：标题 + 目录提示 + body。目录提示让模型知道
/// `scripts/` / `refs/` 等资源的绝对路径根（与 opencode 的 "Base directory"
/// 行同款）。
fn render_skill(name: &str, skill: &SkillEntry) -> String {
    format!(
        "# Skill: {name}\n\n{body}\n\n\
         Skill directory: {dir}\n\
         Relative paths in this skill (e.g. scripts/, refs/) are relative to that directory; \
         read them with `read_file` / `bash` as needed.",
        body = skill.body,
        dir = skill.dir.display(),
    )
}

fn io_err(msg: String) -> std::io::Error {
    std::io::Error::other(msg)
}

#[cfg(test)]
mod test;
