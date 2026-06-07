//! [`ToolRegistry`] implementations: static registry + composite query.
//!
//! - [`StaticToolRegistry`]: process-level (builtin tools) or session-level (MCP tools),
//!   immutable after construction
//! - [`CompositeRegistry`]: chains two registries so the main loop sees a unified interface
//!   (`get` checks session-level first, then process-level; schemas concatenate both)

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::session::ToolRegistry;
use crate::tool::{Tool, ToolSchema};

/// 名 → 工具的不可变映射。
///
/// 用 [`StaticToolRegistry::builder`] 构造；构造后不可增删，确保
/// schemas 顺序与 `get` 行为稳定。
pub struct StaticToolRegistry {
    schemas: Vec<ToolSchema>,
    by_name: HashMap<String, Arc<dyn Tool>>,
}

impl StaticToolRegistry {
    pub fn builder() -> StaticToolRegistryBuilder {
        StaticToolRegistryBuilder::default()
    }

    /// 空注册表。便于测试 / placeholder。
    pub fn empty() -> Self {
        Self {
            schemas: Vec::new(),
            by_name: HashMap::new(),
        }
    }
}

impl ToolRegistry for StaticToolRegistry {
    fn schemas(&self) -> Vec<ToolSchema> {
        self.schemas.clone()
    }

    fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.by_name.get(name).cloned()
    }
}

/// [`StaticToolRegistry`] 的构造器。
#[derive(Default)]
pub struct StaticToolRegistryBuilder {
    schemas: Vec<ToolSchema>,
    by_name: HashMap<String, Arc<dyn Tool>>,
}

impl StaticToolRegistryBuilder {
    /// 注册一个工具。重名时覆盖，并把 schemas 中旧条目替换为新的（保持
    /// schemas 顺序稳定，便于诊断）。
    pub fn insert(mut self, tool: Arc<dyn Tool>) -> Self {
        let schema = tool.schema().clone();
        if let Some(pos) = self.schemas.iter().position(|s| s.name == schema.name) {
            if let Some(slot) = self.schemas.get_mut(pos) {
                *slot = schema.clone();
            }
        } else {
            self.schemas.push(schema.clone());
        }
        self.by_name.insert(schema.name, tool);
        self
    }

    pub fn build(self) -> StaticToolRegistry {
        StaticToolRegistry {
            schemas: self.schemas,
            by_name: self.by_name,
        }
    }
}

/// 进程级 + 会话级两层注册表的"复合"视图。
///
/// 查找语义：先会话级（per-session MCP），再进程级（内置）。这样会话级
/// 的 MCP 工具可以"覆盖"同名内置工具——这是 MCP 模型的常规约定。
pub struct CompositeRegistry {
    session: Arc<dyn ToolRegistry>,
    process: Arc<dyn ToolRegistry>,
}

impl CompositeRegistry {
    pub fn new(session: Arc<dyn ToolRegistry>, process: Arc<dyn ToolRegistry>) -> Self {
        Self { session, process }
    }
}

impl ToolRegistry for CompositeRegistry {
    fn schemas(&self) -> Vec<ToolSchema> {
        let mut session_schemas = self.session.schemas();
        let mut process_schemas = self.process.schemas();
        // 会话级覆盖：从 process 中剔除 session 已经声明的同名 schema。
        let session_names: HashSet<&str> =
            session_schemas.iter().map(|s| s.name.as_str()).collect();
        process_schemas.retain(|s| !session_names.contains(s.name.as_str()));
        session_schemas.append(&mut process_schemas);
        session_schemas
    }

    fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.session.get(name).or_else(|| self.process.get(name))
    }
}

#[cfg(test)]
mod tests;
