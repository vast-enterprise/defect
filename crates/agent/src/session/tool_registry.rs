//! [`ToolRegistry`] implementations: static registry + composite query.
//!
//! - [`StaticToolRegistry`]: process-level (builtin tools) or session-level (MCP tools),
//!   immutable after construction
//! - [`CompositeRegistry`]: chains two registries so the main loop sees a unified
//!   interface (`get` checks session-level first, then process-level; schemas concatenate
//!   both)

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::session::ToolRegistry;
use crate::tool::{Tool, ToolSchema};

/// An immutable mapping from names to tools.
///
/// Construct via [`StaticToolRegistry::builder`]; once built, no tools can be added or
/// removed, ensuring that the schema order and `get` behavior remain stable.
pub struct StaticToolRegistry {
    schemas: Vec<ToolSchema>,
    by_name: HashMap<String, Arc<dyn Tool>>,
}

impl StaticToolRegistry {
    pub fn builder() -> StaticToolRegistryBuilder {
        StaticToolRegistryBuilder::default()
    }

    /// An empty registry, useful for testing or as a placeholder.
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

/// A builder for [`StaticToolRegistry`].
#[derive(Default)]
pub struct StaticToolRegistryBuilder {
    schemas: Vec<ToolSchema>,
    by_name: HashMap<String, Arc<dyn Tool>>,
}

impl StaticToolRegistryBuilder {
    /// Registers a tool. If a tool with the same name already exists, it is overwritten
    /// and the old entry in `schemas` is replaced with the new one (keeping `schemas`
    /// order stable for diagnostics).
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

/// A "composite" view of the process-level and session-level registries.
///
/// Lookup semantics: session-level (per-session MCP) first, then process-level
/// (built-in). This allows session-level MCP tools to "shadow" built-in tools with the
/// same name — a common convention in the MCP model.
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
        // Session-level override: remove schemas from `process` that are already declared
        // in `session` with the same name.
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

/// Restricts a `base` registry to the subset named in `allow`, producing a new static
/// registry. Used to enforce a profile's tool allowlist **after** the full session tool
/// pool (built-in + MCP) is assembled — applying it earlier (against a static, MCP-free
/// pool) would reject `mcp__*` tools that have not yet been connected.
///
/// `spawn_agent` is special-cased by the callers (it is injected/excluded based on the
/// recursion depth, not the allowlist), so any `spawn_agent` entry in `allow` is ignored
/// here and the caller decides whether to re-add it.
///
/// # Errors
/// Returns `Err(name)` for the first allowlisted name that is absent from `base`
/// (fail-loud: a profile that allows a non-existent tool is a configuration error).
pub fn filter_registry_by_allowlist(
    base: &Arc<dyn ToolRegistry>,
    allow: &[String],
    skip: &str,
) -> Result<Arc<dyn ToolRegistry>, String> {
    let mut builder = StaticToolRegistry::builder();
    for name in allow {
        if name == skip {
            continue;
        }
        match base.get(name) {
            Some(tool) => builder = builder.insert(tool),
            None => return Err(name.clone()),
        }
    }
    Ok(Arc::new(builder.build()))
}

#[cfg(test)]
mod tests;
