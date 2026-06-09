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

/// The result of matching a profile tool allowlist against a tool pool.
#[derive(Debug)]
pub struct AllowlistMatch {
    /// Names of **real** pool tools matched, deduped and in pool order. Never contains
    /// `spawn_agent` (that is reported separately via [`Self::spawn_agent`]).
    pub tools: Vec<String>,
    /// Whether the virtual `spawn_agent` member matched any pattern. `spawn_agent` is
    /// never returned as a real pool tool because its actual availability is governed by
    /// the recursion **depth gate** (a child must get a fresh, depth-decremented instance,
    /// not the parent's). The caller decides whether to inject it based on this flag.
    pub spawn_agent: bool,
}

/// Match a profile's `allow` list against the names in `base` **plus** the virtual
/// `spawn_agent` member. Each entry in `allow` is a glob pattern (via [`globset`], the same
/// engine as hook `tool_glob` / skill triggers); a bare tool name is the degenerate case
/// of a glob with no wildcards, so exact allowlists keep working unchanged.
///
/// Applied **after** the full session tool pool (built-in + MCP) is assembled — matching
/// earlier against a static, MCP-free pool would drop `mcp__*` tools not yet connected.
///
/// # Errors
/// - An invalid glob pattern (returns the pattern text).
/// - A pattern that matches **nothing** — no real pool tool and not `spawn_agent`
///   (fail-loud: a profile that allows a tool/pattern matching nothing is a configuration
///   error, e.g. a misspelled server prefix). Returns the offending pattern text.
pub fn match_tool_allowlist(
    base: &Arc<dyn ToolRegistry>,
    allow: &[String],
) -> Result<AllowlistMatch, String> {
    let schemas = base.schemas();
    // Real candidates exclude `spawn_agent`; it is only a virtual member here.
    let pool_names: Vec<&str> = schemas
        .iter()
        .map(|s| s.name.as_str())
        .filter(|n| *n != crate::tool::SPAWN_AGENT_TOOL_NAME)
        .collect();

    let mut tools: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut spawn_agent = false;

    for pattern in allow {
        let matcher = globset::Glob::new(pattern)
            .map_err(|e| format!("invalid tool pattern `{pattern}`: {e}"))?
            .compile_matcher();
        let mut hit = false;
        for name in &pool_names {
            if matcher.is_match(name) {
                hit = true;
                if seen.insert((*name).to_string()) {
                    tools.push((*name).to_string());
                }
            }
        }
        if matcher.is_match(crate::tool::SPAWN_AGENT_TOOL_NAME) {
            hit = true;
            spawn_agent = true;
        }
        if !hit {
            return Err(pattern.clone());
        }
    }

    Ok(AllowlistMatch { tools, spawn_agent })
}

/// Restricts a `base` registry to the subset allowed by `allow` (glob patterns; see
/// [`match_tool_allowlist`]), producing a new static registry. Used by the top-level
/// `--profile` path, which is a leaf agent: a matched `spawn_agent` is intentionally
/// **dropped** (a top-level profile does not dispatch sub-agents).
///
/// # Errors
/// Propagates [`match_tool_allowlist`] errors (invalid glob / pattern matching nothing).
pub fn filter_registry_by_allowlist(
    base: &Arc<dyn ToolRegistry>,
    allow: &[String],
) -> Result<Arc<dyn ToolRegistry>, String> {
    let matched = match_tool_allowlist(base, allow)?;
    let mut builder = StaticToolRegistry::builder();
    for name in &matched.tools {
        if let Some(tool) = base.get(name) {
            builder = builder.insert(tool);
        }
    }
    Ok(Arc::new(builder.build()))
}

#[cfg(test)]
mod tests;
