//! Tool schema — the wire-facing description of a tool's parameters.
//!
//! This is the minimal, self-contained piece of the tool layer that providers need:
//! `defect-llm` serializes [`ToolSchema`] into provider wire JSON. The full tool trait,
//! `ToolContext`, and runtime plumbing live in `defect-agent::tool` (they depend on the
//! session runtime and so cannot live here).

use serde::{Deserialize, Serialize};

/// A tool's wire-facing schema: name, description, and a JSON Schema for its parameters.
///
/// Providers don't hold `dyn Tool`; they serialize schemas into wire JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    /// JSON Schema for the parameters. Uses a subset of Draft 2020-12 (the exact subset
    /// and escaping rules are documented in `tool-trait.md`).
    pub input_schema: serde_json::Value,
}
