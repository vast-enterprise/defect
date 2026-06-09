use super::*;

use std::pin::Pin;

use futures::Stream;
use futures::future::BoxFuture;
use serde_json::json;

use crate::tool::{
    SafetyClass, Tool, ToolCallDescription, ToolContext, ToolEvent, ToolSchema, ToolStream,
};

struct StubTool {
    schema: ToolSchema,
}

impl StubTool {
    fn new(name: &str) -> Self {
        Self {
            schema: ToolSchema {
                name: name.to_string(),
                description: format!("stub {name}"),
                input_schema: json!({"type": "object"}),
            },
        }
    }
}

impl Tool for StubTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }

    fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
        SafetyClass::ReadOnly
    }

    fn describe<'a>(
        &'a self,
        _args: &'a serde_json::Value,
        _ctx: ToolContext<'a>,
    ) -> BoxFuture<'a, ToolCallDescription> {
        Box::pin(async {
            ToolCallDescription {
                fields: Default::default(),
            }
        })
    }

    fn execute(&self, _args: serde_json::Value, _ctx: ToolContext<'_>) -> ToolStream {
        let stream: Pin<Box<dyn Stream<Item = ToolEvent> + Send>> =
            Box::pin(futures::stream::empty());
        stream
    }
}

#[test]
fn static_registry_lookup() {
    let reg = StaticToolRegistry::builder()
        .insert(Arc::new(StubTool::new("foo")))
        .insert(Arc::new(StubTool::new("bar")))
        .build();
    assert_eq!(reg.schemas().len(), 2);
    assert!(reg.get("foo").is_some());
    assert!(reg.get("baz").is_none());
}

#[test]
fn composite_session_overrides_process() {
    let process = Arc::new(
        StaticToolRegistry::builder()
            .insert(Arc::new(StubTool::new("fs")))
            .insert(Arc::new(StubTool::new("grep")))
            .build(),
    ) as Arc<dyn ToolRegistry>;
    let session = Arc::new(
        StaticToolRegistry::builder()
            .insert(Arc::new(StubTool::new("fs"))) // overrides the process registry's "fs" entry
            .insert(Arc::new(StubTool::new("mcp__linear")))
            .build(),
    ) as Arc<dyn ToolRegistry>;

    let comp = CompositeRegistry::new(session, process);
    let names: Vec<String> = comp.schemas().into_iter().map(|s| s.name).collect();
    // session first; duplicate `fs` from process is removed
    assert_eq!(names, vec!["fs", "mcp__linear", "grep"]);
    assert!(comp.get("fs").is_some());
    assert!(comp.get("grep").is_some());
    assert!(comp.get("mcp__linear").is_some());
    assert!(comp.get("nope").is_none());
}

fn pool(names: &[&str]) -> Arc<dyn ToolRegistry> {
    let mut b = StaticToolRegistry::builder();
    for n in names {
        b = b.insert(Arc::new(StubTool::new(n)));
    }
    Arc::new(b.build())
}

#[test]
fn allowlist_exact_names_still_work() {
    let p = pool(&["read_file", "search", "bash"]);
    let m = match_tool_allowlist(&p, &["read_file".into(), "search".into()]).expect("match");
    assert_eq!(m.tools, vec!["read_file", "search"]);
    assert!(!m.spawn_agent);
}

#[test]
fn allowlist_glob_matches_mcp_server_prefix() {
    let p = pool(&[
        "read_file",
        "mcp__ange__validate",
        "mcp__ange__format",
        "mcp__other__x",
    ]);
    let m = match_tool_allowlist(&p, &["mcp__ange__*".into()]).expect("match");
    // Pool order preserved; only the ange server's tools.
    assert_eq!(m.tools, vec!["mcp__ange__validate", "mcp__ange__format"]);
}

#[test]
fn allowlist_star_matches_everything_and_spawn_agent() {
    let p = pool(&["read_file", "mcp__ange__validate"]);
    let m = match_tool_allowlist(&p, &["*".into()]).expect("match");
    assert_eq!(m.tools, vec!["read_file", "mcp__ange__validate"]);
    // `*` matches the virtual spawn_agent member too.
    assert!(m.spawn_agent);
}

#[test]
fn allowlist_pattern_matching_nothing_is_error() {
    let p = pool(&["read_file", "mcp__ange__validate"]);
    let err = match_tool_allowlist(&p, &["mcp__nope__*".into()]).expect_err("should error");
    assert_eq!(err, "mcp__nope__*");
}

#[test]
fn allowlist_invalid_glob_is_error() {
    let p = pool(&["read_file"]);
    let err = match_tool_allowlist(&p, &["[bad".into()]).expect_err("should error");
    assert!(err.contains("invalid tool pattern"), "{err}");
}

#[test]
fn allowlist_explicit_spawn_agent_sets_flag_not_tool() {
    let p = pool(&["read_file"]);
    let m = match_tool_allowlist(&p, &["read_file".into(), "spawn_agent".into()]).expect("match");
    assert_eq!(m.tools, vec!["read_file"]); // spawn_agent never a real pool tool
    assert!(m.spawn_agent);
}

#[test]
fn allowlist_dedups_overlapping_patterns() {
    let p = pool(&["mcp__ange__validate", "mcp__ange__format"]);
    let m = match_tool_allowlist(&p, &["mcp__ange__*".into(), "mcp__ange__validate".into()])
        .expect("match");
    assert_eq!(m.tools, vec!["mcp__ange__validate", "mcp__ange__format"]);
}

#[test]
fn filter_registry_builds_subset() {
    let p = pool(&["read_file", "search", "bash"]);
    let filtered =
        filter_registry_by_allowlist(&p, &["read_file".into(), "search".into()]).expect("filter");
    let names: Vec<String> = filtered.schemas().into_iter().map(|s| s.name).collect();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"read_file".to_string()));
    assert!(!names.contains(&"bash".to_string()));
}
