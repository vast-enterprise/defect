use super::*;

use std::path::{Path, PathBuf};

use agent_client_protocol_schema::ContentBlock;
use futures::StreamExt;

use crate::fs::{FsBackend, NoopFsBackend};
use crate::http::NoopHttpClient;
use crate::shell::{NoopShellBackend, ShellBackend};
use crate::tool::ToolContext;
use tokio_util::sync::CancellationToken;

fn skills_with(entries: &[(&str, &str, &str)]) -> Arc<BTreeMap<String, SkillEntry>> {
    let mut m = BTreeMap::new();
    for (name, description, body) in entries {
        m.insert(
            (*name).to_string(),
            SkillEntry {
                description: (*description).to_string(),
                body: (*body).to_string(),
                dir: PathBuf::from(format!("/skills/{name}")),
                always: false,
                triggers: crate::tool::SkillTriggers::default(),
            },
        );
    }
    Arc::new(m)
}

fn run_tool(tool: &SkillTool, args: serde_json::Value, cwd: &Path) -> Vec<ToolEvent> {
    let fs: Arc<dyn FsBackend> = Arc::new(NoopFsBackend);
    let shell: Arc<dyn ShellBackend> = Arc::new(NoopShellBackend);
    let http = Arc::new(NoopHttpClient);
    let ctx = ToolContext::new(cwd, CancellationToken::new(), fs, shell, http, "fake-1");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("rt");
    rt.block_on(async {
        let mut stream = tool.execute(args, ctx);
        let mut out = Vec::new();
        while let Some(ev) = stream.next().await {
            out.push(ev);
        }
        out
    })
}

fn completed_text(events: &[ToolEvent]) -> Option<String> {
    events.iter().find_map(|ev| match ev {
        ToolEvent::Completed(fields) => fields.content.as_ref().and_then(|c| {
            c.iter().find_map(|cc| match cc {
                ToolCallContent::Content(content) => match &content.content {
                    ContentBlock::Text(t) => Some(t.text.clone()),
                    _ => None,
                },
                _ => None,
            })
        }),
        _ => None,
    })
}

#[test]
fn schema_has_name_enum_and_catalog() {
    let tool = SkillTool::new(skills_with(&[
        ("code-review", "review Rust diffs", "body a"),
        ("debug", "find bugs", "body b"),
    ]));
    let schema = tool.schema();
    assert_eq!(schema.name, "skill");
    // Catalog the skills into the description (L1 checklist).
    assert!(
        schema
            .description
            .contains("- code-review: review Rust diffs")
    );
    assert!(schema.description.contains("- debug: find bugs"));
    // The `name` enum contains the discovered names (BTreeMap ⇒ stable ordering).
    let enum_vals = schema.input_schema["properties"]["name"]["enum"]
        .as_array()
        .expect("enum array");
    assert_eq!(enum_vals.len(), 2);
    assert_eq!(enum_vals[0], "code-review");
    assert_eq!(enum_vals[1], "debug");
}

#[test]
fn loads_body_and_directory_hint() {
    let tmp = std::path::Path::new("/");
    let tool = SkillTool::new(skills_with(&[(
        "code-review",
        "review",
        "Run clippy then summarize.",
    )]));
    let events = run_tool(&tool, json!({"name": "code-review"}), tmp);
    let text = completed_text(&events).expect("completed text");
    assert!(text.contains("# Skill: code-review"));
    assert!(text.contains("Run clippy then summarize."));
    // The directory hint allows the model to construct absolute paths to resource files.
    assert!(text.contains("/skills/code-review"));
}

#[test]
fn unknown_skill_fails_loud() {
    let tmp = std::path::Path::new("/");
    let tool = SkillTool::new(skills_with(&[("real", "d", "b")]));
    let events = run_tool(&tool, json!({"name": "ghost"}), tmp);
    assert!(matches!(
        events.last(),
        Some(ToolEvent::Failed(ToolError::InvalidArgs(_)))
    ));
}

#[test]
fn has_skills_reflects_emptiness() {
    let empty: BTreeMap<String, SkillEntry> = BTreeMap::new();
    assert!(!SkillTool::has_skills(&empty));
    let one = skills_with(&[("x", "d", "b")]);
    assert!(SkillTool::has_skills(&one));
}
