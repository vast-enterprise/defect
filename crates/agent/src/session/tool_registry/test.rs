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
            .insert(Arc::new(StubTool::new("fs"))) // override
            .insert(Arc::new(StubTool::new("mcp.linear")))
            .build(),
    ) as Arc<dyn ToolRegistry>;

    let comp = CompositeRegistry::new(session, process);
    let names: Vec<String> = comp.schemas().into_iter().map(|s| s.name).collect();
    // session 在前；process 中重名的 fs 被剔除
    assert_eq!(names, vec!["fs", "mcp.linear", "grep"]);
    assert!(comp.get("fs").is_some());
    assert!(comp.get("grep").is_some());
    assert!(comp.get("mcp.linear").is_some());
    assert!(comp.get("nope").is_none());
}
