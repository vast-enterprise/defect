//! [`ToolRegistry`] 的两种实现：静态注册表 + 复合查询。
//!
//! 设计详见 `docs/internal/session.md` §6：
//! - [`StaticToolRegistry`]：进程级（内置工具）或会话级（MCP 工具）
//!   各自一份，构造后不可变
//! - [`CompositeRegistry`]：把两份串起来，主循环只看到一个统一接口
//!   （`get` 时先查会话级、再查进程级；schemas 拼接两份）

use std::collections::HashMap;
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
            self.schemas[pos] = schema.clone();
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
        let session_names: std::collections::HashSet<&str> =
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
mod tests {
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
}
