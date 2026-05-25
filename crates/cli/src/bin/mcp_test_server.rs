use std::collections::HashMap;
use std::sync::Arc;

use rmcp::ErrorData as McpError;
use rmcp::ServiceExt;
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, JsonObject, ListToolsResult, PaginatedRequestParams,
    ServerCapabilities, ServerInfo, Tool, ToolAnnotations,
};
use serde::Deserialize;
use serde_json::json;

#[derive(Clone)]
struct TestToolServer {
    tools: Arc<Vec<Tool>>,
}

impl TestToolServer {
    fn new() -> Result<Self, serde_json::Error> {
        Ok(Self {
            tools: Arc::new(vec![Self::echo_tool()?]),
        })
    }

    fn echo_tool() -> Result<Tool, serde_json::Error> {
        let schema: JsonObject = serde_json::from_value(json!({
            "type": "object",
            "properties": {
                "message": { "type": "string" }
            },
            "required": ["message"],
            "additionalProperties": false
        }))?;
        Ok(Tool::new(
            "echo",
            "Echo back the provided message and include environment data.",
            Arc::new(schema),
        )
        .with_annotations(ToolAnnotations::new().read_only(true)))
    }
}

#[derive(Deserialize)]
struct EchoArgs {
    message: String,
}

impl ServerHandler for TestToolServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
    }

    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: rmcp::service::RequestContext<rmcp::service::RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListToolsResult, McpError>> + Send + '_ {
        let tools = self.tools.clone();
        async move {
            Ok(ListToolsResult {
                tools: (*tools).clone(),
                next_cursor: None,
                meta: None,
            })
        }
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: rmcp::service::RequestContext<rmcp::service::RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let arguments = request
            .arguments
            .ok_or_else(|| McpError::invalid_params("missing arguments for echo tool", None))?;
        let args: EchoArgs = serde_json::from_value(serde_json::Value::Object(arguments))
            .map_err(|err| McpError::invalid_params(err.to_string(), None))?;
        let env_snapshot: HashMap<String, String> = std::env::vars().collect();
        let structured_content = json!({
            "echo": args.message,
            "env": env_snapshot.get("MCP_TEST_VALUE"),
        });
        Ok(CallToolResult::structured(structured_content))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let service = TestToolServer::new()?;
    let running = service
        .serve((tokio::io::stdin(), tokio::io::stdout()))
        .await?;
    let _quit_reason = running.waiting().await?;
    Ok(())
}
