use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use rmcp::ErrorData as McpError;
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, JsonObject, ListToolsResult, PaginatedRequestParams,
    ServerCapabilities, ServerInfo, Tool, ToolAnnotations,
};
use rmcp::transport::StreamableHttpServerConfig;
use rmcp::transport::StreamableHttpService;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
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
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let bound_addr = listener.local_addr()?;
    if let Ok(bound_addr_file) = std::env::var("MCP_STREAMABLE_HTTP_BOUND_ADDR_FILE") {
        std::fs::write(bound_addr_file, bound_addr.to_string())?;
    }

    let router = Router::new().nest_service(
        "/mcp",
        StreamableHttpService::new(
            || TestToolServer::new().map_err(Into::into),
            Arc::new(LocalSessionManager::default()),
            StreamableHttpServerConfig::default()
                .disable_allowed_hosts()
                .with_sse_keep_alive(Some(Duration::from_secs(1))),
        ),
    );

    axum::serve(listener, router).await?;
    Ok(())
}
