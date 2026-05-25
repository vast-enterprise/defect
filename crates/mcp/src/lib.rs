//! MCP 客户端适配层。
//!
//! 把外部 MCP server 暴露的工具包装为 [`defect_agent`] 的 per-session
//! 工具表。

use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol::schema::{Content as AcpContent, McpServer, McpServerStdio};
use agent_client_protocol::schema::{ToolCallContent, ToolCallUpdateFields};
use defect_agent::error::BoxError;
use defect_agent::session::{SessionToolFactory, StaticToolRegistryBuilder, ToolRegistry};
use defect_agent::tool::{
    SafetyClass, Tool, ToolCallDescription, ToolContext, ToolEvent, ToolSchema, ToolStream,
};
use futures::future::BoxFuture;
use futures::stream;
use rmcp::model::{CallToolRequestParams, RawContent, Tool as McpTool};
use rmcp::service::{Peer, RoleClient, RunningService};
use rmcp::transport::child_process::TokioChildProcess;
use rmcp::{ClientHandler, ServiceExt};
use serde_json::Value;
use thiserror::Error;

/// MCP 适配错误。
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum McpAdapterError {
    #[error("unsupported MCP transport: {0}")]
    UnsupportedTransport(String),

    #[error("rmcp initialization failed: {0}")]
    Initialize(#[source] std::io::Error),

    #[error("rmcp request failed: {0}")]
    Request(#[source] std::io::Error),
}

/// 最小 MCP 工厂。
#[derive(Debug, Default, Clone)]
pub struct McpToolFactory;

impl McpToolFactory {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl SessionToolFactory for McpToolFactory {
    fn build_registry(
        &self,
        cwd: PathBuf,
        mcp_servers: Vec<McpServer>,
    ) -> BoxFuture<'_, Result<Arc<dyn ToolRegistry>, BoxError>> {
        Box::pin(async move {
            let mut builder = StaticToolRegistryBuilder::default();
            for server in mcp_servers {
                let tools = load_server_tools(cwd.clone(), server).await?;
                for tool in tools {
                    builder = builder.insert(tool);
                }
            }
            Ok(Arc::new(builder.build()) as Arc<dyn ToolRegistry>)
        })
    }
}

/// 按 transport 配置加载 MCP 工具。
///
/// # Errors
///
/// 当 transport 不受支持、连接初始化失败、或远端工具列表拉取失败时返回错误。
async fn load_server_tools(
    cwd: PathBuf,
    server: McpServer,
) -> Result<Vec<Arc<dyn Tool>>, BoxError> {
    match server {
        McpServer::Stdio(stdio) => load_stdio_server_tools(cwd, stdio).await,
        other => Err(BoxError::new(McpAdapterError::UnsupportedTransport(
            format!("{other:?}"),
        ))),
    }
}

/// 拉起一个 stdio MCP server，并把其工具包装为本地工具。
///
/// # Errors
///
/// 当子进程启动失败、rmcp 初始化失败、或工具列表请求失败时返回错误。
async fn load_stdio_server_tools(
    cwd: PathBuf,
    server: McpServerStdio,
) -> Result<Vec<Arc<dyn Tool>>, BoxError> {
    let mut command = tokio::process::Command::new(&server.command);
    command.args(&server.args);
    command.current_dir(cwd);
    command.stdin(std::process::Stdio::piped());
    command.stdout(std::process::Stdio::piped());
    command.stderr(std::process::Stdio::inherit());
    for env in server.env {
        command.env(env.name, env.value);
    }

    let transport = TokioChildProcess::new(command)
        .map_err(|source| BoxError::new(McpAdapterError::Initialize(source)))?;
    let client = EmptyClient.serve(transport).await.map_err(service_error)?;
    let peer = client.peer().clone();
    let connection = Arc::new(McpConnection::new(peer.clone(), client));
    let tools = peer.list_all_tools().await.map_err(service_error)?;

    Ok(tools
        .into_iter()
        .map(|tool| Arc::new(McpToolAdapter::new(connection.clone(), tool)) as Arc<dyn Tool>)
        .collect())
}

#[derive(Clone, Default)]
struct EmptyClient;

impl ClientHandler for EmptyClient {}

struct McpConnection {
    peer: Peer<RoleClient>,
    _client: RunningService<RoleClient, EmptyClient>,
}

impl McpConnection {
    fn new(peer: Peer<RoleClient>, client: RunningService<RoleClient, EmptyClient>) -> Self {
        Self {
            peer,
            _client: client,
        }
    }
}

struct McpToolAdapter {
    connection: Arc<McpConnection>,
    name: String,
    schema: ToolSchema,
    safety: SafetyClass,
}

impl McpToolAdapter {
    fn new(connection: Arc<McpConnection>, tool: McpTool) -> Self {
        let input_schema = match serde_json::to_value(tool.input_schema.as_ref()) {
            Ok(schema) => schema,
            Err(err) => {
                tracing::warn!(
                    tool = %tool.name,
                    error = %err,
                    "failed to serialize MCP tool input schema; falling back to empty object"
                );
                Value::Object(Default::default())
            }
        };
        let schema = ToolSchema {
            name: tool.name.to_string(),
            description: tool.description.clone().unwrap_or_default().to_string(),
            input_schema,
        };
        Self {
            connection,
            name: tool.name.to_string(),
            schema,
            safety: infer_safety(&tool),
        }
    }
}

impl Tool for McpToolAdapter {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }

    fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
        self.safety
    }

    fn describe<'a>(
        &'a self,
        _args: &'a serde_json::Value,
        _ctx: ToolContext<'a>,
    ) -> BoxFuture<'a, ToolCallDescription> {
        Box::pin(async move {
            ToolCallDescription {
                fields: ToolCallUpdateFields::new().title(self.schema.description.clone()),
            }
        })
    }

    fn execute(&self, args: serde_json::Value, _ctx: ToolContext<'_>) -> ToolStream {
        let peer = self.connection.peer.clone();
        let name = self.name.clone();
        Box::pin(stream::once(async move {
            let params = match build_call_params(name, args) {
                Ok(params) => params,
                Err(err) => return ToolEvent::Failed(err),
            };

            match peer.call_tool(params).await {
                Ok(call) => completed_event(call),
                Err(err) => ToolEvent::Failed(defect_agent::tool::ToolError::Execution(
                    BoxError::new(std::io::Error::other(err.to_string())),
                )),
            }
        }))
    }
}

fn infer_safety(tool: &McpTool) -> SafetyClass {
    let Some(annotations) = tool.annotations.as_ref() else {
        return SafetyClass::Mutating;
    };
    if annotations.read_only_hint == Some(true) {
        return SafetyClass::ReadOnly;
    }
    if annotations.destructive_hint == Some(true) {
        return SafetyClass::Destructive;
    }
    SafetyClass::Mutating
}

fn build_call_params(
    name: String,
    args: Value,
) -> Result<CallToolRequestParams, defect_agent::tool::ToolError> {
    match args {
        Value::Object(arguments) => Ok(CallToolRequestParams::new(name).with_arguments(arguments)),
        Value::Null => Ok(CallToolRequestParams::new(name)),
        other => Err(defect_agent::tool::ToolError::InvalidArgs(BoxError::new(
            std::io::Error::other(format!("expected object args, got {other}")),
        ))),
    }
}

fn completed_event(call: rmcp::model::CallToolResult) -> ToolEvent {
    let mut content = call
        .content
        .iter()
        .filter_map(content_text)
        .map(|text| ToolCallContent::Content(AcpContent::new(text)))
        .collect::<Vec<_>>();

    if content.is_empty()
        && let Some(structured_content) = call.structured_content.as_ref()
    {
        content.push(ToolCallContent::Content(AcpContent::new(
            structured_content.to_string(),
        )));
    }

    let raw_output = serde_json::to_value(&call).ok();
    ToolEvent::Completed(
        ToolCallUpdateFields::new()
            .content((!content.is_empty()).then_some(content))
            .raw_output(raw_output),
    )
}

fn content_text(content: &rmcp::model::Content) -> Option<String> {
    match &content.raw {
        RawContent::Text(text) => Some(text.text.clone()),
        RawContent::Resource(resource) => match &resource.resource {
            rmcp::model::ResourceContents::TextResourceContents { text, .. } => Some(text.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn service_error<E>(err: E) -> BoxError
where
    E: std::error::Error,
{
    BoxError::new(McpAdapterError::Request(std::io::Error::other(
        err.to_string(),
    )))
}

#[cfg(test)]
mod test;
