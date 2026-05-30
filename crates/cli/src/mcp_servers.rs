//! 把 `[mcp]` 段的 typed 配置翻译成 ACP `McpServer` 列表。
//!
//! 仅装配 `enabled_servers` 中显式列出的；未在白名单里的 server 段配置
//! 即使存在也不被 client 看到。

use std::path::PathBuf;

use agent_client_protocol_schema::{
    EnvVariable, HttpHeader, McpServer, McpServerHttp, McpServerSse, McpServerStdio,
};
use defect_config::{LoadedConfig, McpServerConfig as ConfigMcpServerConfig};

/// 默认 MCP server 列表——session 启动期 client 会用这个数组生成
/// session-level MCP factory。
pub fn build_default_mcp_servers(config: &LoadedConfig) -> Vec<McpServer> {
    config
        .effective
        .mcp
        .enabled_servers
        .iter()
        .filter_map(|name| {
            let server = config.effective.mcp.servers.get(name)?;
            Some(match server {
                ConfigMcpServerConfig::Stdio(server) => McpServer::Stdio(
                    McpServerStdio::new(name, PathBuf::from(&server.command))
                        .args(server.args.clone())
                        .env(
                            server
                                .env
                                .iter()
                                .map(|(name, value)| EnvVariable::new(name, value))
                                .collect(),
                        ),
                ),
                ConfigMcpServerConfig::Http(server) => McpServer::Http(
                    McpServerHttp::new(name, &server.url).headers(
                        server
                            .headers
                            .iter()
                            .map(|(name, value)| HttpHeader::new(name, value))
                            .collect(),
                    ),
                ),
                ConfigMcpServerConfig::Sse(server) => McpServer::Sse(
                    McpServerSse::new(name, &server.url).headers(
                        server
                            .headers
                            .iter()
                            .map(|(name, value)| HttpHeader::new(name, value))
                            .collect(),
                    ),
                ),
            })
        })
        .collect()
}
