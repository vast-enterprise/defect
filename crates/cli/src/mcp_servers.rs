//! Translates the typed configuration from the `[mcp]` section into a list of ACP
//! `McpServer` values.
//!
//! Only assembles servers explicitly listed in `enabled_servers`; server section configs
//! not in the allowlist are invisible to the client even if present.

use std::path::PathBuf;

use agent_client_protocol_schema::{
    EnvVariable, HttpHeader, McpServer, McpServerHttp, McpServerSse, McpServerStdio,
};
use defect_config::{LoadedConfig, McpServerConfig as ConfigMcpServerConfig};

/// Default MCP server list — the client uses this array during session startup to build
/// the session-level MCP factory.
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
