use std::collections::BTreeMap;
use std::path::Path;

use crate::types::{
    McpConfig, McpRemoteServerConfig, McpSection, McpServerConfig, McpServerSection,
    McpStdioServerConfig, McpTransportKind,
};

pub(crate) fn resolve_mcp_config(path: &Path, section: McpSection) -> Result<McpConfig, String> {
    let servers = section
        .servers
        .unwrap_or_default()
        .into_iter()
        .map(|(name, server)| {
            resolve_mcp_server_config(path, &name, server).map(|server| (name, server))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let enabled_servers = section.enabled_servers.unwrap_or_default();

    for name in &enabled_servers {
        if !servers.contains_key(name) {
            return Err(format!(
                "mcp.enabled_servers references undefined server `{name}` at {}",
                path.display()
            ));
        }
    }

    Ok(McpConfig {
        enabled_servers,
        servers,
    })
}

fn resolve_mcp_server_config(
    path: &Path,
    name: &str,
    server: McpServerSection,
) -> Result<McpServerConfig, String> {
    let Some(transport) = server.transport else {
        return Err(format!(
            "mcp.servers.{name}.transport is required at {}",
            path.display()
        ));
    };

    match transport {
        McpTransportKind::Stdio => resolve_stdio_server(path, name, server),
        McpTransportKind::Http => resolve_remote_server(path, name, server, McpTransportKind::Http),
        McpTransportKind::Sse => resolve_remote_server(path, name, server, McpTransportKind::Sse),
    }
}

fn resolve_stdio_server(
    path: &Path,
    name: &str,
    server: McpServerSection,
) -> Result<McpServerConfig, String> {
    let McpServerSection {
        transport: _,
        command,
        args,
        env,
        url,
        headers,
    } = server;
    let Some(command) = command else {
        return Err(format!(
            "mcp.servers.{name}.command is required for stdio transport at {}",
            path.display()
        ));
    };
    if url.is_some() {
        return Err(format!(
            "mcp.servers.{name}.url is not allowed for stdio transport at {}",
            path.display()
        ));
    }
    if headers.is_some() {
        return Err(format!(
            "mcp.servers.{name}.headers is not allowed for stdio transport at {}",
            path.display()
        ));
    }

    Ok(McpServerConfig::Stdio(McpStdioServerConfig {
        command,
        args: args.unwrap_or_default(),
        env: env.unwrap_or_default(),
    }))
}

fn resolve_remote_server(
    path: &Path,
    name: &str,
    server: McpServerSection,
    transport: McpTransportKind,
) -> Result<McpServerConfig, String> {
    let McpServerSection {
        transport: _,
        command,
        args,
        env,
        url,
        headers,
    } = server;
    let Some(url) = url else {
        return Err(format!(
            "mcp.servers.{name}.url is required for {} transport at {}",
            transport_name(transport),
            path.display()
        ));
    };
    if command.is_some() {
        return Err(format!(
            "mcp.servers.{name}.command is not allowed for {} transport at {}",
            transport_name(transport),
            path.display()
        ));
    }
    if args.is_some() {
        return Err(format!(
            "mcp.servers.{name}.args is not allowed for {} transport at {}",
            transport_name(transport),
            path.display()
        ));
    }
    if env.is_some() {
        return Err(format!(
            "mcp.servers.{name}.env is not allowed for {} transport at {}",
            transport_name(transport),
            path.display()
        ));
    }

    let config = McpRemoteServerConfig {
        url,
        headers: headers.unwrap_or_default(),
    };
    Ok(match transport {
        McpTransportKind::Http => McpServerConfig::Http(config),
        McpTransportKind::Sse => McpServerConfig::Sse(config),
        McpTransportKind::Stdio => unreachable!("remote resolver only accepts http/sse"),
    })
}

const fn transport_name(transport: McpTransportKind) -> &'static str {
    match transport {
        McpTransportKind::Stdio => "stdio",
        McpTransportKind::Http => "http",
        McpTransportKind::Sse => "sse",
    }
}

