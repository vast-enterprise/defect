//! `.mcp.json` 解析：兼容 Claude Code / Cursor 的事实标准 schema，让用户把一份
//! `.mcp.json` 丢在仓库根即可生效（「定义即启用」）。
//!
//! Schema（与生态对齐）：
//! ```json
//! {
//!   "mcpServers": {
//!     "fs":   { "command": "npx", "args": ["-y", "@x/fs"], "env": { "ROOT": "/x" } },
//!     "docs": { "url": "https://example.com/mcp", "headers": { "x-key": "..." } }
//!   }
//! }
//! ```
//! transport 推断：显式 `"type"` 优先；否则有 `command` 视作 stdio、有 `url` 视作
//! 远程（默认 http，`"type": "sse"` 时为 sse）。与 TOML `[mcp]` 不同——TOML 要求
//! 显式 `transport`，而 `.mcp.json` 沿用生态惯例做字段推断，降低粘贴成本。

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

use serde::Deserialize;

use crate::types::{
    ConfigWarning, McpConfig, McpRemoteServerConfig, McpServerConfig, McpStdioServerConfig,
};

/// Repo-root filename for the ecosystem-standard MCP config.
pub(crate) const MCP_JSON_RELATIVE: &str = ".mcp.json";

/// Merge a repo-root `.mcp.json` into the TOML-derived [`McpConfig`].
///
/// "Define = enable": every server in `.mcp.json` is added to both `servers` and
/// `enabled_servers`. On a name collision the TOML `[mcp]` entry wins (it is the more
/// explicit, layered source) and a [`ConfigWarning::McpJsonOverridden`] is emitted.
///
/// Returns the warnings produced (collisions). A missing file is not an error.
pub(crate) fn merge_repo_mcp_json(
    repo_root: Option<&Path>,
    config: &mut McpConfig,
) -> Result<Vec<ConfigWarning>, String> {
    let Some(root) = repo_root else {
        return Ok(Vec::new());
    };
    let path = root.join(MCP_JSON_RELATIVE);
    let raw = match fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(format!("failed to read {}: {err}", path.display())),
    };

    let servers = parse_mcp_json(&path, &raw)?;
    Ok(merge_servers(&path, config, servers))
}

fn merge_servers(
    path: &Path,
    config: &mut McpConfig,
    servers: BTreeMap<String, McpServerConfig>,
) -> Vec<ConfigWarning> {
    let mut warnings = Vec::new();
    for (name, server) in servers {
        if config.servers.contains_key(&name) {
            // TOML [mcp] is the more explicit source — keep it, warn about the shadowed
            // .mcp.json entry.
            warnings.push(ConfigWarning::McpJsonOverridden {
                path: path.to_path_buf(),
                server: name,
            });
            continue;
        }
        config.servers.insert(name.clone(), server);
        if !config.enabled_servers.contains(&name) {
            config.enabled_servers.push(name);
        }
    }
    warnings
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct McpJsonFile {
    #[serde(rename = "mcpServers")]
    mcp_servers: BTreeMap<String, McpJsonServer>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct McpJsonServer {
    #[serde(rename = "type")]
    kind: Option<String>,
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    url: Option<String>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
}

/// Parse a `.mcp.json` document into resolved server configs.
///
/// Returns the servers keyed by name. The caller decides enablement
/// (`.mcp.json` is "define = enable"); this function only validates shape.
pub(crate) fn parse_mcp_json(
    path: &Path,
    raw: &str,
) -> Result<BTreeMap<String, McpServerConfig>, String> {
    let file: McpJsonFile = serde_json::from_str(raw)
        .map_err(|err| format!("invalid .mcp.json at {}: {err}", path.display()))?;

    file.mcp_servers
        .into_iter()
        .map(|(name, server)| resolve_server(path, &name, server).map(|cfg| (name, cfg)))
        .collect()
}

fn resolve_server(
    path: &Path,
    name: &str,
    server: McpJsonServer,
) -> Result<McpServerConfig, String> {
    let McpJsonServer {
        kind,
        command,
        args,
        env,
        url,
        headers,
    } = server;

    // 显式 type 优先；否则按字段存在性推断。
    let kind = kind.as_deref().map(str::to_ascii_lowercase);
    match kind.as_deref() {
        Some("stdio") => build_stdio(path, name, command, args, env, url, headers),
        Some("http") => build_remote(path, name, false, command, url, headers),
        Some("sse") => build_remote(path, name, true, command, url, headers),
        Some(other) => Err(format!(
            "mcpServers.{name}.type `{other}` is not one of stdio/http/sse at {}",
            path.display()
        )),
        None => {
            if command.is_some() {
                build_stdio(path, name, command, args, env, url, headers)
            } else if url.is_some() {
                build_remote(path, name, false, command, url, headers)
            } else {
                Err(format!(
                    "mcpServers.{name} needs either `command` (stdio) or `url` (http/sse) at {}",
                    path.display()
                ))
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_stdio(
    path: &Path,
    name: &str,
    command: Option<String>,
    args: Vec<String>,
    env: BTreeMap<String, String>,
    url: Option<String>,
    headers: BTreeMap<String, String>,
) -> Result<McpServerConfig, String> {
    let Some(command) = command else {
        return Err(format!(
            "mcpServers.{name} stdio transport requires `command` at {}",
            path.display()
        ));
    };
    if url.is_some() {
        return Err(format!(
            "mcpServers.{name} stdio transport must not set `url` at {}",
            path.display()
        ));
    }
    if !headers.is_empty() {
        return Err(format!(
            "mcpServers.{name} stdio transport must not set `headers` at {}",
            path.display()
        ));
    }
    Ok(McpServerConfig::Stdio(McpStdioServerConfig {
        command,
        args,
        env,
    }))
}

fn build_remote(
    path: &Path,
    name: &str,
    sse: bool,
    command: Option<String>,
    url: Option<String>,
    headers: BTreeMap<String, String>,
) -> Result<McpServerConfig, String> {
    let transport = if sse { "sse" } else { "http" };
    let Some(url) = url else {
        return Err(format!(
            "mcpServers.{name} {transport} transport requires `url` at {}",
            path.display()
        ));
    };
    if command.is_some() {
        return Err(format!(
            "mcpServers.{name} {transport} transport must not set `command` at {}",
            path.display()
        ));
    }
    let config = McpRemoteServerConfig { url, headers };
    Ok(if sse {
        McpServerConfig::Sse(config)
    } else {
        McpServerConfig::Http(config)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p() -> &'static Path {
        Path::new("/repo/.mcp.json")
    }

    #[test]
    fn infers_stdio_from_command() {
        let raw = r#"{ "mcpServers": { "fs": { "command": "npx", "args": ["-y", "x"], "env": { "R": "1" } } } }"#;
        let out = parse_mcp_json(p(), raw).expect("parse");
        match out.get("fs").expect("fs") {
            McpServerConfig::Stdio(s) => {
                assert_eq!(s.command, "npx");
                assert_eq!(s.args, vec!["-y".to_string(), "x".to_string()]);
                assert_eq!(s.env.get("R").map(String::as_str), Some("1"));
            }
            other => panic!("expected stdio, got {other:?}"),
        }
    }

    #[test]
    fn infers_http_from_url() {
        let raw =
            r#"{ "mcpServers": { "docs": { "url": "https://x/mcp", "headers": { "k": "v" } } } }"#;
        let out = parse_mcp_json(p(), raw).expect("parse");
        match out.get("docs").expect("docs") {
            McpServerConfig::Http(r) => {
                assert_eq!(r.url, "https://x/mcp");
                assert_eq!(r.headers.get("k").map(String::as_str), Some("v"));
            }
            other => panic!("expected http, got {other:?}"),
        }
    }

    #[test]
    fn explicit_sse_type() {
        let raw = r#"{ "mcpServers": { "s": { "type": "sse", "url": "https://x/sse" } } }"#;
        let out = parse_mcp_json(p(), raw).expect("parse");
        assert!(matches!(out.get("s"), Some(McpServerConfig::Sse(_))));
    }

    #[test]
    fn stdio_with_url_is_rejected() {
        let raw = r#"{ "mcpServers": { "bad": { "command": "x", "url": "https://y" } } }"#;
        let err = parse_mcp_json(p(), raw).expect_err("should reject");
        assert!(err.contains("must not set `url`"), "{err}");
    }

    #[test]
    fn missing_command_and_url_is_rejected() {
        let raw = r#"{ "mcpServers": { "bad": { "env": { "A": "B" } } } }"#;
        let err = parse_mcp_json(p(), raw).expect_err("should reject");
        assert!(err.contains("needs either `command`"), "{err}");
    }

    #[test]
    fn unknown_field_is_rejected() {
        let raw = r#"{ "mcpServers": { "x": { "command": "c", "bogus": 1 } } }"#;
        let err = parse_mcp_json(p(), raw).expect_err("should reject");
        assert!(err.contains("invalid .mcp.json"), "{err}");
    }
}
