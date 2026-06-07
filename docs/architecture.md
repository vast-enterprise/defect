# Defect Architecture

## 1. Goal & Positioning

Defect is a **headless** general-purpose agent emphasizing:

- **High configurability**: LLM provider, tool set, sandbox policy, storage backend — all replaceable
- **High compatibility**: ACP protocol for any frontend (Zed, etc.); MCP for third-party tool ecosystem
- **Clean harness**: Unified abstraction for main loop, event stream, and tool-call semantics — adding a provider/tool doesn't require touching core
- **Resource efficiency**: Pure Rust, single binary deployment

This repository **provides no UI**. Frontends communicate with defect via ACP (Agent Client Protocol).

## 2. Key Architecture Decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| External protocol | Zed's ACP ([agentclientprotocol.com](https://agentclientprotocol.com)) | Existing spec; Zed and other frontends connect directly, no custom protocol needed |
| LLM provider scope | Anthropic + OpenAI-compatible API | Covers Claude, OpenAI, DeepSeek, Qwen, local vllm, and most other backends |
| Tool extension model | Builtin trait + MCP dual track | Crate-internal trait for builtin tools (performance/semantics); MCP for third-party ecosystem |
| Sandbox (v0 scope) | Policy decision layer (read-only / auto / full + path allowlist) | OS-level isolation (landlock/seatbelt/seccomp) as future pluggable backend |
| Session persistence | v0 writes immediately: jsonl append-only, resumable | Indexed storage (sqlite etc.) for future evolution |
| Naming | Bare crate directory names, `defect-` package prefix | Clean directory structure, clear publish namespace |

## 2.1 Config Layers

Current config layer hierarchy:

```text
default < user < project < project-local < CLI
```

Corresponding locations:
- User config: `$XDG_CONFIG_HOME/defect/config.toml` or `~/.config/defect/config.toml`
- Project shared config: `<repo>/.defect/config.toml`
- Project local override: `<repo>/.defect/config.local.toml`

Shared project config targets repository content with security constraints; local project overrides are machine-local and excluded from git by default.

## 3. Crate Layout

```
crates/
├── agent/    → defect-agent      Core: Session/Turn/Event, LlmProvider/Tool traits
├── llm/      → defect-llm       Anthropic + OpenAI-compatible providers
├── tools/    → defect-tools     Builtin tools: fs/edit/bash/grep/...
├── mcp/      → defect-mcp       MCP client, wraps external servers as Tools
├── sandbox/  → defect-sandbox   Permission policy + path allowlist
├── storage/  → defect-storage   Session persistence (jsonl)
├── http/     → defect-http      HTTP client with retry/proxy/tracing
├── obs/      → defect-obs       Observability (tracing, Langfuse integration)
├── config/   → defect-config    Config loading, merging, schema validation
├── acp/      → defect-acp       ACP transport layer (stdio, Unix socket)
├── cli/      → defect-cli       CLI binary entry point
```

## 4. Data Flow

```
ACP client (stdin/stdout or socket)
        ↕
   defect-acp    ←→   defect-agent (Session / Turn / Event)
        ↕                      ↕
   AgentEvent stream    LlmProvider trait
        ↕                      ↕
   defect-obs          defect-llm (vendor implementations)
   (tracing)           defect-mcp (MCP tools)
                        defect-tools (builtin tools)
                        defect-sandbox (policy decisions)
                        defect-storage (jsonl persistence)
```