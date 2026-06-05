use std::fs;
use std::path::Path;

use tempfile::TempDir;

use crate::loader::{dotenv_updates_from_str, load_config};
use crate::overrides::{merge_toml_values, parse_cli_override};
use crate::types::{
    CliOverrides, ConfigError, ConfigSource, HookCommandSpec, HookHandlerSpec, HttpProxyMode,
    LoadConfigOptions, PROJECT_LOCAL_CONFIG_RELATIVE, ProviderKind, ProviderProtocol,
    ReasoningEffort,
};
use defect_agent::session::WebSearchCapabilityMode;
use defect_agent::tool::SafetyClass;

fn test_options(root: &TempDir) -> LoadConfigOptions {
    LoadConfigOptions {
        cwd: root.path().join("repo"),
        cli: CliOverrides::default(),
        xdg_config_home: Some(root.path().join("xdg")),
        home_dir: None,
        local: false,
    }
}

fn write(path: &Path, body: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("parent dirs");
    }
    fs::write(path, body).expect("write file");
}

#[test]
fn merges_user_project_and_local_by_precedence() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[default]
provider = "echo"
model = "user-model"

[turn]
max_llm_retries = 5
max_hook_continues = 7
"#,
    );
    write(
        &repo.join(".defect/config.toml"),
        r#"
[default]
model = "project-model"
"#,
    );
    write(
        &repo.join(PROJECT_LOCAL_CONFIG_RELATIVE),
        r#"
[default]
model = "local-model"
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");

    assert_eq!(loaded.effective.cli.provider, ProviderKind::Echo);
    assert_eq!(loaded.effective.cli.model, "local-model");
    assert_eq!(loaded.effective.turn.max_llm_retries, 5);
    assert_eq!(loaded.effective.turn.max_hook_continues, 7);
    assert_eq!(loaded.layers.layers.len(), 4);
}

#[test]
fn cli_overrides_win_over_local_layer() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &repo.join(PROJECT_LOCAL_CONFIG_RELATIVE),
        r#"
[default]
provider = "openai"
model = "local-model"
"#,
    );

    let mut opts = test_options(&tmp);
    opts.cli.provider = Some(ProviderKind::Anthropic);
    opts.cli.model = Some("cli-model".into());
    let loaded = load_config(opts).expect("load config");

    assert_eq!(loaded.effective.cli.provider, ProviderKind::Anthropic);
    assert_eq!(loaded.effective.cli.model, "cli-model");
}

#[test]
fn provider_models_and_default_model_flow_into_effective_config() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[default]
provider = "openai"

[providers.openai]
default_model = "gpt-4.1-mini"
models = ["gpt-4.1-mini", "gpt-4.1", "o4-mini"]
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");

    assert_eq!(loaded.effective.cli.provider, ProviderKind::Openai);
    assert_eq!(loaded.effective.cli.model, "gpt-4.1-mini");
    assert_eq!(loaded.effective.turn.model, "gpt-4.1-mini");
    assert_eq!(
        loaded.effective.turn.allowed_models.as_deref(),
        Some(
            [
                "gpt-4.1-mini".to_string(),
                "gpt-4.1".to_string(),
                "o4-mini".to_string(),
            ]
            .as_slice(),
        )
    );
    assert_eq!(
        loaded.effective.providers.openai.default_model.as_deref(),
        Some("gpt-4.1-mini")
    );
    let model_ids: Vec<&str> = loaded
        .effective
        .providers
        .openai
        .models
        .as_ref()
        .expect("openai models present")
        .iter()
        .map(|m| m.id())
        .collect();
    assert_eq!(model_ids, vec!["gpt-4.1-mini", "gpt-4.1", "o4-mini"]);
}

/// `models` 接受裸字符串与 `{ id, name }` table 混用：前者展示名 `None`
/// （UI fallback 到 id），后者带展示名。
#[test]
fn provider_models_accept_id_and_named_table() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[default]
provider = "openai"

[providers.openai]
default_model = "gpt-4.1-mini"
models = [
    "gpt-4.1-mini",
    { id = "gpt-4.1", name = "GPT 4.1" },
]
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");
    let models = loaded
        .effective
        .providers
        .openai
        .models
        .as_ref()
        .expect("openai models present");

    assert_eq!(models[0].id(), "gpt-4.1-mini");
    assert_eq!(models[0].name(), None);
    assert_eq!(models[1].id(), "gpt-4.1");
    assert_eq!(models[1].name(), Some("GPT 4.1"));

    // allowed_models 白名单仍只含 id（展示名不参与白名单匹配）。
    assert_eq!(
        loaded.effective.turn.allowed_models.as_deref(),
        Some(["gpt-4.1-mini", "gpt-4.1"].map(String::from).as_slice())
    );
}

#[test]
fn multiple_configured_providers_contribute_allowed_models() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[default]
provider = "openai"

[providers.openai]
default_model = "gpt-4o-mini"
models = ["gpt-4o-mini"]

[providers.litellm]
default_model = "anthropic/claude-sonnet-4-5"
models = ["anthropic/claude-sonnet-4-5", "openai/gpt-4o"]
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");

    assert_eq!(
        loaded.effective.turn.allowed_models.as_deref(),
        Some(
            [
                "gpt-4o-mini".to_string(),
                "anthropic/claude-sonnet-4-5".to_string(),
                "openai/gpt-4o".to_string(),
            ]
            .as_slice(),
        )
    );
}

#[test]
fn litellm_provider_uses_builtin_section_and_requires_declared_model() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[default]
provider = "litellm"

[providers.litellm]
base_url = "http://localhost:4000/v1"
default_model = "openai/gpt-4o-mini"
models = ["openai/gpt-4o-mini", "anthropic/claude-sonnet-4-5"]
api_key_env = "LITELLM_API_KEY"
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");

    assert_eq!(loaded.effective.cli.provider, ProviderKind::Litellm);
    assert_eq!(loaded.effective.cli.model, "openai/gpt-4o-mini");
    assert_eq!(
        loaded.effective.providers.litellm.base_url.as_deref(),
        Some("http://localhost:4000/v1")
    );
    assert_eq!(
        loaded.effective.providers.litellm.api_key_env.as_deref(),
        Some("LITELLM_API_KEY")
    );
    assert_eq!(
        loaded.effective.turn.allowed_models.as_deref(),
        Some(
            [
                "openai/gpt-4o-mini".to_string(),
                "anthropic/claude-sonnet-4-5".to_string(),
            ]
            .as_slice(),
        )
    );
    assert!(loaded.warnings.is_empty(), "{:?}", loaded.warnings);
}

#[test]
fn litellm_provider_requires_default_model() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[default]
provider = "litellm"
"#,
    );

    let err = load_config(test_options(&tmp)).expect_err("invalid config");

    match err {
        ConfigError::Invalid { message, .. } => {
            assert!(
                message.contains("default.model or providers.litellm.default_model is required")
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn custom_provider_uses_named_section_and_openai_chat_protocol() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[default]
provider = "siliconflow"

[providers.siliconflow]
protocol = "openai-chat"
base_url = "https://api.siliconflow.cn/v1"
default_model = "deepseek-ai/DeepSeek-V3"
models = ["deepseek-ai/DeepSeek-V3"]
display_name = "SiliconFlow"
api_key_env = "SILICONFLOW_API_KEY"
reasoning_effort = "medium"

[providers.siliconflow.headers]
x-provider-test = "enabled"
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");

    let provider = ProviderKind::Custom("siliconflow".to_string());
    assert_eq!(loaded.effective.cli.provider, provider);
    assert_eq!(loaded.effective.cli.model, "deepseek-ai/DeepSeek-V3");
    let custom = loaded
        .effective
        .providers
        .custom
        .get("siliconflow")
        .expect("custom provider");
    assert_eq!(custom.protocol, Some(ProviderProtocol::OpenaiChat));
    assert_eq!(
        custom.base_url.as_deref(),
        Some("https://api.siliconflow.cn/v1")
    );
    assert_eq!(custom.display_name.as_deref(), Some("SiliconFlow"));
    assert_eq!(custom.api_key_env.as_deref(), Some("SILICONFLOW_API_KEY"));
    assert_eq!(
        custom.headers.get("x-provider-test").map(String::as_str),
        Some("enabled")
    );
    assert!(loaded.warnings.is_empty(), "{:?}", loaded.warnings);
}

#[test]
fn custom_provider_parses_aws_section() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[default]
provider = "bedrock"

[providers.bedrock]
protocol = "anthropic-messages"
default_model = "anthropic.claude-sonnet-4-5-20250929-v1:0"

[providers.bedrock.aws]
profile = "work"
region = "us-west-2"
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");

    let provider = ProviderKind::Custom("bedrock".to_string());
    assert_eq!(loaded.effective.cli.provider, provider);
    let custom = loaded
        .effective
        .providers
        .custom
        .get("bedrock")
        .expect("bedrock provider");
    assert_eq!(custom.protocol, Some(ProviderProtocol::AnthropicMessages));
    let aws = custom.aws.as_ref().expect("aws config");
    assert_eq!(aws.profile.as_deref(), Some("work"));
    assert_eq!(aws.region.as_deref(), Some("us-west-2"));
    assert!(loaded.warnings.is_empty(), "{:?}", loaded.warnings);
}

#[test]
fn custom_provider_requires_matching_section() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[default]
provider = "missing"
"#,
    );

    let err = load_config(test_options(&tmp)).expect_err("invalid config");

    match err {
        ConfigError::Invalid { message, .. } => {
            assert!(message.contains("has no matching [providers.missing] section"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn custom_provider_requires_a_default_model() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[default]
provider = "localai"

[providers.localai]
protocol = "openai-chat"
base_url = "http://localhost:8000/v1"
"#,
    );

    let err = load_config(test_options(&tmp)).expect_err("invalid config");

    match err {
        ConfigError::Invalid { message, .. } => {
            assert!(
                message.contains("default.model or providers.localai.default_model is required")
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn provider_reasoning_effort_parses_per_provider() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[providers.openai]
reasoning_effort = "high"

[providers.deepseek]
reasoning_effort = "xhigh"
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");

    assert_eq!(
        loaded.effective.providers.openai.reasoning_effort,
        Some(ReasoningEffort::High)
    );
    assert_eq!(
        loaded.effective.providers.deepseek.reasoning_effort,
        Some(ReasoningEffort::Xhigh)
    );
    assert!(loaded.warnings.is_empty(), "{:?}", loaded.warnings);
}

#[test]
fn base_prompt_uses_highest_precedence_layer_and_resolves_relative_file() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");

    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[base_prompt]
text = "user base"
"#,
    );
    write(
        &repo.join(".defect/config.toml"),
        r#"
[base_prompt]
file = "prompts/project.md"
"#,
    );
    write(&repo.join(".defect/prompts/project.md"), "project base");
    write(
        &repo.join(PROJECT_LOCAL_CONFIG_RELATIVE),
        r#"
[base_prompt]
text = "local base"
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");

    assert_eq!(
        loaded.effective.base_prompt.text.as_deref(),
        Some("local base")
    );
    assert_eq!(loaded.effective.base_prompt.file, None);
    assert_eq!(
        loaded.effective.turn.base_prompt.text.as_deref(),
        Some("local base")
    );
    assert_eq!(loaded.effective.turn.base_prompt.file, None);
}

#[test]
fn base_prompt_preserves_declaring_file_base_path() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &repo.join(".defect/config.toml"),
        r#"
[base_prompt]
file = "prompts/base.md"
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");

    assert_eq!(
        loaded.effective.base_prompt.file.as_deref(),
        Some(repo.join(".defect/prompts/base.md").as_path())
    );
}

#[test]
fn shared_project_layer_can_set_provider_and_endpoint() {
    // 设计宗旨是最小化：不再帮用户审查仓库共享配置是否劫持流量/凭据。
    // 仓库内 .defect/config.toml 与本地层一样，可设 provider / base_url 等。
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &repo.join(".defect/config.toml"),
        r#"
[default]
provider = "openai"

[providers.openai]
base_url = "https://example.invalid/v1"
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");

    assert_eq!(loaded.effective.cli.provider, ProviderKind::Openai);
    assert_eq!(
        loaded.effective.providers.openai.base_url.as_deref(),
        Some("https://example.invalid/v1")
    );
    assert!(loaded.warnings.is_empty(), "{:?}", loaded.warnings);
}

#[test]
fn project_local_layer_can_override_endpoint() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &repo.join(PROJECT_LOCAL_CONFIG_RELATIVE),
        r#"
[default]
provider = "openai"

[providers.openai]
base_url = "https://example.invalid/v1"
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");

    assert_eq!(loaded.effective.cli.provider, ProviderKind::Openai);
    assert_eq!(
        loaded.effective.providers.openai.base_url.as_deref(),
        Some("https://example.invalid/v1")
    );
}

#[test]
fn parses_dotted_cli_override_values() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");

    let mut opts = test_options(&tmp);
    opts.cli.config_overrides = vec![
        parse_cli_override("turn.max_llm_retries=9").expect("override"),
        parse_cli_override("providers.openai.base_url=\"https://localhost:1234/v1\"")
            .expect("override"),
    ];

    let loaded = load_config(opts).expect("load config");
    assert_eq!(loaded.effective.turn.max_llm_retries, 9);
    assert_eq!(
        loaded.effective.providers.openai.base_url.as_deref(),
        Some("https://localhost:1234/v1")
    );
}

#[test]
fn loads_tools_and_sandbox_sections_into_effective_config() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[tools.bash]
default_timeout_ms = 1234
max_timeout_ms = 4321

[tools.fs]
read_default_limit = 12
read_max_limit = 34

[sandbox]
mode = "read-only"
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");

    assert_eq!(loaded.effective.tools.bash.default_timeout_ms, 1234);
    assert_eq!(loaded.effective.tools.bash.max_timeout_ms, 4321);
    assert_eq!(loaded.effective.tools.fs.read_default_limit, 12);
    assert_eq!(loaded.effective.tools.fs.read_max_limit, 34);
    assert_eq!(loaded.effective.sandbox.mode.as_str(), "read-only");
}

#[test]
fn loads_mcp_sections_into_effective_config() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[mcp]
enabled_servers = ["echo", "docs"]

[mcp.servers.echo]
transport = "stdio"
command = "mcp-echo"
args = ["--port", "9000"]

[mcp.servers.echo.env]
MCP_TEST_VALUE = "from-config"

[mcp.servers.docs]
transport = "sse"
url = "http://127.0.0.1:8123/mcp"

[mcp.servers.docs.headers]
x-mcp-test = "enabled"
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");

    assert_eq!(loaded.effective.mcp.enabled_servers, ["echo", "docs"]);
    assert!(matches!(
        loaded.effective.mcp.servers.get("echo"),
        Some(crate::types::McpServerConfig::Stdio(server))
            if server.command == "mcp-echo"
                && server.args == vec!["--port".to_string(), "9000".to_string()]
                && server.env.get("MCP_TEST_VALUE").map(String::as_str) == Some("from-config")
    ));
    assert!(matches!(
        loaded.effective.mcp.servers.get("docs"),
        Some(crate::types::McpServerConfig::Sse(server))
            if server.url == "http://127.0.0.1:8123/mcp"
                && server.headers.get("x-mcp-test").map(String::as_str) == Some("enabled")
    ));
}

#[test]
fn rejects_enabled_mcp_server_names_without_matching_definitions() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    let config_path = tmp.path().join("xdg/defect/config.toml");
    write(
        &config_path,
        r#"
[mcp]
enabled_servers = ["missing"]
"#,
    );

    let err = load_config(test_options(&tmp)).expect_err("invalid config");

    match err {
        ConfigError::Invalid { path, message } => {
            assert_eq!(path, Path::new("<merged>"));
            assert!(message.contains("undefined server `missing`"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn custom_provider_name_accepted_but_unknown_field_rejected() {
    // flatten 让 `[providers.<任意名>]` 开放，但内层 ProviderSection 的
    // deny_unknown_fields 仍然校验字段名。详见 docs/internal/config.md §11.1。
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    let config_path = tmp.path().join("xdg/defect/config.toml");
    write(
        &config_path,
        r#"
[default]
provider = "siliconflow"

[providers.siliconflow]
protocol = "openai-chat"
default_model = "deepseek-ai/DeepSeek-V3"
bogus_field = "value"
"#,
    );

    let err = load_config(test_options(&tmp)).expect_err("unknown provider field must fail");
    match err {
        ConfigError::Invalid { path, message } => {
            assert_eq!(path, config_path);
            assert!(
                message.contains("bogus_field"),
                "unexpected message: {message}"
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn rejects_unknown_keys_with_source_path() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    let config_path = tmp.path().join("xdg/defect/config.toml");
    write(
        &config_path,
        r#"
[default]
provider = "echo"
bogus = "value"
"#,
    );

    // 未知 key 现在直接报错（不再是 warning），且错误带上声明它的文件路径。
    let err = load_config(test_options(&tmp)).expect_err("unknown key must fail");
    match err {
        ConfigError::Invalid { path, message } => {
            assert_eq!(path, config_path);
            assert!(message.contains("bogus"), "unexpected message: {message}");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn dotenv_updates_skip_existing_keys_and_invalid_lines() {
    let updates =
        dotenv_updates_from_str("A=1\n# comment\nBROKEN\nB='two'\nC = \"three\"\n", &["B"]);

    assert_eq!(
        updates,
        vec![
            ("A".to_string(), "1".to_string()),
            ("C".to_string(), "three".to_string()),
        ]
    );
}

#[test]
fn parse_error_reports_source_path() {
    let tmp = TempDir::new().expect("tmp");
    let config_path = tmp.path().join("xdg/defect/config.toml");
    write(
        &config_path,
        r#"
[default
provider = "echo"
"#,
    );

    let err = load_config(test_options(&tmp)).expect_err("parse error");

    match err {
        ConfigError::Parse { path, .. } => assert_eq!(path, config_path),
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn missing_config_files_do_not_error() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");

    let loaded = load_config(test_options(&tmp)).expect("load config");

    assert_eq!(loaded.layers.layers.len(), 1);
    assert!(loaded.warnings.is_empty());
    assert_eq!(loaded.effective.cli.provider, ProviderKind::Echo);
}

#[test]
fn loads_http_section_into_effective_config() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[http]
total_timeout_ms = 90000
transport_retries = 4
initial_backoff_ms = 500
user_agent = "my-agent/1.0"

[http.proxy]
mode = "explicit"
http_proxy = "http://127.0.0.1:10808"
https_proxy = "http://127.0.0.1:10808"
no_proxy = ["localhost", ".internal"]
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");

    let http = &loaded.effective.http;
    assert_eq!(http.total_timeout_ms, Some(90_000));
    assert_eq!(http.transport_retries, Some(4));
    assert_eq!(http.initial_backoff_ms, Some(500));
    assert_eq!(http.user_agent.as_deref(), Some("my-agent/1.0"));
    assert_eq!(http.proxy.mode, HttpProxyMode::Explicit);
    assert_eq!(
        http.proxy.explicit.http_proxy.as_deref(),
        Some("http://127.0.0.1:10808")
    );
    assert_eq!(http.proxy.explicit.no_proxy, vec!["localhost", ".internal"]);
}

#[test]
fn http_section_default_proxy_is_from_env() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");

    let loaded = load_config(test_options(&tmp)).expect("load config");

    assert_eq!(loaded.effective.http.proxy.mode, HttpProxyMode::FromEnv);
    assert!(loaded.effective.http.user_agent.is_none());
}

#[test]
fn shared_project_layer_can_set_http_proxy() {
    // 最小化：仓库共享配置可设 http.proxy（与其它 http 字段一视同仁）。
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &repo.join(".defect/config.toml"),
        r#"
[http]
total_timeout_ms = 30000
user_agent = "team-agent/2.0"

[http.proxy]
mode = "explicit"
http_proxy = "http://proxy.internal:8080"
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");

    assert_eq!(loaded.effective.http.proxy.mode, HttpProxyMode::Explicit);
    assert_eq!(
        loaded.effective.http.proxy.explicit.http_proxy.as_deref(),
        Some("http://proxy.internal:8080")
    );
    assert_eq!(loaded.effective.http.total_timeout_ms, Some(30_000));
    assert_eq!(
        loaded.effective.http.user_agent.as_deref(),
        Some("team-agent/2.0")
    );
    assert!(loaded.warnings.is_empty(), "{:?}", loaded.warnings);
}

#[test]
fn cli_override_can_disable_http_proxy() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");

    let mut opts = test_options(&tmp);
    opts.cli.config_overrides =
        vec![parse_cli_override("http.proxy.mode=\"disabled\"").expect("override")];

    let loaded = load_config(opts).expect("load config");

    assert_eq!(loaded.effective.http.proxy.mode, HttpProxyMode::Disabled);
}

#[test]
fn project_local_layer_can_set_http_proxy() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &repo.join(PROJECT_LOCAL_CONFIG_RELATIVE),
        r#"
[http.proxy]
mode = "disabled"
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");

    assert_eq!(loaded.effective.http.proxy.mode, HttpProxyMode::Disabled);
    assert!(loaded.warnings.is_empty());
}

#[test]
fn capabilities_web_search_default_is_disabled() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");

    let loaded = load_config(test_options(&tmp)).expect("load config");

    assert_eq!(
        loaded.effective.capabilities.web_search.mode,
        WebSearchCapabilityMode::Disabled
    );
    // 三个 provider 的覆写默认为 None（未声明覆写）。
    assert!(
        loaded
            .effective
            .providers
            .anthropic
            .capabilities
            .web_search
            .is_none()
    );
    assert!(
        loaded
            .effective
            .providers
            .openai
            .capabilities
            .web_search
            .is_none()
    );
    assert!(
        loaded
            .effective
            .providers
            .deepseek
            .capabilities
            .web_search
            .is_none()
    );
}

#[test]
fn capabilities_web_search_mode_parses_two_values() {
    for (value, expected) in [
        ("delegate", WebSearchCapabilityMode::Delegate),
        ("disabled", WebSearchCapabilityMode::Disabled),
    ] {
        let tmp = TempDir::new().expect("tmp");
        let repo = tmp.path().join("repo");
        fs::create_dir_all(repo.join(".git")).expect("git");
        write(
            &tmp.path().join("xdg/defect/config.toml"),
            &format!(
                r#"
[capabilities.web_search]
mode = "{value}"
"#
            ),
        );

        let loaded = load_config(test_options(&tmp)).expect("load config");
        assert_eq!(
            loaded.effective.capabilities.web_search.mode, expected,
            "mode = {value}"
        );
    }
}

#[test]
fn provider_capability_overrides_are_loaded() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[capabilities.web_search]
mode = "disabled"

[providers.anthropic.capabilities.web_search]
mode = "delegate"

[providers.openai.capabilities.web_search]
mode = "delegate"

[providers.deepseek.capabilities.web_search]
mode = "disabled"
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");

    assert_eq!(
        loaded.effective.capabilities.web_search.mode,
        WebSearchCapabilityMode::Disabled
    );
    assert_eq!(
        loaded
            .effective
            .providers
            .anthropic
            .capabilities
            .web_search
            .map(|s| s.mode),
        Some(WebSearchCapabilityMode::Delegate)
    );
    assert_eq!(
        loaded
            .effective
            .providers
            .openai
            .capabilities
            .web_search
            .map(|s| s.mode),
        Some(WebSearchCapabilityMode::Delegate)
    );
    assert_eq!(
        loaded
            .effective
            .providers
            .deepseek
            .capabilities
            .web_search
            .map(|s| s.mode),
        Some(WebSearchCapabilityMode::Disabled)
    );
}

#[test]
fn provider_capability_override_merge_falls_back_to_global() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[capabilities.web_search]
mode = "delegate"

[providers.deepseek.capabilities.web_search]
mode = "disabled"
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");

    // Anthropic 没有覆写，merge 后回落到全局 delegate。
    let anthropic_session = loaded
        .effective
        .providers
        .anthropic
        .capabilities
        .merge_into(loaded.effective.capabilities)
        .to_session_capabilities();
    assert_eq!(
        anthropic_session.web_search.mode,
        WebSearchCapabilityMode::Delegate
    );

    // DeepSeek 覆写为 disabled，应当压过全局 delegate。
    let deepseek_session = loaded
        .effective
        .providers
        .deepseek
        .capabilities
        .merge_into(loaded.effective.capabilities)
        .to_session_capabilities();
    assert_eq!(
        deepseek_session.web_search.mode,
        WebSearchCapabilityMode::Disabled
    );
}

#[test]
fn loads_tools_fetch_section_into_effective_config() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[tools.fetch]
enabled = false
default_timeout_secs = 15
max_timeout_secs = 60
max_response_bytes = 1048576
default_format = "html"
html_to_markdown = false
follow_redirects = false
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");

    let fetch = &loaded.effective.tools.fetch;
    assert!(!fetch.enabled);
    assert_eq!(fetch.default_timeout_secs, 15);
    assert_eq!(fetch.max_timeout_secs, 60);
    assert_eq!(fetch.max_response_bytes, 1_048_576);
    assert!(!fetch.html_to_markdown);
    assert!(!fetch.follow_redirects);
}

#[test]
fn tools_fetch_defaults_when_absent() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");

    let loaded = load_config(test_options(&tmp)).expect("load config");

    let fetch = &loaded.effective.tools.fetch;
    assert!(fetch.enabled);
    assert_eq!(fetch.default_timeout_secs, 30);
    assert_eq!(fetch.max_timeout_secs, 120);
    assert_eq!(fetch.max_response_bytes, 5 * 1024 * 1024);
    assert!(fetch.html_to_markdown);
    assert!(fetch.follow_redirects);
}

#[test]
fn loads_tools_background_section_into_effective_config() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[tools.background]
default_recent_blocks = 16
block_text_limit = 200
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");

    let bg = &loaded.effective.tools.background;
    assert_eq!(bg.default_recent_blocks, 16);
    assert_eq!(bg.block_text_limit, 200);
}

#[test]
fn tools_background_defaults_when_absent() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");

    let loaded = load_config(test_options(&tmp)).expect("load config");

    let bg = &loaded.effective.tools.background;
    // 默认：最近 10 条、正文上限 0（鸟瞰、不灌子 turn 正文）。
    assert_eq!(bg.default_recent_blocks, 10);
    assert_eq!(bg.block_text_limit, 0);
}

#[test]
fn arrays_replace_instead_of_append() {
    let mut base = toml::from_str::<toml::Value>(
        r#"
items = ["user", "project"]
"#,
    )
    .expect("base");
    let overlay = toml::from_str::<toml::Value>(
        r#"
items = ["cli"]
"#,
    )
    .expect("overlay");

    merge_toml_values(&mut base, &overlay);

    assert_eq!(
        base.get("items").and_then(toml::Value::as_array),
        Some(&vec![toml::Value::String("cli".to_string())])
    );
}

// ---------------------------------------------------------------------------
// hooks
// ---------------------------------------------------------------------------

#[test]
fn parses_hooks_section_full_shape() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[[hooks.after_session_enter]]
handler = { type = "builtin", name = "preload-readme" }

[[hooks.before_ingest]]
handler = { type = "builtin", name = "skill-router" }

[[hooks.before_tool_apply]]
match = { tool = "bash", safety = ["destructive"] }
handler = { type = "command", argv = ["./scripts/audit.sh"], timeout_sec = 10 }

[[hooks.before_tool_apply]]
match = { tool_glob = "fs.*" }
handler = { type = "command", shell = "bash", command = "echo hi" }

[[hooks.after_tool_apply]]
handler = { type = "builtin", name = "tracing-audit" }

[[hooks.before_turn_end]]
handler = { type = "prompt", system = "diagnose", render = { type = "json" }, timeout_sec = 5 }
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");
    let hooks = &loaded.effective.hooks;

    assert_eq!(hooks.get("after_session_enter").len(), 1);
    assert!(matches!(
        &hooks.get("after_session_enter")[0].handler,
        HookHandlerSpec::Builtin { name } if name == "preload-readme"
    ));

    assert_eq!(hooks.get("before_tool_apply").len(), 2);
    let first = &hooks.get("before_tool_apply")[0];
    assert_eq!(first.matcher.tool.as_deref(), Some("bash"));
    assert_eq!(first.matcher.safety, vec![SafetyClass::Destructive]);
    match &first.handler {
        HookHandlerSpec::Command(HookCommandSpec::Argv {
            argv, timeout_sec, ..
        }) => {
            assert_eq!(argv, &vec!["./scripts/audit.sh".to_string()]);
            assert_eq!(*timeout_sec, Some(10));
        }
        other => panic!("expected argv command, got {other:?}"),
    }

    let second = &hooks.get("before_tool_apply")[1];
    assert!(matches!(
        &second.handler,
        HookHandlerSpec::Command(HookCommandSpec::Shell { .. })
    ));

    assert_eq!(hooks.get("after_tool_apply").len(), 1);
    assert_eq!(hooks.get("before_turn_end").len(), 1);
    assert!(matches!(
        &hooks.get("before_turn_end")[0].handler,
        HookHandlerSpec::Prompt(_)
    ));

    assert_eq!(
        hooks.get("after_session_enter")[0].source,
        ConfigSource::User
    );
}

#[test]
fn hooks_append_across_layers_in_declaration_order() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[[hooks.before_tool_apply]]
handler = { type = "builtin", name = "user-hook" }
"#,
    );
    write(
        &repo.join(".defect/config.toml"),
        r#"
[[hooks.before_tool_apply]]
handler = { type = "builtin", name = "project-hook" }
"#,
    );
    write(
        &repo.join(PROJECT_LOCAL_CONFIG_RELATIVE),
        r#"
[[hooks.before_tool_apply]]
handler = { type = "builtin", name = "local-hook" }
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");
    let names: Vec<&str> = loaded
        .effective
        .hooks
        .get("before_tool_apply")
        .iter()
        .map(|e| match &e.handler {
            HookHandlerSpec::Builtin { name } => name.as_str(),
            _ => "<other>",
        })
        .collect();
    assert_eq!(names, vec!["user-hook", "project-hook", "local-hook"]);

    let sources: Vec<ConfigSource> = loaded
        .effective
        .hooks
        .get("before_tool_apply")
        .iter()
        .map(|e| e.source)
        .collect();
    assert_eq!(
        sources,
        vec![
            ConfigSource::User,
            ConfigSource::Project,
            ConfigSource::ProjectLocal,
        ]
    );
}

#[test]
fn hooks_dedupe_identical_entries_across_layers() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    let body = r#"
[[hooks.after_tool_apply]]
handler = { type = "builtin", name = "tracing-audit" }
"#;
    write(&tmp.path().join("xdg/defect/config.toml"), body);
    write(&repo.join(".defect/config.toml"), body);

    let loaded = load_config(test_options(&tmp)).expect("load config");
    assert_eq!(loaded.effective.hooks.get("after_tool_apply").len(), 1);
}

#[test]
fn hooks_disable_removes_upstream_entry() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[[hooks.after_tool_apply]]
handler = { type = "builtin", name = "tracing-audit" }
"#,
    );
    write(
        &repo.join(PROJECT_LOCAL_CONFIG_RELATIVE),
        r#"
[[hooks.disable]]
event = "after_tool_apply"
handler = { type = "builtin", name = "tracing-audit" }
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");
    assert!(loaded.effective.hooks.get("after_tool_apply").is_empty());
}

#[test]
fn hooks_invalid_command_handler_errors_loud() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[[hooks.before_tool_apply]]
handler = { type = "command", argv = [], timeout_sec = 1 }
"#,
    );

    let err = load_config(test_options(&tmp)).expect_err("expected invalid");
    match err {
        ConfigError::Invalid { message, .. } => {
            assert!(message.contains("argv"), "unexpected message: {message}");
        }
        other => panic!("expected ConfigError::Invalid, got {other:?}"),
    }
}

#[test]
fn parses_langfuse_from_user_config() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[tracing.langfuse]
enabled = true
host = "https://eu.cloud.langfuse.com"
public_key = "pk-lf-xxx"
secret_key = "sk-lf-yyy"
flush_interval_ms = 5000
max_batch = 50
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");
    let lf = loaded
        .effective
        .tracing
        .langfuse
        .expect("langfuse config present");
    assert!(lf.enabled);
    assert_eq!(lf.host.as_deref(), Some("https://eu.cloud.langfuse.com"));
    assert_eq!(lf.public_key.as_deref(), Some("pk-lf-xxx"));
    assert_eq!(lf.secret_key.as_deref(), Some("sk-lf-yyy"));
    assert_eq!(lf.flush_interval_ms, Some(5000));
    assert_eq!(lf.max_batch, Some(50));
}

#[test]
fn shared_project_config_can_set_langfuse() {
    // 最小化：不再剥离仓库共享配置里的 langfuse 段，原样生效。
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &repo.join(".defect/config.toml"),
        r#"
[tracing.langfuse]
enabled = true
host = "https://eu.cloud.langfuse.com"
secret_key = "sk-lf-team"
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("load config");
    let lf = loaded
        .effective
        .tracing
        .langfuse
        .expect("langfuse config present");
    assert!(lf.enabled);
    assert_eq!(lf.host.as_deref(), Some("https://eu.cloud.langfuse.com"));
    assert_eq!(lf.secret_key.as_deref(), Some("sk-lf-team"));
    assert!(loaded.warnings.is_empty(), "{:?}", loaded.warnings);
}

#[test]
fn compact_soft_ratio_not_below_hard_is_rejected() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[default]
provider = "echo"
model = "m"

[turn]
compact_ratio = 0.6
compact_soft_ratio = 0.7
"#,
    );

    let err = load_config(test_options(&tmp)).expect_err("inverted watermarks");
    match err {
        ConfigError::Invalid { message, .. } => {
            assert!(
                message.contains("compact_soft_ratio") && message.contains("compact_ratio"),
                "unexpected message: {message}"
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn compact_ratio_out_of_range_is_rejected() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[default]
provider = "echo"
model = "m"

[turn]
compact_ratio = 1.5
"#,
    );

    let err = load_config(test_options(&tmp)).expect_err("ratio out of range");
    match err {
        ConfigError::Invalid { message, .. } => {
            assert!(
                message.contains("compact_ratio") && message.contains("(0, 1]"),
                "unexpected message: {message}"
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn valid_three_tier_watermarks_load() {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[default]
provider = "echo"
model = "m"

[turn]
microcompact_ratio = 0.5
compact_soft_ratio = 0.65
compact_ratio = 0.8
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("valid watermarks load");
    assert_eq!(loaded.effective.turn.microcompact_ratio, Some(0.5));
    assert_eq!(loaded.effective.turn.compact_soft_ratio, Some(0.65));
    assert_eq!(loaded.effective.turn.compact_ratio, Some(0.8));
}

#[test]
fn disabled_tier_skips_ordering_check() {
    // soft 比例倒挂，但 background_compact 关闭 → 该档不参与排序约束 → 应放行。
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    write(
        &tmp.path().join("xdg/defect/config.toml"),
        r#"
[default]
provider = "echo"
model = "m"

[turn]
background_compact_enabled = false
compact_soft_ratio = 0.9
compact_ratio = 0.8
"#,
    );

    let loaded = load_config(test_options(&tmp)).expect("disabled tier skips check");
    assert!(!loaded.effective.turn.background_compact_enabled);
}
