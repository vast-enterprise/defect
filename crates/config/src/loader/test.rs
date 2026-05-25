use std::fs;
use std::path::Path;

use tempfile::TempDir;

use crate::loader::{dotenv_updates_from_str, load_config};
use crate::overrides::{merge_toml_values, parse_cli_override};
use crate::types::{
    CliOverrides, ConfigError, ConfigWarning, LoadConfigOptions, PROJECT_LOCAL_CONFIG_RELATIVE,
    ProviderKind,
};

fn test_options(root: &TempDir) -> LoadConfigOptions {
    LoadConfigOptions {
        cwd: root.path().join("repo"),
        cli: CliOverrides::default(),
        xdg_config_home: Some(root.path().join("xdg")),
        home_dir: None,
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
fn shared_project_layer_denylist_warns_and_ignores() {
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

    assert_eq!(loaded.effective.cli.provider, ProviderKind::Echo);
    assert_eq!(loaded.warnings.len(), 2);
    assert!(loaded.warnings.iter().any(|warning| matches!(
        warning,
        ConfigWarning::IgnoredProjectKey { key, .. } if key == "default.provider"
    )));
    assert_eq!(loaded.effective.providers.openai.base_url, None);
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
fn warns_on_unknown_keys_with_source_path() {
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

    let loaded = load_config(test_options(&tmp)).expect("load config");

    assert!(loaded.warnings.iter().any(|warning| matches!(
        warning,
        ConfigWarning::UnknownKey { path, key }
            if path == &config_path && key == "default.bogus"
    )));
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
