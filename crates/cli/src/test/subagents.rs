//! subagent / `--profile` 装配接线测试。
//!
//! 验证 CLI 装配层（`tools::build_process_tools_with_subagents` /
//! `filter_tools_by_allowlist`）正确把发现到的 profile 暴露成 `spawn_agent`
//! 工具、空 profile 时不暴露、顶层白名单裁剪生效。嵌套 turn 的执行语义由
//! `defect-agent` 侧的 spawn_agent 测试覆盖。

use std::collections::BTreeMap;
use std::fs;
use std::sync::Arc;

use defect_acp::EchoProvider;
use defect_agent::llm::{LlmProvider, ModelInfo, ProviderRegistry};
use defect_agent::policy::{AskWritesPolicy, SandboxPolicy};
use defect_config::{LoadConfigOptions, LoadedConfig, ProfileSpec, discover_profiles, load_config};
use tempfile::TempDir;

use crate::tools::{build_process_tools_with_subagents, filter_tools_by_allowlist};

fn echo_registry() -> Arc<ProviderRegistry> {
    let provider: Arc<dyn LlmProvider> = Arc::new(EchoProvider::new());
    ProviderRegistry::single(
        provider,
        ModelInfo {
            id: "echo-1".to_string(),
            display_name: None,
            context_window: None,
            max_output_tokens: None,
            deprecated: false,
            capabilities_overrides: Default::default(),
        },
    )
}

/// 造一个最小的 repo + 加载默认配置（echo provider）。返回 (tmp, config, opts)。
fn setup() -> (TempDir, LoadedConfig, LoadConfigOptions) {
    let tmp = TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");
    let opts = LoadConfigOptions {
        cwd: repo,
        xdg_config_home: Some(tmp.path().join("xdg")),
        ..LoadConfigOptions::default()
    };
    let config = load_config(opts.clone()).expect("load config");
    (tmp, config, opts)
}

fn write_profile(opts: &LoadConfigOptions, name: &str, config_toml: &str, system_md: &str) {
    let dir = opts.cwd.join(".defect/agents").join(name);
    fs::create_dir_all(&dir).expect("mkdir profile");
    fs::write(dir.join("config.toml"), config_toml).expect("config.toml");
    fs::write(dir.join("system.md"), system_md).expect("system.md");
}

fn discover(opts: &LoadConfigOptions) -> BTreeMap<String, ProfileSpec> {
    discover_profiles(opts).expect("discover")
}

fn policy() -> Arc<dyn SandboxPolicy> {
    Arc::new(AskWritesPolicy::new())
}

#[test]
fn spawn_agent_registered_when_profiles_exist() {
    let (_tmp, config, opts) = setup();
    write_profile(
        &opts,
        "reviewer",
        "description = \"review diffs\"\n",
        "you are reviewer",
    );
    let profiles = discover(&opts);

    let tools =
        build_process_tools_with_subagents(&config, &profiles, &echo_registry(), &policy(), None);
    let names: Vec<String> = tools.schemas().into_iter().map(|s| s.name).collect();
    assert!(names.contains(&"spawn_agent".to_string()), "got: {names:?}");
    // base 工具仍在。
    assert!(names.contains(&"read_file".to_string()));
}

#[test]
fn spawn_agent_absent_when_no_profiles() {
    let (_tmp, config, opts) = setup();
    let profiles = discover(&opts);
    assert!(profiles.is_empty());

    let tools =
        build_process_tools_with_subagents(&config, &profiles, &echo_registry(), &policy(), None);
    let names: Vec<String> = tools.schemas().into_iter().map(|s| s.name).collect();
    assert!(!names.contains(&"spawn_agent".to_string()), "got: {names:?}");
}

#[test]
fn spawn_agent_schema_lists_profile_in_enum() {
    let (_tmp, config, opts) = setup();
    write_profile(
        &opts,
        "reviewer",
        "description = \"review diffs for races\"\n",
        "sys",
    );
    let profiles = discover(&opts);
    let tools =
        build_process_tools_with_subagents(&config, &profiles, &echo_registry(), &policy(), None);

    let schema = tools
        .schemas()
        .into_iter()
        .find(|s| s.name == "spawn_agent")
        .expect("spawn_agent schema");
    assert!(schema.description.contains("review diffs for races"));
    let enum_vals = schema.input_schema["properties"]["profile"]["enum"]
        .as_array()
        .expect("enum");
    assert!(enum_vals.iter().any(|v| v == "reviewer"));
}

#[test]
fn top_level_profile_filters_tools_by_allowlist() {
    let (_tmp, config, opts) = setup();
    write_profile(
        &opts,
        "reader",
        "description = \"reads\"\n[tools]\nallow = [\"read_file\", \"search\"]\n",
        "sys",
    );
    let profiles = discover(&opts);
    let spec = &profiles["reader"];

    // 顶层 --profile：base 裁成白名单子集。
    let base = crate::tools::build_process_tools(&config);
    let filtered = filter_tools_by_allowlist(&base, &spec.tool_allow).expect("filter");
    let names: Vec<String> = filtered.schemas().into_iter().map(|s| s.name).collect();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"read_file".to_string()));
    assert!(names.contains(&"search".to_string()));
    assert!(!names.contains(&"bash".to_string()));
    assert!(!names.contains(&"write_file".to_string()));
}

#[test]
fn top_level_profile_unknown_tool_fails_loud() {
    let base = crate::tools::build_process_tools(&setup().1);
    match filter_tools_by_allowlist(&base, &["nonexistent_tool".to_string()]) {
        Err(name) => assert_eq!(name, "nonexistent_tool"),
        Ok(_) => panic!("expected unknown-tool error"),
    }
}
