//! Tests for subagent / `--profile` assembly wiring.
//!
//! Verifies that the CLI assembly layer (`tools::build_process_tools_with_subagents` /
//! `filter_tools_by_allowlist`) correctly exposes discovered profiles as `spawn_agent`
//! tools, does not expose them when the profile is empty, and that top-level allowlist
//! filtering takes effect. Nested turn execution semantics are covered by the
//! `spawn_agent` tests on the `defect-agent` side.

use std::collections::BTreeMap;
use std::fs;
use std::sync::Arc;

use defect_acp::EchoProvider;
use defect_agent::hooks::builtin::BuiltinRegistry;
use defect_agent::llm::{LlmProvider, ModelInfo, ProviderRegistry};
use defect_agent::policy::{AskWritesPolicy, SandboxPolicy};
use defect_agent::tool::SkillEntry;
use defect_config::{
    LoadConfigOptions, LoadedConfig, ProfileSpec, discover_profiles, discover_skills, load_config,
};
use tempfile::TempDir;

use crate::hooks::HookEngineCtx;
use crate::tools::{build_process_tools_with_subagents, filter_tools_by_allowlist, project_skills};

/// Most subagent tests don't involve skills — pass an empty index.
fn no_skills() -> BTreeMap<String, SkillEntry> {
    BTreeMap::new()
}

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

/// Create a minimal repo and load the default config (echo provider). Returns (tmp,
/// config, opts).
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

fn write_skill(opts: &LoadConfigOptions, name: &str, skill_md: &str) {
    let dir = opts.cwd.join(".defect/skills").join(name);
    fs::create_dir_all(&dir).expect("mkdir skill");
    fs::write(dir.join("SKILL.md"), skill_md).expect("SKILL.md");
}

fn discover_skill_index(opts: &LoadConfigOptions) -> BTreeMap<String, SkillEntry> {
    project_skills(&discover_skills(opts).expect("discover skills"))
}

fn policy() -> Arc<dyn SandboxPolicy> {
    Arc::new(AskWritesPolicy::new())
}

/// Test wrapper: assembles the toolset using the default builtin registry and echo model
/// context, then unwraps.
/// Any `[hooks]` assembly errors in the profile will panic here (test profiles never
/// include hooks, or
/// assembly failures are tested explicitly by calling the lower-level function directly).
fn assemble(
    config: &LoadedConfig,
    profiles: &BTreeMap<String, ProfileSpec>,
    skills: &BTreeMap<String, SkillEntry>,
    registry: &Arc<ProviderRegistry>,
    policy: &Arc<dyn SandboxPolicy>,
    base_prompt: Option<String>,
) -> Arc<dyn defect_agent::session::ToolRegistry> {
    let builtins = BuiltinRegistry::defaults();
    let hook_rt = HookEngineCtx {
        registry,
        default_model: "echo-1",
    };
    build_process_tools_with_subagents(
        config,
        profiles,
        skills,
        registry,
        policy,
        base_prompt,
        &builtins,
        &hook_rt,
    )
    .expect("assemble tools")
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

    let tools = assemble(
        &config,
        &profiles,
        &no_skills(),
        &echo_registry(),
        &policy(),
        None,
    );
    let names: Vec<String> = tools.schemas().into_iter().map(|s| s.name).collect();
    assert!(names.contains(&"spawn_agent".to_string()), "got: {names:?}");
    // Base tools are still present.
    assert!(names.contains(&"read_file".to_string()));
}

#[test]
fn spawn_agent_absent_when_no_profiles() {
    let (_tmp, config, opts) = setup();
    let profiles = discover(&opts);
    assert!(profiles.is_empty());

    let tools = assemble(
        &config,
        &profiles,
        &no_skills(),
        &echo_registry(),
        &policy(),
        None,
    );
    let names: Vec<String> = tools.schemas().into_iter().map(|s| s.name).collect();
    assert!(
        !names.contains(&"spawn_agent".to_string()),
        "got: {names:?}"
    );
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
    let tools = assemble(
        &config,
        &profiles,
        &no_skills(),
        &echo_registry(),
        &policy(),
        None,
    );

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

    // Top-level --profile: base trimmed to an allowlist subset.
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

#[test]
fn skill_tool_registered_when_skills_exist() {
    let (_tmp, config, opts) = setup();
    write_skill(
        &opts,
        "code-review",
        "+++\nname = \"code-review\"\ndescription = \"review Rust diffs\"\n+++\nbody\n",
    );
    let skills = discover_skill_index(&opts);
    let profiles = discover(&opts);

    let tools = assemble(
        &config,
        &profiles,
        &skills,
        &echo_registry(),
        &policy(),
        None,
    );
    let schema = tools
        .schemas()
        .into_iter()
        .find(|s| s.name == "skill")
        .expect("skill schema");
    assert!(schema.description.contains("review Rust diffs"));
    let enum_vals = schema.input_schema["properties"]["name"]["enum"]
        .as_array()
        .expect("enum");
    assert!(enum_vals.iter().any(|v| v == "code-review"));
    // Base tools still present; no profile → spawn_agent not attached.
    let names: Vec<String> = tools.schemas().into_iter().map(|s| s.name).collect();
    assert!(names.contains(&"read_file".to_string()));
    assert!(
        !names.contains(&"spawn_agent".to_string()),
        "got: {names:?}"
    );
}

#[test]
fn skill_tool_absent_when_no_skills() {
    let (_tmp, config, opts) = setup();
    let skills = discover_skill_index(&opts);
    assert!(skills.is_empty());

    let tools = assemble(
        &config,
        &discover(&opts),
        &skills,
        &echo_registry(),
        &policy(),
        None,
    );
    let names: Vec<String> = tools.schemas().into_iter().map(|s| s.name).collect();
    assert!(!names.contains(&"skill".to_string()), "got: {names:?}");
}

/// When a skill is detected, the main session engine automatically mounts
/// `skill-manifest` (after session enter, injecting the L1 manifest + always-on body) and
/// `skill-triggers` (before ingest, activated by prompt) — no manual entry in `[hooks]`
/// is needed.
#[test]
fn main_session_auto_mounts_skill_hooks() {
    use defect_agent::hooks::HookCtx;
    use defect_agent::hooks::step::{AfterSessionEnter, BeforeIngest, IngestSource, SessionSource};

    let (_tmp, config, opts) = setup();
    write_skill(
        &opts,
        "style",
        "+++\nname = \"style\"\ndescription = \"coding style\"\nalways = true\n+++\nALWAYS USE TABS\n",
    );
    write_skill(
        &opts,
        "sql",
        "+++\nname = \"sql\"\ndescription = \"sql help\"\n[triggers]\nglobs = [\"**/*.sql\"]\n+++\nSQL body\n",
    );
    let skills = Arc::new(discover_skill_index(&opts));
    assert_eq!(skills.len(), 2);

    let builtins = BuiltinRegistry::defaults();
    let registry = echo_registry();
    let hook_rt = HookEngineCtx {
        registry: &registry,
        default_model: "echo-1",
    };
    let engine = crate::hooks::build_main_session_engine(
        &config.effective.hooks,
        &builtins,
        &hook_rt,
        &skills,
        None,
    )
    .expect("engine");

    let session_id = agent_client_protocol_schema::SessionId::new("s1");
    let cwd = std::path::Path::new("/");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("rt");

    // after_session_enter: injects the L1 manifest and always-on body.
    let mut enter = AfterSessionEnter {
        cwd: "/".to_string(),
        source: SessionSource::New,
        additional_context: Vec::new(),
    };
    rt.block_on(engine.dispatch(
        &mut enter,
        HookCtx::new(&session_id, cwd, tokio_util::sync::CancellationToken::new()),
    ));
    let injected = enter
        .additional_context
        .iter()
        .filter_map(|b| match b {
            agent_client_protocol_schema::ContentBlock::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(injected.contains("**style**"), "L1 manifest: {injected}");
    assert!(
        injected.contains("ALWAYS USE TABS"),
        "always-on body missing"
    );

    // When the prompt mentions a `.sql` path, the SQL skill is prepended before
    // ingestion.
    let mut ingest = BeforeIngest {
        source: IngestSource::User,
        input: vec![agent_client_protocol_schema::ContentBlock::from(
            "please edit migrations/0001.sql",
        )],
    };
    rt.block_on(engine.dispatch(
        &mut ingest,
        HookCtx::new(&session_id, cwd, tokio_util::sync::CancellationToken::new()),
    ));
    // Original block retained, with a SQL hint prepended.
    let texts: Vec<&str> = ingest
        .input
        .iter()
        .filter_map(|b| match b {
            agent_client_protocol_schema::ContentBlock::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect();
    assert!(texts.iter().any(|t| t.contains("`sql`")), "got: {texts:?}");
    assert!(
        texts.iter().any(|t| t.contains("migrations/0001.sql")),
        "original prompt dropped: {texts:?}"
    );
}
