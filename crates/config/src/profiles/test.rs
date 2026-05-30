use super::*;

use std::fs;

use tempfile::TempDir;

/// 在 `<root>/.config/defect/agents/<name>/` (user) 或
/// `<root>/proj/.defect/agents/<name>/` (project) 写一个 profile。
fn write_profile(agents_dir: &Path, name: &str, config_toml: &str, system_md: Option<&str>) {
    let dir = agents_dir.join(name);
    fs::create_dir_all(&dir).expect("mkdir profile");
    fs::write(dir.join("config.toml"), config_toml).expect("write config.toml");
    if let Some(md) = system_md {
        fs::write(dir.join("system.md"), md).expect("write system.md");
    }
}

/// 在 `<agents_dir>/<name>.md` 写一个单文件版 profile。
fn write_single_file(agents_dir: &Path, name: &str, contents: &str) {
    fs::create_dir_all(agents_dir).expect("mkdir agents");
    fs::write(agents_dir.join(format!("{name}.md")), contents).expect("write .md");
}

/// 造一个含 .git 的 repo root（让 find_repo_root 命中），返回 (tmp, repo_root)。
fn repo(tmp: &TempDir) -> PathBuf {
    let root = tmp.path().join("proj");
    fs::create_dir_all(root.join(".git")).expect("mkdir .git");
    root
}

fn opts_with(tmp: &TempDir, repo_root: &Path) -> LoadConfigOptions {
    LoadConfigOptions {
        cwd: repo_root.to_path_buf(),
        xdg_config_home: Some(tmp.path().join("xdg")),
        ..LoadConfigOptions::default()
    }
}

#[test]
fn discovers_project_and_user_profiles() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);

    let user_agents = tmp.path().join("xdg/defect/agents");
    write_profile(
        &user_agents,
        "userbot",
        "description = \"a user-layer agent\"\n",
        Some("you are userbot"),
    );

    let project_agents = repo_root.join(".defect/agents");
    write_profile(
        &project_agents,
        "reviewer",
        "description = \"review diffs\"\n[tools]\nallow = [\"read_file\"]\n",
        Some("you are reviewer"),
    );

    let profiles = discover_profiles(&opts_with(&tmp, &repo_root)).expect("discover");
    assert_eq!(profiles.len(), 2);
    assert_eq!(profiles["userbot"].description, "a user-layer agent");
    assert_eq!(profiles["reviewer"].tool_allow, vec!["read_file"]);
    assert_eq!(profiles["reviewer"].system_prompt_text, "you are reviewer");
}

#[test]
fn project_overrides_user_on_name_collision() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);

    write_profile(
        &tmp.path().join("xdg/defect/agents"),
        "bot",
        "description = \"user version\"\n",
        Some("user prompt"),
    );
    write_profile(
        &repo_root.join(".defect/agents"),
        "bot",
        "description = \"project version\"\n",
        Some("project prompt"),
    );

    let profiles = discover_profiles(&opts_with(&tmp, &repo_root)).expect("discover");
    assert_eq!(profiles.len(), 1);
    assert_eq!(profiles["bot"].description, "project version");
    assert_eq!(profiles["bot"].system_prompt_text, "project prompt");
}

#[test]
fn missing_description_is_hard_error() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_profile(
        &repo_root.join(".defect/agents"),
        "bad",
        "model = \"x\"\n",
        Some("prompt"),
    );

    let err = discover_profiles(&opts_with(&tmp, &repo_root)).expect_err("must fail");
    assert!(matches!(err, ConfigError::Invalid { .. }));
}

#[test]
fn allow_omitted_defaults_to_read_only_set() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_profile(
        &repo_root.join(".defect/agents"),
        "reader",
        "description = \"reads\"\n",
        Some("prompt"),
    );

    let profiles = discover_profiles(&opts_with(&tmp, &repo_root)).expect("discover");
    assert_eq!(profiles["reader"].tool_allow, vec!["read_file", "search"]);
}

#[test]
fn unknown_key_is_hard_error() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_profile(
        &repo_root.join(".defect/agents"),
        "typo",
        "description = \"x\"\nmdoel = \"oops\"\n",
        Some("prompt"),
    );

    let err = discover_profiles(&opts_with(&tmp, &repo_root)).expect_err("must fail");
    assert!(matches!(err, ConfigError::Invalid { .. }));
}

#[test]
fn prompt_file_escaping_profile_dir_is_rejected() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    // 在 repo root 放一个 secret，profile 想用 ../../secret.md 偷读。
    fs::write(repo_root.join("secret.md"), "TOPSECRET").expect("write secret");
    write_profile(
        &repo_root.join(".defect/agents"),
        "escaper",
        "description = \"x\"\n[prompt]\nfile = \"../../secret.md\"\n",
        Some("decoy"),
    );

    let err = discover_profiles(&opts_with(&tmp, &repo_root)).expect_err("must reject escape");
    match err {
        ConfigError::Invalid { message, .. } => {
            assert!(message.contains("prompt.file"), "got: {message}");
        }
        other => panic!("expected Invalid, got {other:?}"),
    }
}

#[test]
fn empty_when_no_agents_dirs() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    let profiles = discover_profiles(&opts_with(&tmp, &repo_root)).expect("discover");
    assert!(profiles.is_empty());
}

#[test]
fn subdir_without_config_toml_is_skipped() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    let agents = repo_root.join(".defect/agents");
    fs::create_dir_all(agents.join("not-a-profile")).expect("mkdir");
    write_profile(&agents, "real", "description = \"r\"\n", Some("p"));

    let profiles = discover_profiles(&opts_with(&tmp, &repo_root)).expect("discover");
    assert_eq!(profiles.len(), 1);
    assert!(profiles.contains_key("real"));
}

// --- 单文件版（+++ TOML frontmatter）-----------------------------------

#[test]
fn discovers_single_file_profile() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_single_file(
        &repo_root.join(".defect/agents"),
        "reviewer",
        "+++\ndescription = \"review diffs\"\nmodel = \"opus\"\n[tools]\nallow = [\"read_file\"]\n+++\nYou are a reviewer.\n",
    );

    let profiles = discover_profiles(&opts_with(&tmp, &repo_root)).expect("discover");
    assert_eq!(profiles.len(), 1);
    let p = &profiles["reviewer"];
    assert_eq!(p.description, "review diffs");
    assert_eq!(p.model.as_deref(), Some("opus"));
    assert_eq!(p.tool_allow, vec!["read_file"]);
    assert_eq!(p.system_prompt_text, "You are a reviewer.");
}

#[test]
fn single_file_allow_omitted_defaults_read_only() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_single_file(
        &repo_root.join(".defect/agents"),
        "reader",
        "+++\ndescription = \"reads\"\n+++\nbody\n",
    );
    let profiles = discover_profiles(&opts_with(&tmp, &repo_root)).expect("discover");
    assert_eq!(profiles["reader"].tool_allow, vec!["read_file", "search"]);
}

#[test]
fn single_file_missing_frontmatter_errors() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_single_file(
        &repo_root.join(".defect/agents"),
        "bad",
        "no frontmatter here\njust text\n",
    );
    let err = discover_profiles(&opts_with(&tmp, &repo_root)).expect_err("must fail");
    assert!(matches!(err, ConfigError::Invalid { .. }));
}

#[test]
fn single_file_missing_description_errors() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_single_file(
        &repo_root.join(".defect/agents"),
        "bad",
        "+++\nmodel = \"x\"\n+++\nbody\n",
    );
    let err = discover_profiles(&opts_with(&tmp, &repo_root)).expect_err("must fail");
    assert!(matches!(err, ConfigError::Invalid { .. }));
}

#[test]
fn single_file_prompt_table_is_rejected() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_single_file(
        &repo_root.join(".defect/agents"),
        "bad",
        "+++\ndescription = \"d\"\n[prompt]\nfile = \"x.md\"\n+++\nbody\n",
    );
    let err = discover_profiles(&opts_with(&tmp, &repo_root)).expect_err("must reject [prompt]");
    match err {
        ConfigError::Invalid { message, .. } => {
            assert!(message.contains("[prompt]"), "got: {message}");
        }
        other => panic!("expected Invalid, got {other:?}"),
    }
}

#[test]
fn single_file_unknown_key_errors() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_single_file(
        &repo_root.join(".defect/agents"),
        "typo",
        "+++\ndescription = \"d\"\nmdoel = \"oops\"\n+++\nbody\n",
    );
    let err = discover_profiles(&opts_with(&tmp, &repo_root)).expect_err("must fail");
    assert!(matches!(err, ConfigError::Invalid { .. }));
}

#[test]
fn folder_and_single_file_same_name_same_layer_conflicts() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    let agents = repo_root.join(".defect/agents");
    write_profile(&agents, "dup", "description = \"folder\"\n", Some("folder prompt"));
    write_single_file(&agents, "dup", "+++\ndescription = \"file\"\n+++\nfile prompt\n");

    let err = discover_profiles(&opts_with(&tmp, &repo_root)).expect_err("must conflict");
    match err {
        ConfigError::Invalid { message, .. } => {
            assert!(message.contains("duplicate"), "got: {message}");
        }
        other => panic!("expected Invalid, got {other:?}"),
    }
}

#[test]
fn single_file_project_overrides_user() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_single_file(
        &tmp.path().join("xdg/defect/agents"),
        "bot",
        "+++\ndescription = \"user\"\n+++\nuser prompt\n",
    );
    write_single_file(
        &repo_root.join(".defect/agents"),
        "bot",
        "+++\ndescription = \"project\"\n+++\nproject prompt\n",
    );
    let profiles = discover_profiles(&opts_with(&tmp, &repo_root)).expect("discover");
    assert_eq!(profiles.len(), 1);
    assert_eq!(profiles["bot"].description, "project");
    assert_eq!(profiles["bot"].system_prompt_text, "project prompt");
}

// --- 单文件版 YAML frontmatter（--- 分隔，需 yaml feature）---------------

#[cfg(feature = "yaml")]
#[test]
fn discovers_yaml_frontmatter_profile() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_single_file(
        &repo_root.join(".defect/agents"),
        "reviewer",
        "---\ndescription: review diffs\nmodel: opus\ntools:\n  allow: [read_file, search]\n---\nYou are a reviewer.\n",
    );

    let profiles = discover_profiles(&opts_with(&tmp, &repo_root)).expect("discover");
    let p = &profiles["reviewer"];
    assert_eq!(p.description, "review diffs");
    assert_eq!(p.model.as_deref(), Some("opus"));
    assert_eq!(p.tool_allow, vec!["read_file", "search"]);
    assert_eq!(p.system_prompt_text, "You are a reviewer.");
}

#[cfg(feature = "yaml")]
#[test]
fn yaml_unknown_key_errors() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_single_file(
        &repo_root.join(".defect/agents"),
        "typo",
        "---\ndescription: d\nmdoel: oops\n---\nbody\n",
    );
    let err = discover_profiles(&opts_with(&tmp, &repo_root)).expect_err("must fail");
    assert!(matches!(err, ConfigError::Invalid { .. }));
}

#[cfg(feature = "yaml")]
#[test]
fn yaml_prompt_table_is_rejected() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_single_file(
        &repo_root.join(".defect/agents"),
        "bad",
        "---\ndescription: d\nprompt:\n  file: x.md\n---\nbody\n",
    );
    let err = discover_profiles(&opts_with(&tmp, &repo_root)).expect_err("must reject prompt");
    match err {
        ConfigError::Invalid { message, .. } => {
            assert!(message.contains("[prompt]"), "got: {message}");
        }
        other => panic!("expected Invalid, got {other:?}"),
    }
}

/// yaml feature 关闭时，`---` 头必须以可操作错误 hard fail（不静默降级）。
#[cfg(not(feature = "yaml"))]
#[test]
fn yaml_frontmatter_without_feature_errors() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_single_file(
        &repo_root.join(".defect/agents"),
        "y",
        "---\ndescription: d\n---\nbody\n",
    );
    let err = discover_profiles(&opts_with(&tmp, &repo_root)).expect_err("must fail without yaml");
    match err {
        ConfigError::Invalid { message, .. } => {
            assert!(message.contains("yaml"), "got: {message}");
        }
        other => panic!("expected Invalid, got {other:?}"),
    }
}
