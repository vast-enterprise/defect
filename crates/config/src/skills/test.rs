use super::*;

use std::fs;

use tempfile::TempDir;

/// 在 `<skills_dir>/<name>/SKILL.md` 写一个 skill。
fn write_skill(skills_dir: &Path, name: &str, skill_md: &str) {
    let dir = skills_dir.join(name);
    fs::create_dir_all(&dir).expect("mkdir skill");
    fs::write(dir.join("SKILL.md"), skill_md).expect("write SKILL.md");
}

/// 造一个含 .git 的 repo root（让 find_repo_root 命中），返回 repo_root。
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
fn discovers_project_and_user_skills() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);

    write_skill(
        &tmp.path().join("xdg/defect/skills"),
        "user-skill",
        "+++\nname = \"user-skill\"\ndescription = \"a user-layer skill\"\n+++\nUser skill body.\n",
    );
    write_skill(
        &repo_root.join(".defect/skills"),
        "code-review",
        "+++\nname = \"code-review\"\ndescription = \"review Rust diffs\"\n+++\n# Review\n\nDo the thing.\n",
    );

    let skills = discover_skills(&opts_with(&tmp, &repo_root)).expect("discover");
    assert_eq!(skills.len(), 2);
    assert_eq!(skills["user-skill"].description, "a user-layer skill");
    assert_eq!(skills["code-review"].description, "review Rust diffs");
    assert_eq!(skills["code-review"].body, "# Review\n\nDo the thing.");
    assert!(skills["code-review"].dir.ends_with("code-review"));
}

#[test]
fn project_overrides_user_on_name_collision() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);

    write_skill(
        &tmp.path().join("xdg/defect/skills"),
        "dup",
        "+++\nname = \"dup\"\ndescription = \"user version\"\n+++\nuser body\n",
    );
    write_skill(
        &repo_root.join(".defect/skills"),
        "dup",
        "+++\nname = \"dup\"\ndescription = \"project version\"\n+++\nproject body\n",
    );

    let skills = discover_skills(&opts_with(&tmp, &repo_root)).expect("discover");
    assert_eq!(skills.len(), 1);
    assert_eq!(skills["dup"].description, "project version");
    assert_eq!(skills["dup"].body, "project body");
}

#[test]
fn missing_description_is_hard_error() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_skill(
        &repo_root.join(".defect/skills"),
        "bad",
        "+++\nname = \"bad\"\n+++\nbody\n",
    );
    let err = discover_skills(&opts_with(&tmp, &repo_root)).expect_err("must fail");
    assert!(matches!(err, ConfigError::Invalid { .. }));
}

#[test]
fn missing_frontmatter_is_hard_error() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_skill(
        &repo_root.join(".defect/skills"),
        "bad",
        "no frontmatter here\njust text\n",
    );
    let err = discover_skills(&opts_with(&tmp, &repo_root)).expect_err("must fail");
    assert!(matches!(err, ConfigError::Invalid { .. }));
}

#[test]
fn name_mismatch_with_dir_is_hard_error() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_skill(
        &repo_root.join(".defect/skills"),
        "code-review",
        "+++\nname = \"reviewer\"\ndescription = \"d\"\n+++\nbody\n",
    );
    let err = discover_skills(&opts_with(&tmp, &repo_root)).expect_err("must fail");
    match err {
        ConfigError::Invalid { message, .. } => {
            assert!(message.contains("directory name"), "got: {message}");
        }
        other => panic!("expected Invalid, got {other:?}"),
    }
}

#[test]
fn unknown_key_is_hard_error() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_skill(
        &repo_root.join(".defect/skills"),
        "typo",
        "+++\nname = \"typo\"\ndescription = \"d\"\ntirggers = []\n+++\nbody\n",
    );
    let err = discover_skills(&opts_with(&tmp, &repo_root)).expect_err("must fail");
    assert!(matches!(err, ConfigError::Invalid { .. }));
}

#[test]
fn empty_when_no_skills_dirs() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    let skills = discover_skills(&opts_with(&tmp, &repo_root)).expect("discover");
    assert!(skills.is_empty());
}

/// open-standard 字段（always / triggers / allowed_tools）v0 解析但不消费——
/// 写了**不报错**（吃得下 Anthropic / Codex 格式 skill），且不影响 name/desc/body。
#[test]
fn open_standard_fields_parse_but_are_ignored() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_skill(
        &repo_root.join(".defect/skills"),
        "code-review",
        "+++\n\
         name = \"code-review\"\n\
         description = \"review Rust diffs\"\n\
         always = false\n\
         allowed_tools = [\"bash\", \"read_file\"]\n\
         [triggers]\n\
         globs = [\"**/*.rs\"]\n\
         keywords = [\"clippy\", \"lint\"]\n\
         +++\n\
         # Review body\n",
    );
    let skills = discover_skills(&opts_with(&tmp, &repo_root)).expect("must parse");
    let s = &skills["code-review"];
    assert_eq!(s.description, "review Rust diffs");
    assert_eq!(s.body, "# Review body");
}

/// Anthropic 用连字符 `allowed-tools`——alias 应当也吃得下。
#[test]
fn allowed_tools_hyphen_alias_parses() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_skill(
        &repo_root.join(".defect/skills"),
        "x",
        "+++\nname = \"x\"\ndescription = \"d\"\nallowed-tools = [\"bash\"]\n+++\nbody\n",
    );
    let skills = discover_skills(&opts_with(&tmp, &repo_root)).expect("must parse");
    assert!(skills.contains_key("x"));
}

#[test]
fn subdir_without_skill_md_is_skipped() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    let skills_dir = repo_root.join(".defect/skills");
    fs::create_dir_all(skills_dir.join("not-a-skill")).expect("mkdir");
    write_skill(
        &skills_dir,
        "real",
        "+++\nname = \"real\"\ndescription = \"r\"\n+++\nbody\n",
    );

    let skills = discover_skills(&opts_with(&tmp, &repo_root)).expect("discover");
    assert_eq!(skills.len(), 1);
    assert!(skills.contains_key("real"));
}

#[test]
fn non_dir_entry_is_skipped() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    let skills_dir = repo_root.join(".defect/skills");
    fs::create_dir_all(&skills_dir).expect("mkdir");
    // 一个散落的 .md 文件不是 skill（skill 只走 dir-per-skill）。
    fs::write(skills_dir.join("loose.md"), "+++\nname=\"x\"\n+++\n").expect("write");
    write_skill(
        &skills_dir,
        "real",
        "+++\nname = \"real\"\ndescription = \"r\"\n+++\nbody\n",
    );

    let skills = discover_skills(&opts_with(&tmp, &repo_root)).expect("discover");
    assert_eq!(skills.len(), 1);
    assert!(skills.contains_key("real"));
}

// --- YAML frontmatter（--- 分隔，需 yaml feature）------------------------

#[cfg(feature = "yaml")]
#[test]
fn discovers_yaml_frontmatter_skill() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_skill(
        &repo_root.join(".defect/skills"),
        "code-review",
        "---\nname: code-review\ndescription: review Rust diffs\n---\n# Review\n",
    );

    let skills = discover_skills(&opts_with(&tmp, &repo_root)).expect("discover");
    let s = &skills["code-review"];
    assert_eq!(s.description, "review Rust diffs");
    assert_eq!(s.body, "# Review");
}

/// yaml feature 关闭时，`---` 头必须以可操作错误 hard fail（不静默降级）。
#[cfg(not(feature = "yaml"))]
#[test]
fn yaml_frontmatter_without_feature_errors() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_skill(
        &repo_root.join(".defect/skills"),
        "y",
        "---\nname: y\ndescription: d\n---\nbody\n",
    );
    let err = discover_skills(&opts_with(&tmp, &repo_root)).expect_err("must fail without yaml");
    match err {
        ConfigError::Invalid { message, .. } => {
            assert!(message.contains("yaml"), "got: {message}");
        }
        other => panic!("expected Invalid, got {other:?}"),
    }
}
