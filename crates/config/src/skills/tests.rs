use super::*;

use std::fs;

use tempfile::TempDir;

/// Write a skill to `<skills_dir>/<name>/SKILL.md`.
fn write_skill(skills_dir: &Path, name: &str, skill_md: &str) {
    let dir = skills_dir.join(name);
    fs::create_dir_all(&dir).expect("mkdir skill");
    fs::write(dir.join("SKILL.md"), skill_md).expect("write SKILL.md");
}

/// Create a repo root containing a `.git` directory (so that `find_repo_root` matches),
/// and return the path.
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

/// open-standard fields: `always` / `triggers` are consumed; `allowed_tools` is still a
/// placeholder (writing it does not error, and the parser accepts Anthropic /
/// Codex-format skills).
#[test]
fn always_and_triggers_are_consumed() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_skill(
        &repo_root.join(".defect/skills"),
        "code-review",
        "+++\n\
         name = \"code-review\"\n\
         description = \"review Rust diffs\"\n\
         always = true\n\
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
    // always / keywords consumed; globs compiled into a `GlobSet` and can match.
    assert!(s.always);
    assert_eq!(s.triggers.keywords, vec!["clippy", "lint"]);
    let set = s.triggers.globs.as_ref().expect("globs compiled");
    assert!(set.is_match("crates/agent/src/main.rs"));
    assert!(!set.is_match("Cargo.toml"));
}

/// No `[triggers]` table → empty triggers (globs=None, keywords empty), always=false.
#[test]
fn no_triggers_defaults_to_empty() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_skill(
        &repo_root.join(".defect/skills"),
        "plain",
        "+++\nname = \"plain\"\ndescription = \"d\"\n+++\nbody\n",
    );
    let skills = discover_skills(&opts_with(&tmp, &repo_root)).expect("must parse");
    let s = &skills["plain"];
    assert!(!s.always);
    assert!(s.triggers.globs.is_none());
    assert!(s.triggers.keywords.is_empty());
}

/// Invalid trigger globs must hard-fail at parse time (fail loud, not silently
/// swallowed).
#[test]
fn invalid_trigger_glob_is_hard_error() {
    let tmp = TempDir::new().expect("tmp");
    let repo_root = repo(&tmp);
    write_skill(
        &repo_root.join(".defect/skills"),
        "bad",
        "+++\nname = \"bad\"\ndescription = \"d\"\n[triggers]\nglobs = [\"[unclosed\"]\n+++\nbody\n",
    );
    let err = discover_skills(&opts_with(&tmp, &repo_root)).expect_err("must fail");
    match err {
        ConfigError::Invalid { message, .. } => {
            assert!(message.contains("glob"), "got: {message}");
        }
        other => panic!("expected Invalid, got {other:?}"),
    }
}

/// Anthropic uses a hyphen in `allowed-tools` — the alias should also be accepted.
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
    // A loose .md file is not a skill (skills are dir-per-skill only).
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

// --- YAML frontmatter (delimited by `---`, requires the `yaml` feature) ---

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

/// When the `yaml` feature is disabled, `---` frontmatter must hard-fail with an
/// actionable error (no silent degradation).
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
