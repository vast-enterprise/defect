use std::fs;
use std::path::Path;

use crate::session::context::{Frontend, RunningContext};
use crate::session::{BasePromptConfig, PromptConfig, resolve_system_prompt};

fn write(path: &std::path::Path, body: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("parent dirs");
    }
    fs::write(path, body).expect("write file");
}

fn ctx(cwd: &Path) -> RunningContext<'_> {
    RunningContext::new(
        Frontend::Acp {
            fs_delegated: false,
            shell_delegated: false,
        },
        cwd,
    )
}

#[test]
fn resolves_layers_in_order_with_headings_and_rules() {
    let tmp = tempfile::TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    let cwd = repo.join("apps/web");
    fs::create_dir_all(repo.join(".git")).expect("git");
    fs::create_dir_all(&cwd).expect("cwd");
    write(&repo.join("AGENTS.md"), "repo prompt");
    write(&cwd.join("AGENTS.md"), "cwd prompt");
    write(&repo.join("prompts/base.md"), "base file");

    let prompt = PromptConfig {
        file: "AGENTS.md".to_owned(),
        text: Some("user prompt".to_owned()),
        provider_overlays: [("deepseek".to_owned(), "provider overlay".to_owned())].into(),
        model_overlays: [("deepseek-v4-pro".to_owned(), "model overlay".to_owned())].into(),
    };
    let base_prompt = BasePromptConfig {
        file: Some(repo.join("prompts/base.md")),
        text: Some("base text".to_owned()),
    };

    let resolved = resolve_system_prompt(
        &ctx(&cwd),
        "deepseek",
        "deepseek-v4-pro",
        &base_prompt,
        &prompt,
        Some("session overlay"),
    )
    .expect("resolve")
    .expect("system prompt");

    // 各段套一级标题、以 markdown 水平线相隔。标题出现顺序即拼接顺序。
    let titles: Vec<&str> = resolved.lines().filter(|l| l.starts_with("# ")).collect();
    assert_eq!(
        titles,
        [
            "# Base Prompt", // base file
            "# Base Prompt", // base text
            "# Environment",
            "# System Instructions", // prompt.text
            "# Project Instructions (AGENTS.md)",
            "# Project Instructions (apps/web/AGENTS.md)",
            "# Provider Notes (deepseek)",
            "# Model Notes (deepseek-v4-pro)",
            "# Session Instructions",
        ]
    );

    // 片段之间是水平分割线。
    assert!(resolved.contains("\n\n---\n\n"));

    // Environment 段紧跟 base prompt 之后、project 之前。
    let env_at = resolved.find("# Environment").expect("env section");
    let base_at = resolved.find("base file").expect("base section");
    let project_at = resolved.find("repo prompt").expect("project section");
    assert!(base_at < env_at && env_at < project_at);

    // 正文内容仍在。
    assert!(resolved.contains("provider overlay"));
    assert!(resolved.contains("session overlay"));
    assert!(resolved.contains("- frontend: ACP (fs: local, shell: local)"));
}

#[test]
fn emits_environment_even_without_prompts() {
    let tmp = tempfile::TempDir::new().expect("tmp");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git")).expect("git");

    let resolved = resolve_system_prompt(
        &ctx(&repo),
        "openai",
        "gpt-4o-mini",
        &BasePromptConfig::default(),
        &PromptConfig::default(),
        None,
    )
    .expect("resolve")
    .expect("environment always present");

    // 即便没有任何 prompt 配置，Environment 段始终注入。
    assert!(resolved.starts_with("# Environment"));
    assert!(resolved.contains("- platform: "));
    assert!(resolved.contains("- defect version: "));
}
