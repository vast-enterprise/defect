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

    // Each section is wrapped in a top-level heading, separated by a Markdown horizontal
    // rule. The headings appear in concatenation order.
    let titles: Vec<&str> = resolved.lines().filter(|l| l.starts_with("# ")).collect();
    assert_eq!(
        titles,
        [
            "# Base Prompt", // from the base file
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

    // Horizontal rules separate the fragments.
    assert!(resolved.contains("\n\n---\n\n"));

    // The `# Environment` section immediately follows the base prompt and precedes the
    // project section.
    let env_at = resolved.find("# Environment").expect("env section");
    let base_at = resolved.find("base file").expect("base section");
    let project_at = resolved.find("repo prompt").expect("project section");
    assert!(base_at < env_at && env_at < project_at);

    // The body content is still present.
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

    // The `Environment` segment is always injected, even without any prompt
    // configuration.
    assert!(resolved.starts_with("# Environment"));
    assert!(resolved.contains("- platform: "));
    assert!(resolved.contains("- defect version: "));
}
