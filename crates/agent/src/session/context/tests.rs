use std::path::Path;

use super::{Frontend, RunningContext};

#[test]
fn frontend_describe_renders_delegation() {
    assert_eq!(Frontend::Cli.describe(), "CLI");
    assert_eq!(Frontend::Headless.describe(), "headless");
    assert_eq!(
        Frontend::Acp {
            fs_delegated: true,
            shell_delegated: false,
        }
        .describe(),
        "ACP (fs: delegated, shell: local)",
    );
}

#[test]
fn render_includes_all_fields() {
    let ctx = RunningContext::new(
        Frontend::Acp {
            fs_delegated: false,
            shell_delegated: false,
        },
        Path::new("/tmp/work"),
    );
    let out = ctx.render();

    assert!(out.contains("- platform: "));
    assert!(out.contains(&format!("- defect version: {}", env!("CARGO_PKG_VERSION"))));
    assert!(out.contains("- frontend: ACP (fs: local, shell: local)"));
    assert!(out.contains("- cwd: /tmp/work"));
    assert!(out.contains("- shell: "));
}
