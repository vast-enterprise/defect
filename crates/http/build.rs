//! Inject compile-time metadata: build SHA used as the default `User-Agent` value.
//!
//! Resolution order:
//! 1. Environment variable `DEFECT_HTTP_BUILD_SHA` — for downstream packagers (crates.io
//!    tarballs, Linux distro source packages, Bedrock/Docker image builds) who lack a
//!    `.git` directory and need to explicitly inject a version identifier.
//! 2. `git rev-parse --short=8 HEAD` — in development or monorepo builds, automatically
//!    uses the current commit.
//! 3. Falls back to `"unknown"`.
//!
//! No step errors out; the final value is at least `"unknown"`, making the runtime
//! `User-Agent` readable.

use std::process::Command;

const BUILD_SHA_ENV: &str = "DEFECT_HTTP_BUILD_SHA";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed={BUILD_SHA_ENV}");
    // Also rerun when the git tree changes, using the HEAD file as a sentinel.
    println!("cargo:rerun-if-changed=../../.git/HEAD");

    let sha = resolve_build_sha();
    println!("cargo:rustc-env=DEFECT_HTTP_GIT_SHA={sha}");
}

fn resolve_build_sha() -> String {
    if let Ok(v) = std::env::var(BUILD_SHA_ENV) {
        let trimmed = v.trim();
        if !trimmed.is_empty() {
            return trimmed.to_owned();
        }
    }

    Command::new("git")
        .args(["rev-parse", "--short=8", "HEAD"])
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                String::from_utf8(out.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_owned())
}
