//! 注入编译期 metadata：build sha 用于 `User-Agent` 默认值。
//!
//! 解析顺序：
//! 1. 环境变量 `DEFECT_HTTP_BUILD_SHA`——下游打包者（crates.io 发布的
//!    tarball、Linux 发行版的 source-package、Bedrock/Docker 镜像构建）
//!    没有 `.git` 时显式注入版本标识用。
//! 2. `git rev-parse --short=8 HEAD`——开发环境 / monorepo 内构建时
//!    自动取当前 commit。
//! 3. `"unknown"` 兜底。
//!
//! 任何一步失败都不报错；最终值至少是 `"unknown"`，运行期 `User-Agent`
//! 可读。

use std::process::Command;

const BUILD_SHA_ENV: &str = "DEFECT_HTTP_BUILD_SHA";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed={BUILD_SHA_ENV}");
    // 在 git tree 变化时也重跑——用 HEAD 文件作锚点。
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
