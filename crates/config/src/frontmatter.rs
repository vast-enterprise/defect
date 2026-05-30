//! 共享的 frontmatter 解析。
//!
//! 单文件 subagent profile（[`crate::profiles`]）与 skill 的 `SKILL.md`
//! （[`crate::skills`]）都用同一套 `+++`(TOML) / `---`(YAML) frontmatter 语法
//! （社区标准，对齐 Anthropic / Codex 的 open-standard 文件形态）。解析逻辑
//! 抽到这里复用——避免两处各写一份 fence 切分（CLAUDE.md 规范 11：不造轮子）。
//!
//! YAML 分支需 `yaml` feature（默认开）；关闭后 `---` 头会以可操作错误
//! hard fail，`+++` 仍可用。

use serde::de::DeserializeOwned;

/// frontmatter 语法。由起始 fence 决定：`+++` ⇒ TOML，`---` ⇒ YAML
/// （社区标准；YAML 需 `yaml` feature）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Frontmatter {
    Toml,
    Yaml,
}

/// 切出 frontmatter（`+++`/`---` 分隔）与其后正文，并报告语法种类。
///
/// 约定：去 BOM/前导空白后，首行必须是 `+++` 或 `---`；到下一行同样的 fence
/// 之间是 frontmatter，其余是 body。不符合则返回 `None`。正文前导/尾随空白
/// 被 trim，便于文本干净。闭合 fence 必须与起始 fence 同种——`+++` 头要
/// `+++` 尾，`---` 头要 `---` 尾。
pub(crate) fn split_frontmatter(contents: &str) -> Option<(Frontmatter, &str, &str)> {
    let rest = contents.trim_start_matches(['\u{feff}']).trim_start(); // 去 BOM + 前导空白
    let (kind, fence) = if rest.starts_with("+++") {
        (Frontmatter::Toml, "+++")
    } else if rest.starts_with("---") {
        (Frontmatter::Yaml, "---")
    } else {
        return None;
    };
    let rest = &rest[fence.len()..];
    // 起始 fence 之后必须紧跟换行。
    let rest = rest
        .strip_prefix('\n')
        .or_else(|| rest.strip_prefix("\r\n"))?;
    // 找闭合 fence（必须独占一行、与起始同种）。
    let close = find_closing_fence(rest, fence)?;
    let frontmatter = &rest[..close.start];
    let body = rest[close.end..].trim();
    Some((kind, frontmatter, body))
}

/// 按 frontmatter 语法把头部文本反序列化成 `T`（字段 schema 与格式无关，
/// `deny_unknown_fields` 对 YAML 同样生效）。YAML 分支在 `yaml` feature 关闭时
/// hard fail，给可操作的重编译提示（fail loud，不静默降级）。
///
/// # Errors
/// 反序列化失败时返回 `Err(message)`，由调用方包成带文件路径的配置错误。
pub(crate) fn parse_frontmatter<T: DeserializeOwned>(
    kind: Frontmatter,
    text: &str,
) -> Result<T, String> {
    match kind {
        Frontmatter::Toml => toml::from_str(text).map_err(|e| e.to_string()),
        #[cfg(feature = "yaml")]
        Frontmatter::Yaml => serde_yaml::from_str(text).map_err(|e| e.to_string()),
        #[cfg(not(feature = "yaml"))]
        Frontmatter::Yaml => Err("YAML frontmatter (`---`) requires the `yaml` feature; \
             rebuild with `--features yaml`, or use `+++` TOML frontmatter"
            .to_string()),
    }
}

/// 在 frontmatter 区域里找独占一行的 `fence`，返回该行（含到行尾换行）的字节
/// 范围 `start..end`，供切分。
struct Fence {
    start: usize,
    end: usize,
}

fn find_closing_fence(s: &str, fence: &str) -> Option<Fence> {
    let mut offset = 0;
    for line in s.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\n', '\r']);
        if trimmed.trim() == fence {
            return Some(Fence {
                start: offset,
                end: offset + line.len(),
            });
        }
        offset += line.len();
    }
    None
}
