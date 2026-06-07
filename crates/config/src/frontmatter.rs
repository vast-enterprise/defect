//! Shared frontmatter parsing.
//!
//! Both single-file subagent profiles ([`crate::profiles`]) and skill `SKILL.md` files
//! ([`crate::skills`]) use the same `+++` (TOML) / `---` (YAML) frontmatter syntax
//! (community standard, aligned with the open-standard file format used by Anthropic /
//! Codex).
//! The parsing logic is extracted here to avoid duplicating fence splitting in two places
//! (CLAUDE.md rule 11: don't reinvent the wheel).
//!
//! The YAML variant requires the `yaml` feature (enabled by default); when disabled,
//! `---`
//! headers hard-fail with an actionable error, while `+++` remains usable.

use serde::de::DeserializeOwned;

/// Frontmatter syntax. Determined by the opening fence: `+++` ⇒ TOML, `---` ⇒ YAML
/// (community standard; YAML requires the `yaml` feature).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Frontmatter {
    Toml,
    Yaml,
}

/// Extract the frontmatter (delimited by `+++`/`---`) and the body, and report the syntax
/// kind.
///
/// Convention: after stripping BOM and leading whitespace, the first line must be `+++`
/// or `---`; the content between that line and the next matching fence is the
/// frontmatter, and the rest is the body. Returns `None` if the format is invalid.
/// Leading/trailing whitespace on the body is trimmed for cleanliness. The closing fence
/// must match the opening fence — `+++` head requires `+++` tail, `---` head requires
/// `---` tail.
pub(crate) fn split_frontmatter(contents: &str) -> Option<(Frontmatter, &str, &str)> {
    let rest = contents.trim_start_matches(['\u{feff}']).trim_start(); // Strip BOM and leading whitespace
    let (kind, fence) = if rest.starts_with("+++") {
        (Frontmatter::Toml, "+++")
    } else if rest.starts_with("---") {
        (Frontmatter::Yaml, "---")
    } else {
        return None;
    };
    let rest = &rest[fence.len()..];
    // The opening fence must be immediately followed by a newline.
    let rest = rest
        .strip_prefix('\n')
        .or_else(|| rest.strip_prefix("\r\n"))?;
    // Find the closing fence (must be on its own line and match the opening fence).
    let close = find_closing_fence(rest, fence)?;
    let frontmatter = &rest[..close.start];
    let body = rest[close.end..].trim();
    Some((kind, frontmatter, body))
}

/// Deserializes the frontmatter text into `T` using the appropriate syntax (field schema
/// is format-agnostic; `deny_unknown_fields` also applies to YAML). When the `yaml`
/// feature is disabled, the YAML branch hard-fails with an actionable recompilation hint
/// (fail loud, no silent degradation).
///
/// # Errors
/// Returns `Err(message)` on deserialization failure; the caller wraps it into a
/// configuration error with the file path.
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

/// Find a `fence` that occupies an entire line within the frontmatter region, and return
/// the byte range `start..end` of that line (including the trailing newline) for slicing.
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
