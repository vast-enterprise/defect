//! Compiles a glob pattern into a [`globset::GlobSet`].
//!
//! [`globset::Glob::new`] does not expand brace groups like `{a,b}` — P1 handles that
//! itself. See
//! Glob pattern matching for file name search.

use globset::{Error, Glob, GlobSetBuilder};

pub(super) fn build_globset(pattern: &str) -> Result<globset::GlobSet, Error> {
    let mut builder = GlobSetBuilder::new();
    for expanded in expand_braces(pattern) {
        builder.add(Glob::new(&expanded)?);
    }
    builder.build()
}

/// Expands `src/foo.{ts,tsx}` → `["src/foo.ts", "src/foo.tsx"]`.
///
/// Does not support nested braces — nested braces are treated as literals (let `globset`
/// complain), matching the simplified strategy of claw-code's `expand_braces`.
fn expand_braces(pattern: &str) -> Vec<String> {
    let bytes = pattern.as_bytes();
    let mut depth = 0u32;
    let mut start: Option<usize> = None;
    let mut end: Option<usize> = None;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'{' => {
                depth = depth.saturating_add(1);
                if depth == 1 && start.is_none() {
                    start = Some(i);
                }
            }
            b'}' => {
                if depth == 1 {
                    end = Some(i);
                    break;
                }
                depth = depth.saturating_sub(1);
            }
            _ => {}
        }
    }
    let (Some(s), Some(e)) = (start, end) else {
        return vec![pattern.to_string()];
    };
    let prefix = pattern.get(..s).unwrap_or("");
    let suffix = pattern.get(e + 1..).unwrap_or("");
    let inner = pattern.get(s + 1..e).unwrap_or("");
    if inner.contains('{') {
        return vec![pattern.to_string()];
    }
    let mut out = Vec::new();
    for variant in inner.split(',') {
        let combined = format!("{prefix}{variant}{suffix}");
        out.extend(expand_braces(&combined));
    }
    if out.is_empty() {
        vec![pattern.to_string()]
    } else {
        out
    }
}

#[cfg(test)]
mod tests;
