//! Fault-tolerant string-match chain for `edit_file`.
//!
//! Ported from opencode's `edit.ts` replacer chain (which itself sources from cline /
//! gemini-cli). The chain tries a sequence of progressively looser matchers; the first
//! one that yields a candidate substring **present in the content** wins.
//!
//! ## Safety invariant
//!
//! Every replacer yields *verbatim slices of the original content* (or the literal
//! `find` for the exact matcher). The orchestrator then locates that slice with
//! [`str::find`] and splices `new` into the exact byte span — it never writes back a
//! "normalized re-render" of the matched region. Looser matchers therefore only widen
//! *what we are willing to locate*; the bytes we replace and the bytes we keep around
//! them are always the real file content. Uniqueness is enforced at every level (a
//! candidate that appears more than once is rejected, never auto-resolved to the first
//! hit), so a fuzzy match can never silently edit the wrong occurrence.
//!
//! The one inherent caveat of the looser levels (LineTrimmed / IndentationFlexible /
//! WhitespaceNormalized ...) is that when the model's `old_string` had the wrong
//! indentation/whitespace, the spliced `new_string` is inserted as-is — so the region's
//! original indentation is not preserved. The matched strategy name is surfaced to the
//! caller so this is observable rather than silent.

/// Outcome of a failed match.
pub(crate) enum EditOutcome {
    NotFound,
    /// Candidate(s) were found but none was unique (≥ 2 occurrences).
    Ambiguous(u32),
}

/// A replacer yields candidate substrings of `content` to look for. Each candidate is
/// either the literal `find` (exact matcher) or a verbatim slice of `content`.
type Replacer = fn(content: &str, find: &str) -> Vec<String>;

/// The ordered chain. Earlier (stricter) matchers take precedence.
const CHAIN: &[(&str, Replacer)] = &[
    ("exact", simple_replacer),
    ("line_trimmed", line_trimmed_replacer),
    ("block_anchor", block_anchor_replacer),
    ("whitespace_normalized", whitespace_normalized_replacer),
    ("indentation_flexible", indentation_flexible_replacer),
    ("escape_normalized", escape_normalized_replacer),
    ("trimmed_boundary", trimmed_boundary_replacer),
    ("context_aware", context_aware_replacer),
    ("multi_occurrence", multi_occurrence_replacer),
];

/// Runs the replacer chain. On success returns `(new_content, matches_replaced,
/// matched_strategy)`. `matched_strategy` is the static name of the level that hit
/// (`"exact"` for the strict path).
///
/// Per-level uniqueness (stricter than opencode): for each level we gather *all* distinct
/// match positions its candidates resolve to in `content`. If exactly one position
/// matches, we splice there. If more than one distinct position matches, the level is
/// **ambiguous** and we fall through (rather than silently editing the first), preserving
/// the "no silent wrong edit" invariant even for fuzzy levels. `replace_all` instead
/// rewrites every occurrence of the matched candidate.
pub(crate) fn replace(
    content: &str,
    old: &str,
    new: &str,
    replace_all: bool,
) -> Result<(String, u32, &'static str), EditOutcome> {
    let mut found_any = false;
    let mut ambiguous_count = 0u32;

    for (name, replacer) in CHAIN {
        // Distinct candidate strings this level proposes that actually occur in content.
        let mut candidates: Vec<String> = Vec::new();
        for search in replacer(content, old) {
            if !search.is_empty() && content.contains(&search) && !candidates.contains(&search) {
                candidates.push(search);
            }
        }
        if candidates.is_empty() {
            continue;
        }
        found_any = true;

        if replace_all {
            // Replace all occurrences. A level proposing more than one distinct candidate
            // string under replace_all is ambiguous (we don't know which family to expand).
            let search = match candidates.split_first() {
                Some((only, [])) => only,
                _ => {
                    // More than one distinct candidate family under replace_all is
                    // ambiguous (we don't know which family to expand).
                    ambiguous_count = ambiguous_count.max(candidates.len() as u32);
                    continue;
                }
            };
            let count = content.matches(search).count() as u32;
            return Ok((content.replace(search, new), count, name));
        }

        // Collect every distinct match position (start byte offset + length) across all
        // candidate strings.
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for search in &candidates {
            let mut from = 0usize;
            while let Some(rel) = content.get(from..).and_then(|s| s.find(search)) {
                let start = from + rel;
                let span = (start, search.len());
                if !spans.contains(&span) {
                    spans.push(span);
                }
                from = start + search.len().max(1);
            }
        }

        match spans.first() {
            None => continue,
            Some(&(start, len)) if spans.len() == 1 => {
                let prefix = content.get(..start).unwrap_or("");
                let suffix = content.get(start + len..).unwrap_or("");
                let mut out = String::with_capacity(prefix.len() + new.len() + suffix.len());
                out.push_str(prefix);
                out.push_str(new);
                out.push_str(suffix);
                return Ok((out, 1, name));
            }
            _ => {
                ambiguous_count = ambiguous_count.max(spans.len() as u32);
                continue;
            }
        }
    }

    if found_any {
        Err(EditOutcome::Ambiguous(ambiguous_count.max(2)))
    } else {
        Err(EditOutcome::NotFound)
    }
}

/// Splits `content` into lines on `'\n'` and returns the verbatim byte-slice spanning
/// lines `[start, end]` (inclusive, 0-based). Assumes each line before `end` was
/// terminated by exactly one `'\n'` (true for [`str::split`] on `'\n'`), so byte offsets
/// reconstruct correctly even for CRLF content (the `'\r'` is part of the line's bytes).
fn line_span(content: &str, lines: &[&str], start: usize, end: usize) -> String {
    let mut begin = 0usize;
    for line in lines.iter().take(start) {
        begin += line.len() + 1;
    }
    let mut stop = begin;
    for (k, line) in lines.iter().enumerate().take(end + 1).skip(start) {
        stop += line.len();
        if k < end {
            stop += 1; // the '\n' separator
        }
    }
    content.get(begin..stop).unwrap_or("").to_string()
}

/// Drops a single trailing empty element produced by a trailing `'\n'` in `find`.
fn trim_trailing_empty(mut lines: Vec<&str>) -> Vec<&str> {
    if lines.last() == Some(&"") {
        lines.pop();
    }
    lines
}

fn simple_replacer(_content: &str, find: &str) -> Vec<String> {
    vec![find.to_string()]
}

fn line_trimmed_replacer(content: &str, find: &str) -> Vec<String> {
    let original: Vec<&str> = content.split('\n').collect();
    let search = trim_trailing_empty(find.split('\n').collect());
    if search.is_empty() || original.len() < search.len() {
        return Vec::new();
    }

    let mut out = Vec::new();
    for i in 0..=(original.len() - search.len()) {
        let matches = original
            .iter()
            .skip(i)
            .zip(search.iter())
            .all(|(o, s)| o.trim() == s.trim());
        if matches {
            out.push(line_span(content, &original, i, i + search.len() - 1));
        }
    }
    out
}

fn block_anchor_replacer(content: &str, find: &str) -> Vec<String> {
    let original: Vec<&str> = content.split('\n').collect();
    let mut search: Vec<&str> = find.split('\n').collect();
    if search.len() < 3 {
        return Vec::new();
    }
    search = trim_trailing_empty(search);
    if search.len() < 3 {
        return Vec::new();
    }

    let first_line = search.first().map(|s| s.trim()).unwrap_or("");
    let last_line = search.last().map(|s| s.trim()).unwrap_or("");
    let search_size = search.len();

    // Collect candidate (start, end) line pairs where both anchors match.
    let mut candidates: Vec<(usize, usize)> = Vec::new();
    for (i, line) in original.iter().enumerate() {
        if line.trim() != first_line {
            continue;
        }
        for (j, cand) in original.iter().enumerate().skip(i + 2) {
            if cand.trim() == last_line {
                candidates.push((i, j));
                break; // only the first matching last-line
            }
        }
    }
    if candidates.is_empty() {
        return Vec::new();
    }

    const SINGLE_THRESHOLD: f64 = 0.0;
    const MULTI_THRESHOLD: f64 = 0.3;

    // Single candidate: relaxed threshold (anchors alone are enough).
    if let [(start, end)] = candidates.as_slice() {
        let (start, end) = (*start, *end);
        let actual_size = end - start + 1;
        let similarity = middle_similarity(&original, &search, start, search_size, actual_size);
        if similarity >= SINGLE_THRESHOLD {
            return vec![line_span(content, &original, start, end)];
        }
        return Vec::new();
    }

    // Multiple candidates: pick the most similar above the stricter threshold.
    let mut best: Option<(usize, usize)> = None;
    let mut max_sim = -1.0f64;
    for &(start, end) in &candidates {
        let actual_size = end - start + 1;
        let sim = middle_similarity(&original, &search, start, search_size, actual_size);
        if sim > max_sim {
            max_sim = sim;
            best = Some((start, end));
        }
    }
    if max_sim >= MULTI_THRESHOLD
        && let Some((start, end)) = best
    {
        return vec![line_span(content, &original, start, end)];
    }
    Vec::new()
}

/// Average per-line similarity (1 - levenshtein/maxlen) of the *middle* lines (excluding
/// the two anchors) between the search block and a candidate block.
fn middle_similarity(
    original: &[&str],
    search: &[&str],
    start: usize,
    search_size: usize,
    actual_size: usize,
) -> f64 {
    let to_check = (search_size as isize - 2).min(actual_size as isize - 2);
    if to_check <= 0 {
        return 1.0; // no middle lines to compare; anchors carry it
    }
    let mut similarity = 0.0;
    let mut j = 1;
    while j < search_size - 1 && j < actual_size - 1 {
        let (Some(o_raw), Some(s_raw)) = (original.get(start + j), search.get(j)) else {
            break;
        };
        let o = o_raw.trim();
        let s = s_raw.trim();
        let max_len = o.chars().count().max(s.chars().count());
        if max_len != 0 {
            let dist = levenshtein(o, s) as f64;
            similarity += (1.0 - dist / max_len as f64) / to_check as f64;
        }
        j += 1;
    }
    similarity
}

fn whitespace_normalized_replacer(content: &str, find: &str) -> Vec<String> {
    let normalize = |t: &str| t.split_whitespace().collect::<Vec<_>>().join(" ");
    let normalized_find = normalize(find);
    let lines: Vec<&str> = content.split('\n').collect();
    let mut out = Vec::new();

    // Single-line: whole line matches after normalization.
    for line in &lines {
        if normalize(line) == normalized_find {
            out.push((*line).to_string());
        }
    }

    // Multi-line block matches after normalization.
    let find_lines: Vec<&str> = find.split('\n').collect();
    if find_lines.len() > 1 && lines.len() >= find_lines.len() {
        for i in 0..=(lines.len() - find_lines.len()) {
            let block = line_span(content, &lines, i, i + find_lines.len() - 1);
            if normalize(&block) == normalized_find {
                out.push(block);
            }
        }
    }
    out
}

fn indentation_flexible_replacer(content: &str, find: &str) -> Vec<String> {
    let remove_indent = |text: &str| -> String {
        let lines: Vec<&str> = text.split('\n').collect();
        let min_indent = lines
            .iter()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.len() - l.trim_start().len())
            .min();
        match min_indent {
            None => text.to_string(),
            Some(min) => lines
                .iter()
                .map(|l| {
                    if l.trim().is_empty() {
                        (*l).to_string()
                    } else {
                        l.get(min..).unwrap_or(l).to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    };

    let normalized_find = remove_indent(find);
    let content_lines: Vec<&str> = content.split('\n').collect();
    let find_lines: Vec<&str> = find.split('\n').collect();
    if content_lines.len() < find_lines.len() {
        return Vec::new();
    }

    let mut out = Vec::new();
    for i in 0..=(content_lines.len() - find_lines.len()) {
        let block = line_span(content, &content_lines, i, i + find_lines.len() - 1);
        if remove_indent(&block) == normalized_find {
            out.push(block);
        }
    }
    out
}

fn escape_normalized_replacer(content: &str, find: &str) -> Vec<String> {
    let unescaped_find = unescape(find);
    let mut out = Vec::new();

    if content.contains(&unescaped_find) {
        out.push(unescaped_find.clone());
    }

    let lines: Vec<&str> = content.split('\n').collect();
    let find_lines: Vec<&str> = unescaped_find.split('\n').collect();
    if lines.len() >= find_lines.len() && !find_lines.is_empty() {
        for i in 0..=(lines.len() - find_lines.len()) {
            let block = line_span(content, &lines, i, i + find_lines.len() - 1);
            if unescape(&block) == unescaped_find {
                out.push(block);
            }
        }
    }
    out
}

/// Resolves backslash escape sequences (`\n \t \r \' \" \` \\ \$`) to their literal
/// characters.
fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('\'') => out.push('\''),
            Some('"') => out.push('"'),
            Some('`') => out.push('`'),
            Some('\\') => out.push('\\'),
            Some('\n') => out.push('\n'),
            Some('$') => out.push('$'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

fn trimmed_boundary_replacer(content: &str, find: &str) -> Vec<String> {
    let trimmed = find.trim();
    if trimmed == find {
        return Vec::new(); // already trimmed; nothing new to try
    }

    let mut out = Vec::new();
    if content.contains(trimmed) {
        out.push(trimmed.to_string());
    }

    let lines: Vec<&str> = content.split('\n').collect();
    let find_lines: Vec<&str> = find.split('\n').collect();
    if lines.len() >= find_lines.len() && !find_lines.is_empty() {
        for i in 0..=(lines.len() - find_lines.len()) {
            let block = line_span(content, &lines, i, i + find_lines.len() - 1);
            if block.trim() == trimmed {
                out.push(block);
            }
        }
    }
    out
}

fn context_aware_replacer(content: &str, find: &str) -> Vec<String> {
    let mut find_lines: Vec<&str> = find.split('\n').collect();
    if find_lines.len() < 3 {
        return Vec::new();
    }
    find_lines = trim_trailing_empty(find_lines);
    if find_lines.len() < 3 {
        return Vec::new();
    }

    let content_lines: Vec<&str> = content.split('\n').collect();
    let first_line = find_lines.first().map(|s| s.trim()).unwrap_or("");
    let last_line = find_lines.last().map(|s| s.trim()).unwrap_or("");

    for (i, start_line) in content_lines.iter().enumerate() {
        if start_line.trim() != first_line {
            continue;
        }
        for (j, cand) in content_lines.iter().enumerate().skip(i + 2) {
            if cand.trim() != last_line {
                continue;
            }
            // Found a potential block: same line count, ≥50% of middle lines match.
            if j - i + 1 == find_lines.len() {
                let mut matching = 0usize;
                let mut total_non_empty = 0usize;
                for (block_line, find_line) in content_lines
                    .iter()
                    .skip(i + 1)
                    .take(find_lines.len().saturating_sub(2))
                    .zip(find_lines.iter().skip(1))
                {
                    let block_line = block_line.trim();
                    let find_line = find_line.trim();
                    if !block_line.is_empty() || !find_line.is_empty() {
                        total_non_empty += 1;
                        if block_line == find_line {
                            matching += 1;
                        }
                    }
                }
                if total_non_empty == 0 || matching as f64 / total_non_empty as f64 >= 0.5 {
                    return vec![line_span(content, &content_lines, i, j)];
                }
            }
            break;
        }
    }
    Vec::new()
}

fn multi_occurrence_replacer(content: &str, find: &str) -> Vec<String> {
    // Yields the exact `find` once if present; the orchestrator's replace_all path then
    // handles every occurrence. (Matches opencode: this level exists so replace_all can
    // act on a plain exact substring even after stricter single-match levels declined.)
    if content.contains(find) {
        vec![find.to_string()]
    } else {
        Vec::new()
    }
}

// Classic two-row Levenshtein. All indices are provably in bounds (loops are `1..=len`
// over freshly-sized rows), so the indexing-panic lint is a false positive on this hot
// inner loop; a local allow keeps the algorithm readable rather than littering `.get()`.
#[allow(clippy::indexing_slicing)]
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests;
