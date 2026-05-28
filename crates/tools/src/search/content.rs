//! `mode = content`：grep 文件内容并按 `path:line:text` 渲染。

use std::path::Path;
use std::time::Instant;

use defect_config::SearchToolConfig;
use globset::GlobSet;
use grep_regex::RegexMatcher;
use grep_searcher::{Searcher, Sink, SinkContext, SinkContextKind, SinkFinish, SinkMatch};
use ignore::Walk;
use tokio_util::sync::CancellationToken;

use defect_agent::tool::{ToolError, ToolEvent};

use super::{SearchOutput, display_relative, elapsed_ms, make_completed, truncate_match_line};

#[derive(Debug)]
struct FileBlock {
    relative_path: String,
    /// (line_number, kind, text). kind: 'M' = match, '-' = context.
    lines: Vec<(u64, char, String)>,
    matches_in_file: u32,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    walker: Walk,
    mut searcher: Searcher,
    matcher: RegexMatcher,
    glob: Option<GlobSet>,
    cwd: &Path,
    head_limit: u32,
    cancel: &CancellationToken,
    config: &SearchToolConfig,
    started: Instant,
) -> ToolEvent {
    let mut blocks: Vec<FileBlock> = Vec::new();
    let mut matches_total: u32 = 0;
    let mut files_matched: u32 = 0;
    let mut files_scanned: u64 = 0;
    let mut walked: u64 = 0;
    let mut truncated = false;

    'outer: for entry in walker {
        if cancel.is_cancelled() {
            return ToolEvent::Failed(ToolError::Canceled);
        }
        walked = walked.saturating_add(1);
        if walked > config.max_walk_files {
            truncated = true;
            break;
        }

        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let path = entry.path();
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        // walker 给的是绝对路径；用户的 glob 通常是相对工作区写的
        // （`crates/**/*.rs`），所以三种都试一下，命中即收。与 files
        // 模式（[`super::files::run`]）保持一致。
        if let Some(g) = &glob {
            let rel = path.strip_prefix(cwd).unwrap_or(path);
            let basename = path.file_name();
            let matched = g.is_match(rel)
                || g.is_match(path)
                || basename
                    .map(|n| g.is_match(std::path::Path::new(n)))
                    .unwrap_or(false);
            if !matched {
                continue;
            }
        }

        files_scanned = files_scanned.saturating_add(1);

        let mut sink = ContentSink {
            relative_path: display_relative(cwd, path),
            block: FileBlock {
                relative_path: display_relative(cwd, path),
                lines: Vec::new(),
                matches_in_file: 0,
            },
        };
        // 单文件内的 search 失败（IO / 非 UTF-8 等）跳过——与 ripgrep 行为一致；
        // 不让单个坏文件让整个 search 失败。
        let _ = searcher.search_path(&matcher, path, &mut sink);

        if sink.block.matches_in_file == 0 {
            continue;
        }
        files_matched = files_matched.saturating_add(1);
        matches_total = matches_total.saturating_add(sink.block.matches_in_file);
        blocks.push(sink.block);

        if matches_total >= head_limit {
            // 让最后一个文件的额外 match 也保留（它们在 sink 里已经累积了），
            // 然后 truncate 标记交给下面的 byte 截断逻辑。
            truncated = matches_total > head_limit || walked < u64::MAX;
            // 实际上 head_limit 触达后即可停。
            if matches_total >= head_limit {
                break 'outer;
            }
        }
    }

    // 触达上限（head_limit）时标记 truncated。
    if matches_total >= head_limit {
        truncated = true;
    }

    let (text, kept_matches) = render(&blocks, head_limit, config.max_result_bytes, truncated);
    let truncated = truncated || kept_matches < matches_total;
    let output = SearchOutput {
        mode: "content",
        files_scanned,
        files_matched,
        matches_total: kept_matches,
        truncated,
        elapsed_ms: elapsed_ms(started),
        head_limit,
    };
    make_completed(text, output)
}

struct ContentSink {
    relative_path: String,
    block: FileBlock,
}

impl Sink for ContentSink {
    type Error = std::io::Error;

    fn matched(&mut self, _searcher: &Searcher, mat: &SinkMatch<'_>) -> Result<bool, Self::Error> {
        let line_no = mat.line_number().unwrap_or(0);
        let text = decode_line(mat.bytes());
        // multi_line 模式下 mat.bytes() 可能含多行——按 \n 拆开存成多条 match。
        for (idx, line) in text.split('\n').enumerate() {
            if line.is_empty() && idx > 0 && text.ends_with('\n') {
                continue;
            }
            let display = truncate_match_line(line);
            self.block
                .lines
                .push((line_no.saturating_add(idx as u64), 'M', display));
            self.block.matches_in_file = self.block.matches_in_file.saturating_add(1);
        }
        let _ = &self.relative_path;
        Ok(true)
    }

    fn context(
        &mut self,
        _searcher: &Searcher,
        ctx: &SinkContext<'_>,
    ) -> Result<bool, Self::Error> {
        let line_no = ctx.line_number().unwrap_or(0);
        let text = decode_line(ctx.bytes());
        let kind_char = match ctx.kind() {
            SinkContextKind::Before | SinkContextKind::After => '-',
            SinkContextKind::Other => '-',
        };
        for (idx, line) in text.split('\n').enumerate() {
            if line.is_empty() && idx > 0 && text.ends_with('\n') {
                continue;
            }
            let display = truncate_match_line(line);
            self.block
                .lines
                .push((line_no.saturating_add(idx as u64), kind_char, display));
        }
        Ok(true)
    }

    fn finish(&mut self, _searcher: &Searcher, _finish: &SinkFinish) -> Result<(), Self::Error> {
        Ok(())
    }
}

fn decode_line(bytes: &[u8]) -> String {
    let trimmed = bytes
        .strip_suffix(b"\n")
        .unwrap_or(bytes)
        .strip_suffix(b"\r")
        .unwrap_or_else(|| bytes.strip_suffix(b"\n").unwrap_or(bytes));
    match std::str::from_utf8(trimmed) {
        Ok(s) => s.to_string(),
        Err(_) => String::from_utf8_lossy(trimmed).into_owned(),
    }
}

fn render(
    blocks: &[FileBlock],
    head_limit: u32,
    max_bytes: u64,
    initial_truncated: bool,
) -> (String, u32) {
    if blocks.is_empty() {
        return ("(no matches)".to_string(), 0);
    }
    let mut out = String::new();
    let mut emitted: u32 = 0;
    let mut byte_truncated = false;
    'blocks: for block in blocks {
        let header = format!("{}\n", block.relative_path);
        if would_exceed(&out, &header, max_bytes) {
            byte_truncated = true;
            break;
        }
        out.push_str(&header);
        for (line_no, kind, text) in &block.lines {
            if *kind == 'M' {
                if emitted >= head_limit {
                    byte_truncated = true;
                    break 'blocks;
                }
                emitted = emitted.saturating_add(1);
            }
            let formatted = format!("    L{line_no}: {text}\n");
            if would_exceed(&out, &formatted, max_bytes) {
                byte_truncated = true;
                break 'blocks;
            }
            out.push_str(&formatted);
        }
        out.push('\n');
    }
    let truncated = initial_truncated || byte_truncated;
    if truncated {
        let total_matches: u32 = blocks.iter().map(|b| b.matches_in_file).sum();
        out.push_str(&format!(
            "[truncated; showing {emitted} of {total_matches} matches]\n"
        ));
    }
    (out, emitted)
}

fn would_exceed(current: &str, addition: &str, max_bytes: u64) -> bool {
    (current.len() as u64).saturating_add(addition.len() as u64) > max_bytes
}
