//! `search` 内置工具：在 workspace 内 grep 文件内容（content mode）或
//! 列出匹配 glob 的文件（files mode）。
//!
//! 设计与取舍详见 `docs/internal/tools-search.md`。

use std::cmp::Reverse;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::time::{Instant, SystemTime};

use agent_client_protocol_schema::{
    Content, ContentBlock, TextContent, ToolCallContent, ToolCallLocation, ToolCallUpdateFields,
    ToolKind,
};
use defect_agent::error::BoxError;
use defect_agent::tool::{
    SafetyClass, Tool, ToolCallDescription, ToolContext, ToolError, ToolEvent, ToolSchema,
    ToolStream,
};
use defect_config::SearchToolConfig;
use futures::future::BoxFuture;
use futures::stream;
use grep_regex::RegexMatcherBuilder;
use grep_searcher::{BinaryDetection, SearcherBuilder};
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_util::sync::CancellationToken;

mod content;
mod files;
mod glob;

#[cfg(test)]
mod tests;

const TITLE_TRUNC: usize = 80;
const MAX_MATCH_LINE: usize = 4 * 1024;

/// `search` 工具的内置实现。无运行时状态——参数化的 schema 与上限在构造时
/// 固化。
pub struct SearchTool {
    schema: ToolSchema,
    config: SearchToolConfig,
}

impl SearchTool {
    /// 用 [`SearchToolConfig::default`] 构造。
    pub fn new() -> Self {
        Self::from_config(&SearchToolConfig::default())
    }

    /// 按 [`SearchToolConfig`] 构造。`max_head_limit` 会反映在 schema 的
    /// `head_limit` 上限里。
    pub fn from_config(config: &SearchToolConfig) -> Self {
        let default_head_limit = config.default_head_limit.max(1);
        let max_head_limit = config.max_head_limit.max(default_head_limit);
        let mut effective = config.clone();
        effective.default_head_limit = default_head_limit;
        effective.max_head_limit = max_head_limit;

        let description = format!(
            "Search the workspace. \
             In `content` mode (default) runs a regex over file contents and returns \
             matching lines as `<path> / L<line>: <text>`; \
             in `files` mode lists workspace files matching a glob pattern. \
             Respects .gitignore by default; binary files are skipped in content mode. \
             Results are truncated at `head_limit` (default {default_head_limit}; max {max_head_limit}); \
             files-mode results are sorted by mtime (newest first)."
        );

        let schema = ToolSchema {
            name: "search".to_string(),
            description,
            input_schema: json!({
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["content", "files"],
                        "description": "`content` greps file contents (regex over `pattern`); \
                                        `files` lists files matching `pattern` as a glob. \
                                        Defaults to `content`."
                    },
                    "pattern": {
                        "type": "string",
                        "description": "**Required.** What to search for. \
                                        In `content` mode (default): a Rust regex (RE2 syntax) — e.g. `\"pub struct \"`, `\"TODO|FIXME\"`. \
                                        In `files` mode: a glob — e.g. `\"**/*.rs\"`, `\"src/**/foo.{ts,tsx}\"`. \
                                        To narrow which files content-mode scans, use `path_glob` (not this field)."
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional sub-path under the workspace root. \
                                        Relative paths resolve against the session cwd. \
                                        Must resolve inside the workspace."
                    },
                    "path_glob": {
                        "type": "string",
                        "description": "Content mode only. Optional glob restricting **which files** to scan \
                                        (e.g. `**/*.rs`). This selects the file set; `pattern` is the regex \
                                        applied to their contents. Ignored in `files` mode—use `pattern` directly."
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Content mode only. Defaults to false."
                    },
                    "multiline": {
                        "type": "boolean",
                        "description": "Content mode only. Lets `.` and the regex engine span line breaks. \
                                        Defaults to false."
                    },
                    "before": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 50,
                        "description": "Content mode only. Number of context lines before each match (like grep -B)."
                    },
                    "after": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 50,
                        "description": "Content mode only. Number of context lines after each match (like grep -A)."
                    },
                    "head_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": max_head_limit as i64,
                        "description": format!(
                            "Maximum number of matches (content mode) or files (files mode) to return. \
                             Defaults to {default_head_limit}; clamped at {max_head_limit}."
                        )
                    },
                    "respect_gitignore": {
                        "type": "boolean",
                        "description": "When true (default) honors .gitignore / .ignore / hidden-file rules. \
                                        Set to false to search the full tree."
                    }
                },
                "required": ["pattern"]
            }),
        };
        Self {
            schema,
            config: effective,
        }
    }
}

impl Default for SearchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SearchMode {
    #[default]
    Content,
    Files,
}

#[derive(Debug, Deserialize)]
struct SearchArgs {
    pattern: String,
    #[serde(default)]
    mode: Option<SearchMode>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default, rename = "path_glob")]
    path_glob: Option<String>,
    #[serde(default)]
    case_insensitive: Option<bool>,
    #[serde(default)]
    multiline: Option<bool>,
    #[serde(default)]
    before: Option<u32>,
    #[serde(default)]
    after: Option<u32>,
    #[serde(default)]
    head_limit: Option<u32>,
    #[serde(default)]
    respect_gitignore: Option<bool>,
}

#[derive(Debug, Serialize)]
pub(crate) struct SearchOutput {
    pub(crate) mode: &'static str,
    pub(crate) files_scanned: u64,
    pub(crate) files_matched: u32,
    pub(crate) matches_total: u32,
    pub(crate) truncated: bool,
    pub(crate) elapsed_ms: u64,
    pub(crate) head_limit: u32,
}

impl Tool for SearchTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }

    fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
        SafetyClass::ReadOnly
    }

    fn describe<'a>(
        &'a self,
        args: &'a serde_json::Value,
        _ctx: ToolContext<'a>,
    ) -> BoxFuture<'a, ToolCallDescription> {
        Box::pin(async move {
            let mode = args
                .get("mode")
                .and_then(|v| v.as_str())
                .unwrap_or("content");
            let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
            let path = args.get("path").and_then(|v| v.as_str());

            let title = format_title(mode, pattern, path);
            let mut fields = ToolCallUpdateFields::default();
            fields.title = Some(title);
            fields.kind = Some(ToolKind::Search);
            if let Some(p) = path {
                fields.locations = Some(vec![ToolCallLocation::new(PathBuf::from(p))]);
            }
            ToolCallDescription { fields }
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: ToolContext<'_>) -> ToolStream {
        let cancel = ctx.cancel.clone();
        let cwd = ctx.cwd.to_path_buf();
        let config = self.config.clone();
        let fut = async move { run_search(args, cwd, cancel, config).await };
        let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> = Box::pin(stream::once(fut));
        s
    }
}

async fn run_search(
    args: serde_json::Value,
    cwd: PathBuf,
    cancel: CancellationToken,
    config: SearchToolConfig,
) -> ToolEvent {
    let parsed: SearchArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(err) => return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(err))),
    };

    if parsed.pattern.is_empty() {
        return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "pattern must not be empty",
        ))));
    }

    let mode = parsed.mode.unwrap_or_default();
    let head_limit = parsed
        .head_limit
        .unwrap_or(config.default_head_limit)
        .min(config.max_head_limit)
        .max(1);
    let respect_gitignore = parsed
        .respect_gitignore
        .unwrap_or(config.respect_gitignore_default);

    let start_dir = match resolve_search_path(&cwd, parsed.path.as_deref()) {
        Ok(p) => p,
        Err(e) => return ToolEvent::Failed(e),
    };

    // 在阻塞线程跑 walker / grep——`ignore` + `grep-searcher` 都是同步 IO，
    // 在主 runtime 上跑会阻塞其它 task。
    let cancel_for_task = cancel.clone();
    let cwd_for_task = cwd.clone();
    let join = tokio::task::spawn_blocking(move || {
        run_search_blocking(
            mode,
            parsed,
            start_dir,
            cwd_for_task,
            head_limit,
            respect_gitignore,
            cancel_for_task,
            config,
        )
    });

    match join.await {
        Ok(event) => event,
        Err(err) => ToolEvent::Failed(ToolError::Execution(BoxError::new(err))),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_search_blocking(
    mode: SearchMode,
    parsed: SearchArgs,
    start_dir: PathBuf,
    cwd: PathBuf,
    head_limit: u32,
    respect_gitignore: bool,
    cancel: CancellationToken,
    config: SearchToolConfig,
) -> ToolEvent {
    let started = Instant::now();
    match mode {
        SearchMode::Content => {
            let matcher_build = RegexMatcherBuilder::new()
                .case_insensitive(parsed.case_insensitive.unwrap_or(false))
                .multi_line(parsed.multiline.unwrap_or(false))
                .build(&parsed.pattern);
            let matcher = match matcher_build {
                Ok(m) => m,
                Err(err) => {
                    return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            format!("invalid regex pattern: {err}"),
                        ),
                    )));
                }
            };

            let content_glob = match parsed.path_glob.as_deref() {
                Some(spec) => match glob::build_globset(spec) {
                    Ok(set) => Some(set),
                    Err(err) => {
                        return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                format!("invalid glob pattern: {err}"),
                            ),
                        )));
                    }
                },
                None => None,
            };

            let walker = build_walker(&start_dir, respect_gitignore, &config);
            let searcher = SearcherBuilder::new()
                .binary_detection(BinaryDetection::quit(0))
                .before_context(parsed.before.unwrap_or(0) as usize)
                .after_context(parsed.after.unwrap_or(0) as usize)
                .multi_line(parsed.multiline.unwrap_or(false))
                .build();

            content::run(
                walker,
                searcher,
                matcher,
                content_glob,
                &cwd,
                head_limit,
                &cancel,
                &config,
                started,
            )
        }
        SearchMode::Files => {
            let glob_set = match glob::build_globset(&parsed.pattern) {
                Ok(set) => set,
                Err(err) => {
                    return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            format!("invalid glob pattern: {err}"),
                        ),
                    )));
                }
            };
            let walker = build_walker(&start_dir, respect_gitignore, &config);
            files::run(
                walker, glob_set, &cwd, head_limit, &cancel, &config, started,
            )
        }
    }
}

fn build_walker(start: &Path, respect_gitignore: bool, config: &SearchToolConfig) -> ignore::Walk {
    let mut builder = WalkBuilder::new(start);
    builder
        .standard_filters(respect_gitignore)
        .require_git(false)
        .max_filesize(Some(config.max_file_size_bytes))
        .threads(1);
    builder.build()
}

fn resolve_search_path(cwd: &Path, requested: Option<&str>) -> Result<PathBuf, ToolError> {
    let target = match requested {
        None | Some("") => cwd.to_path_buf(),
        Some(s) => {
            let p = Path::new(s);
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                cwd.join(p)
            }
        }
    };

    let canon_target = std::fs::canonicalize(&target).map_err(|e| {
        ToolError::InvalidArgs(BoxError::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("path {} cannot be resolved: {e}", target.display()),
        )))
    })?;
    let canon_cwd = std::fs::canonicalize(cwd).unwrap_or_else(|_| cwd.to_path_buf());

    if !canon_target.starts_with(&canon_cwd) {
        return Err(ToolError::InvalidArgs(BoxError::new(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!(
                "path {} escapes workspace root {}",
                canon_target.display(),
                canon_cwd.display()
            ),
        ))));
    }

    Ok(canon_target)
}

fn format_title(mode: &str, pattern: &str, path: Option<&str>) -> String {
    let verb = if mode == "files" { "Find" } else { "Search" };
    let pat = truncate_for_title(pattern);
    match path {
        Some(p) if !p.is_empty() => {
            let p = truncate_for_title(p);
            format!("{verb} \"{pat}\" in {p}")
        }
        _ => format!("{verb} \"{pat}\""),
    }
}

fn truncate_for_title(s: &str) -> String {
    if s.chars().count() <= TITLE_TRUNC {
        return s.to_string();
    }
    let truncated: String = s.chars().take(TITLE_TRUNC).collect();
    format!("{truncated}…")
}

/// 把 `path` 转换成相对 `cwd` 的展示字符串；落在 cwd 之外时回退到绝对路径。
pub(crate) fn display_relative(cwd: &Path, path: &Path) -> String {
    path.strip_prefix(cwd)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| path.to_string_lossy().into_owned())
}

pub(crate) fn truncate_match_line(line: &str) -> String {
    if line.len() <= MAX_MATCH_LINE {
        return line.to_string();
    }
    let mut end = MAX_MATCH_LINE;
    while !line.is_char_boundary(end) && end > 0 {
        end -= 1;
    }
    let mut out = String::with_capacity(end + 1);
    out.push_str(line.get(..end).unwrap_or(""));
    out.push('…');
    out
}

pub(crate) fn elapsed_ms(started: Instant) -> u64 {
    let m = started.elapsed().as_millis();
    if m > u64::MAX as u128 {
        u64::MAX
    } else {
        m as u64
    }
}

pub(crate) fn make_completed(text: String, output: SearchOutput) -> ToolEvent {
    let raw_output = serde_json::to_value(&output).unwrap_or(serde_json::Value::Null);
    let mut fields = ToolCallUpdateFields::default();
    fields.content = Some(vec![ToolCallContent::Content(Content::new(
        ContentBlock::Text(TextContent::new(text)),
    ))]);
    fields.raw_output = Some(raw_output);
    ToolEvent::Completed(fields)
}

pub(crate) fn sort_by_mtime_desc(hits: &mut [(PathBuf, Option<SystemTime>)]) {
    hits.sort_by_key(|(_, mtime)| Reverse(*mtime));
}
