//! Output rendering for the `fetch` tool.
//!
//! Content rendering for fetch responses.
//!
//! Rendering matrix:
//!
//! | `format`   | content-type            | behavior                                |
//! |------------|-------------------------|-----------------------------------------|
//! | markdown | text/html / xhtml | html → markdown (returns raw HTML + warning when
//! `html_to_markdown` is off) |
//! | markdown   | text/markdown / text/*  | as-is                                   |
//! | markdown   | binary / unknown        | Err(`unsupported content-type`)         |
//! | html       | text/html / xhtml       | as-is                                   |
//! | html       | non-HTML                | Err(`not HTML`)                         |
//! | text       | text/html               | extract body text (strip tags)          |
//! | text       | text/*                  | as-is                                   |
//! | text       | binary                  | Err(`binary content-type`)              |

use defect_config::{FetchFormat, FetchToolConfig};

/// Unified entry point for HTML→markdown / raw-text return.
///
/// # Errors
///
/// Returns `Err(reason)` — the caller wraps the reason as `ToolError::Execution`.
pub(super) fn render(
    body: &[u8],
    content_type: Option<&str>,
    format: FetchFormat,
    config: &FetchToolConfig,
) -> Result<String, String> {
    // An empty body has no rendering ambiguity — 3xx without redirect following, 204, and
    // HEAD-like responses all reach this point. Return an empty string directly instead
    // of reporting "unsupported" based on content-type.
    if body.is_empty() {
        return Ok(String::new());
    }
    let mime = content_type
        .map(parse_main_type)
        .unwrap_or(MainType::Unknown);

    match format {
        FetchFormat::Markdown => render_markdown(body, mime, content_type, config),
        FetchFormat::Html => render_html(body, mime, content_type),
        FetchFormat::Text => render_text(body, mime, content_type),
    }
}

fn render_markdown(
    body: &[u8],
    mime: MainType,
    raw_ct: Option<&str>,
    config: &FetchToolConfig,
) -> Result<String, String> {
    match mime {
        MainType::Html => {
            let html = body_as_str(body)?;
            if !config.html_to_markdown {
                return Ok(format!(
                    "<!-- html_to_markdown disabled in config; returning raw HTML -->\n{html}"
                ));
            }
            htmd::convert(html).map_err(|e| format!("html-to-markdown conversion failed: {e}"))
        }
        MainType::Text => Ok(body_as_str(body)?.to_string()),
        MainType::Binary | MainType::Unknown => Err(format_unsupported("markdown", raw_ct)),
    }
}

fn render_html(body: &[u8], mime: MainType, raw_ct: Option<&str>) -> Result<String, String> {
    match mime {
        MainType::Html => Ok(body_as_str(body)?.to_string()),
        _ => Err(format!("not HTML: {}", raw_ct.unwrap_or("<unset>"))),
    }
}

fn render_text(body: &[u8], mime: MainType, raw_ct: Option<&str>) -> Result<String, String> {
    match mime {
        MainType::Html => {
            let html = body_as_str(body)?;
            // Simple tag stripping: convert HTML to Markdown, then remove any remaining
            // Markdown syntax.
            // Since the content is not rich text, readable plain text is sufficient.
            let md =
                htmd::convert(html).map_err(|e| format!("html-to-text conversion failed: {e}"))?;
            Ok(strip_markdown(&md))
        }
        MainType::Text => Ok(body_as_str(body)?.to_string()),
        MainType::Binary | MainType::Unknown => Err(format!(
            "binary content-type: {}",
            raw_ct.unwrap_or("<unset>")
        )),
    }
}

fn body_as_str(body: &[u8]) -> Result<&str, String> {
    std::str::from_utf8(body).map_err(|e| format!("response body is not valid UTF-8: {e}"))
}

fn format_unsupported(format: &str, raw_ct: Option<&str>) -> String {
    format!(
        "unsupported content-type for {format} format: {}",
        raw_ct.unwrap_or("<unset>")
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MainType {
    Html,
    Text,
    Binary,
    Unknown,
}

fn parse_main_type(content_type: &str) -> MainType {
    // Take the main type before ';', trim whitespace, and lowercase it.
    let head = content_type
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    if head == "text/html" || head == "application/xhtml+xml" {
        return MainType::Html;
    }
    if head.starts_with("text/") {
        return MainType::Text;
    }
    if head == "application/json" || head == "application/xml" {
        return MainType::Text;
    }
    if head.is_empty() {
        return MainType::Unknown;
    }
    MainType::Binary
}

/// Minimal markdown→plain text: strip common markers.
///
/// This is not a compliant markdown parser — `format = "text"` + HTML input is the
/// fallback path for "LLM wants plain text"; it only needs to correctly strip link
/// and heading markers.
fn strip_markdown(md: &str) -> String {
    let mut out = String::with_capacity(md.len());
    let mut chars = md.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '#' | '*' | '_' | '`' | '>' => {
                // Skip consecutive identical characters (## / ** / __ / ``` / >>).
                while chars.peek() == Some(&c) {
                    chars.next();
                }
                // Markdown heading prefixes are usually followed by a space — just keep
                // it.
            }
            '[' => {
                // [text](url) → text
                let mut text = String::new();
                let mut closed_text = false;
                for ch in chars.by_ref() {
                    if ch == ']' {
                        closed_text = true;
                        break;
                    }
                    text.push(ch);
                }
                out.push_str(&text);
                if closed_text && chars.peek() == Some(&'(') {
                    chars.next();
                    for ch in chars.by_ref() {
                        if ch == ')' {
                            break;
                        }
                    }
                }
            }
            _ => out.push(c),
        }
    }
    out
}
