//! `fetch` 工具的输出渲染。
//!
//! 详见 `docs/internal/tools-fetch.md` §6.2 / §6.3。
//!
//! 渲染矩阵：
//!
//! | `format`   | content-type            | 行为                                    |
//! |------------|-------------------------|-----------------------------------------|
//! | markdown   | text/html / xhtml       | html → markdown（`html_to_markdown` 关时返回原 HTML + warning） |
//! | markdown   | text/markdown / text/*  | 原文                                    |
//! | markdown   | binary / 未知            | Err(`unsupported content-type`)         |
//! | html       | text/html / xhtml       | 原文                                    |
//! | html       | 非 HTML                  | Err(`not HTML`)                         |
//! | text       | text/html               | 抽 body 文本（去标签）                  |
//! | text       | text/*                   | 原文                                    |
//! | text       | binary                   | Err(`binary content-type`)              |

use defect_config::{FetchFormat, FetchToolConfig};

/// HTML→markdown / 原文返回的统一入口。
///
/// 失败时 `Err(reason)`——上层把 reason 包成
/// `ToolError::Execution`。详见 `docs/internal/tools-fetch.md` §6.2。
pub(super) fn render(
    body: &[u8],
    content_type: Option<&str>,
    format: FetchFormat,
    config: &FetchToolConfig,
) -> Result<String, String> {
    // 空 body 没有渲染歧义——3xx 不跟随 / 204 / HEAD-like 响应都走这里，
    // 直接返回空字符串而不是按 content-type 报"unsupported"。
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
            // 简易"去标签"：把 HTML 转成 markdown 后再剥掉残留 markdown 语法
            // 本来就不是富文本，能给出可读 plain text 即可。
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
    // 取 ';' 之前的主类型，去空格并小写。
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

/// 极简 markdown→plain text：去掉常见标记。
///
/// 这不是合规的 markdown 解析器——`format = "text"` + HTML 输入是「LLM
/// 想要纯文本」的兜底路径，能正确剥掉链接 / 标题标记即可。
fn strip_markdown(md: &str) -> String {
    let mut out = String::with_capacity(md.len());
    let mut chars = md.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '#' | '*' | '_' | '`' | '>' => {
                // 跳过紧邻的同字符（## / ** / __ / ``` / >>）。
                while chars.peek() == Some(&c) {
                    chars.next();
                }
                // markdown 标题前缀往往跟一个空格——保留它即可。
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
