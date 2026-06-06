use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::session::context::RunningContext;
use crate::session::turn::{BasePromptConfig, PromptConfig};

const DEFAULT_PROMPT_FILE: &str = "AGENTS.md";

/// 一段带标题的 system prompt 片段。最终各段套一级标题（`#`）、以 markdown
/// 水平分割线（`---`）相隔拼接，让模型把每段当作独立文档理解。
///
/// 约定：注入的标题占用一级（`#`），片段正文（base_prompt / AGENTS.md 等）
/// 建议从二级标题（`##`）起步，自然嵌套在其下。
struct Section {
    title: String,
    body: String,
}

impl Section {
    fn new(title: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            body: body.into(),
        }
    }

    fn render(&self) -> String {
        format!("# {}\n\n{}", self.title, self.body)
    }
}

/// 把各片段套一级标题、以 `\n\n---\n\n` 相隔拼接。
fn render_sections(sections: &[Section]) -> Option<String> {
    (!sections.is_empty()).then(|| {
        sections
            .iter()
            .map(Section::render)
            .collect::<Vec<_>>()
            .join("\n\n---\n\n")
    })
}

/// # Errors
///
/// 读取 prompt 文件失败，或 prompt 文件所在路径不存在时返回错误。
pub fn resolve_system_prompt(
    ctx: &RunningContext,
    provider: &str,
    model: &str,
    base_prompt: &BasePromptConfig,
    prompt: &PromptConfig,
    session_overlay: Option<&str>,
) -> Result<Option<String>, io::Error> {
    let mut sections = Vec::new();

    for body in load_base_prompt(base_prompt)? {
        sections.push(Section::new("Base Prompt", body));
    }

    // 运行环境信息：紧跟 base prompt（身份）之后、project 约定之前，作为稳定
    // 的事实层。
    sections.push(Section::new("Environment", ctx.render()));

    if let Some(text) = prompt.text.as_deref() {
        sections.push(Section::new("System Instructions", text.to_owned()));
    }

    for (path, body) in load_prompt_file(ctx.cwd, &prompt.file)? {
        let title = match path {
            Some(path) => format!("Project Instructions ({path})"),
            None => "Project Instructions".to_owned(),
        };
        sections.push(Section::new(title, body));
    }

    if let Some(provider_overlay) = prompt.provider_overlays.get(provider) {
        sections.push(Section::new(
            format!("Provider Notes ({provider})"),
            provider_overlay.clone(),
        ));
    }

    if let Some(model_overlay) = prompt.model_overlays.get(model) {
        sections.push(Section::new(
            format!("Model Notes ({model})"),
            model_overlay.clone(),
        ));
    }

    if let Some(session_overlay) = session_overlay {
        sections.push(Section::new(
            "Session Instructions",
            session_overlay.to_owned(),
        ));
    }

    Ok(render_sections(&sections))
}

fn load_base_prompt(base_prompt: &BasePromptConfig) -> Result<Vec<String>, io::Error> {
    let mut sections = Vec::new();

    if let Some(file) = base_prompt.file.as_deref() {
        let text = fs::read_to_string(file)?;
        sections.push(text);
    }

    if let Some(text) = base_prompt.text.as_deref() {
        sections.push(text.to_owned());
    }

    Ok(sections)
}

/// 加载 project prompt 文件。返回每段 `(相对来源路径, 正文)`——来源路径用于
/// 标进片段标题（`# Project Instructions (...)`），相对 `cwd` 计算，失败时回退
/// 为文件名本身。非默认文件名只读单个位置；默认 `AGENTS.md` 沿目录树向上收集。
fn load_prompt_file(cwd: &Path, file: &str) -> Result<Vec<(Option<String>, String)>, io::Error> {
    if file != DEFAULT_PROMPT_FILE {
        let path = resolve_prompt_path(cwd, file);
        return match fs::read_to_string(&path) {
            Ok(text) => Ok(vec![(Some(rel_label(cwd, &path)), text)]),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(err) => Err(err),
        };
    }

    // AGENTS.md 沿目录树自 repo root 向下收集，故来源标签相对 repo root 计算
    // （如 `AGENTS.md`、`apps/web/AGENTS.md`）。
    let base = find_repo_root(cwd).unwrap_or_else(|| cwd.to_path_buf());
    let mut sections = Vec::new();
    for dir in prompt_dirs(cwd) {
        let path = dir.join(DEFAULT_PROMPT_FILE);
        match fs::read_to_string(&path) {
            Ok(text) => sections.push((Some(rel_label(&base, &path)), text)),
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => return Err(err),
        }
    }
    Ok(sections)
}

/// 来源路径标签：优先相对 `base`，无法相对化时回退到文件名，再回退到全路径。
fn rel_label(base: &Path, path: &Path) -> String {
    path.strip_prefix(base)
        .ok()
        .map(|rel| rel.display().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| path.display().to_string())
}

fn resolve_prompt_path(cwd: &Path, file: &str) -> PathBuf {
    if file.is_empty() {
        cwd.join(DEFAULT_PROMPT_FILE)
    } else {
        cwd.join(file)
    }
}

fn prompt_dirs(cwd: &Path) -> Vec<PathBuf> {
    let Some(repo_root) = find_repo_root(cwd) else {
        return vec![cwd.to_path_buf()];
    };

    let mut dirs = Vec::new();
    for dir in cwd.ancestors() {
        dirs.push(dir.to_path_buf());
        if dir == repo_root.as_path() {
            break;
        }
    }
    dirs.reverse();
    dirs
}

fn find_repo_root(cwd: &Path) -> Option<PathBuf> {
    cwd.ancestors()
        .find(|dir| dir.join(".git").exists())
        .map(Path::to_path_buf)
}

#[cfg(test)]
mod test;
