use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::session::turn::{BasePromptConfig, PromptConfig};

const DEFAULT_PROMPT_FILE: &str = "AGENTS.md";

/// # Errors
///
/// 读取 prompt 文件失败，或 prompt 文件所在路径不存在时返回错误。
pub fn resolve_system_prompt(
    cwd: &Path,
    provider: &str,
    model: &str,
    base_prompt: &BasePromptConfig,
    prompt: &PromptConfig,
    session_overlay: Option<&str>,
) -> Result<Option<String>, io::Error> {
    let mut sections = Vec::new();

    sections.extend(load_base_prompt(base_prompt)?);

    if let Some(text) = prompt.text.as_deref() {
        sections.push(text.to_owned());
    }

    if let Some(project_prompt) = load_prompt_file(cwd, &prompt.file)? {
        sections.extend(project_prompt);
    }

    if let Some(provider_overlay) = prompt.provider_overlays.get(provider) {
        sections.push(provider_overlay.clone());
    }

    if let Some(model_overlay) = prompt.model_overlays.get(model) {
        sections.push(model_overlay.clone());
    }

    if let Some(session_overlay) = session_overlay {
        sections.push(session_overlay.to_owned());
    }

    Ok((!sections.is_empty()).then(|| sections.join("\n\n")))
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

fn load_prompt_file(cwd: &Path, file: &str) -> Result<Option<Vec<String>>, io::Error> {
    if file != DEFAULT_PROMPT_FILE {
        let path = resolve_prompt_path(cwd, file);
        return match fs::read_to_string(&path) {
            Ok(text) => Ok(Some(vec![text])),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err),
        };
    }

    let mut sections = Vec::new();
    for dir in prompt_dirs(cwd) {
        let path = dir.join(DEFAULT_PROMPT_FILE);
        match fs::read_to_string(&path) {
            Ok(text) => sections.push(text),
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => return Err(err),
        }
    }
    Ok((!sections.is_empty()).then_some(sections))
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
#[path = "prompt/test.rs"]
mod test;
