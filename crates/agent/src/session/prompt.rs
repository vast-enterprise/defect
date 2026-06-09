use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::session::context::RunningContext;
use crate::session::turn::{BasePromptConfig, PromptConfig};

const DEFAULT_PROMPT_FILE: &str = "AGENTS.md";

/// A system prompt section with a title. Each section is wrapped in a level-1 heading
/// (`#`) and separated by a markdown horizontal rule (`---`), so the model treats each
/// section as an independent document.
///
/// Convention: the injected title uses level-1 (`#`); the section body (e.g.
/// `base_prompt` / `AGENTS.md`) should start at level-2 (`##`) and nest naturally
/// underneath.
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

/// Wrap each section in a level-1 heading and join them separated by `\n\n---\n\n`.
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
/// Returns an error if reading the prompt file fails or the prompt file path does not
/// exist.
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

    // Environment info: placed immediately after the base prompt (identity) and before
    // project conventions, serving as a stable fact layer.
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

/// Load **only** the project instruction layer (`AGENTS.md`, collected up the directory
/// tree) as a single rendered string, or `None` if there is none.
///
/// This is the "project world knowledge" slice of [`resolve_system_prompt`] — deliberately
/// excluding the base prompt, environment block, provider/model overlays, and session
/// overlay (all of which are the *parent agent's* identity/runtime, not shareable project
/// context). A subagent profile may opt in to this layer (`inherit_project_prompt = true`)
/// so it gets build/test/architecture conventions without inheriting the parent's identity.
///
/// # Errors
/// Propagates IO errors from reading `AGENTS.md` files (NotFound is not an error).
pub fn load_project_prompt(cwd: &Path) -> Result<Option<String>, io::Error> {
    let mut sections = Vec::new();
    for (path, body) in load_prompt_file(cwd, DEFAULT_PROMPT_FILE)? {
        let title = match path {
            Some(path) => format!("Project Instructions ({path})"),
            None => "Project Instructions".to_owned(),
        };
        sections.push(Section::new(title, body));
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

/// Loads the project prompt file. Returns a list of `(relative source path, text)` pairs
/// — the source path is used in the section heading (`# Project Instructions (...)`),
/// computed relative to `cwd`, falling back to the bare filename on failure. For
/// non-default filenames, only a single location is read; for the default `AGENTS.md`,
/// files are collected by walking up the directory tree.
fn load_prompt_file(cwd: &Path, file: &str) -> Result<Vec<(Option<String>, String)>, io::Error> {
    if file != DEFAULT_PROMPT_FILE {
        let path = resolve_prompt_path(cwd, file);
        return match fs::read_to_string(&path) {
            Ok(text) => Ok(vec![(Some(rel_label(cwd, &path)), text)]),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(err) => Err(err),
        };
    }

    // AGENTS.md is collected downward from the repo root along the directory tree, so
    // source labels are computed relative to the repo root (e.g. `AGENTS.md`,
    // `apps/web/AGENTS.md`).
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

/// Label for the source path: prefer a path relative to `base`, fall back to the file
/// name, then to the full path.
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
mod tests;
