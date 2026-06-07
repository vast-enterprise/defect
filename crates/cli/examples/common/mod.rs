//! Shared helper for `defect-cli` examples. **For `examples/` only** — not part of the
//! crate's public API (`defect-cli` is a binary crate with no lib target).
//!
//! Difference from [`crates/llm/examples/common/mod.rs`]: this module is ACP-oriented —
//! - subscriber always uses `with_writer(std::io::stderr)` (stdout is reserved for wire)
//! - default `EnvFilter` silences toac's `INFO request` events (which contain
//!   authorization header in plain text; see tracing design)
//!
//! See tracing design.

use std::io::IsTerminal;
use std::path::Path;

/// Set up tracing: defaults to `info,toac=warn`, overridden entirely by the `RUST_LOG`
/// environment variable.
///
/// `toac=warn` is part of the default directive — the toac wire crate emits request
/// events at `info` level with `headers={"authorization": "Bearer ..."}`, so they must be
/// silenced by default to avoid leaking credentials to stderr. To inspect wire requests,
/// explicitly set `RUST_LOG=...,toac=debug` (debug level does not include headers).
///
/// **stderr is mandatory**: the stdio ACP occupies stdout; writing to stdout from the
/// subscriber would corrupt the protocol wire, causing client decode failures.
///
/// # Panics
///
/// Re-initialization panics — examples call this only once in `main`.
pub fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,toac=warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_ansi(std::io::stderr().is_terminal())
        .init();
}

/// A minimal `.env` loader: one `KEY=VALUE` per line, lines starting with `#` are
/// comments, blank lines are skipped; outer `"..."` / `'...'` quotes are stripped.
/// **Variables already set in the process environment are preserved**, so `.env` cannot
/// override a shell-exported variable. Missing file or parse errors only produce a
/// warning.
///
/// This is the same implementation as `crates/cli/src/main.rs::load_env_file` — examples
/// cannot reference private functions from the binary crate, so this is a copy.
pub fn load_env_file(path: &Path) {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return,
        Err(err) => {
            eprintln!("warning: failed to read {}: {err}", path.display());
            return;
        }
    };
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let k = k.trim();
        let v = v.trim().trim_matches('"').trim_matches('\'');
        if k.is_empty() || std::env::var_os(k).is_some() {
            continue;
        }
        // SAFETY: no other threads have been spawned at the entry point of the examples.
        unsafe { std::env::set_var(k, v) };
    }
}
