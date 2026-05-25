# Config P1 Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the P1 config work so `defect-config` owns dotenv loading, full P1 schema, warnings, and runtime wiring for tools and sandbox.

**Architecture:** Extend `defect-config` into the single configuration assembly point, then thread the resulting typed config into CLI, tool construction, and sandbox policy selection. Keep the rollout bounded to `docs/internal/config.md` P1 and prove behavior through config-crate white-box tests plus thin runtime checks.

**Tech Stack:** Rust 2024, `toml`, `serde`, `clap`, existing `defect-agent` policy abstractions, existing `defect-tools` implementations.

---

### Task 1: Expand Config Types And Loader Surface

**Files:**
- Modify: `crates/config/src/types.rs`
- Modify: `crates/config/src/lib.rs`
- Test: `crates/config/src/loader/test.rs`

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn loads_tools_and_sandbox_sections_into_effective_config() { /* ... */ }

#[test]
fn warns_on_unknown_keys_with_source_path() { /* ... */ }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p defect-config loads_tools_and_sandbox_sections_into_effective_config warns_on_unknown_keys_with_source_path`
Expected: FAIL because the new schema and warnings are not implemented yet.

- [ ] **Step 3: Write minimal implementation**

```rust
pub struct EffectiveConfig {
    pub cli: CliConfig,
    pub turn: TurnConfig,
    pub providers: ProviderConfigs,
    pub tools: ToolsConfig,
    pub sandbox: SandboxConfig,
    pub tracing: TracingConfig,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p defect-config loads_tools_and_sandbox_sections_into_effective_config warns_on_unknown_keys_with_source_path`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/config/src/types.rs crates/config/src/lib.rs crates/config/src/loader/test.rs
git commit -m "feat: expand config schema for tools and sandbox"
```

### Task 2: Add Dotenv Compatibility And Complete Loader Behavior

**Files:**
- Modify: `crates/config/src/loader.rs`
- Modify: `crates/config/src/lib.rs`
- Test: `crates/config/src/loader/test.rs`

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn loads_dotenv_compat_without_overwriting_existing_env() { /* ... */ }

#[test]
fn returns_parse_errors_with_source_path() { /* ... */ }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p defect-config loads_dotenv_compat_without_overwriting_existing_env returns_parse_errors_with_source_path`
Expected: FAIL because dotenv loading is still in CLI and loader coverage is incomplete.

- [ ] **Step 3: Write minimal implementation**

```rust
pub fn load_dotenv_compat(cwd: &Path) -> Result<(), ConfigError> {
    // read cwd/.env, preserve pre-existing env vars
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p defect-config loads_dotenv_compat_without_overwriting_existing_env returns_parse_errors_with_source_path`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/config/src/loader.rs crates/config/src/lib.rs crates/config/src/loader/test.rs
git commit -m "feat: add config-managed dotenv loading"
```

### Task 3: Complete P1 Merge Matrix And Diagnostics

**Files:**
- Modify: `crates/config/src/loader.rs`
- Modify: `crates/config/src/overrides.rs`
- Test: `crates/config/src/loader/test.rs`

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn arrays_replace_instead_of_append() { /* ... */ }

#[test]
fn config_files_can_be_missing_without_error() { /* ... */ }

#[test]
fn local_project_keeps_denied_shared_keys_working() { /* ... */ }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p defect-config arrays_replace_instead_of_append config_files_can_be_missing_without_error local_project_keeps_denied_shared_keys_working`
Expected: FAIL where loader behavior is still incomplete.

- [ ] **Step 3: Write minimal implementation**

```rust
fn merge_toml_values(base: &mut TomlValue, overlay: &TomlValue) {
    // recurse on tables, replace arrays and scalars wholesale
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p defect-config arrays_replace_instead_of_append config_files_can_be_missing_without_error local_project_keeps_denied_shared_keys_working`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/config/src/loader.rs crates/config/src/overrides.rs crates/config/src/loader/test.rs
git commit -m "feat: complete config merge diagnostics"
```

### Task 4: Wire CLI, Tools, And Sandbox To Effective Config

**Files:**
- Modify: `crates/cli/src/main.rs`
- Modify: `crates/tools/src/bash/mod.rs`
- Modify: `crates/tools/src/fs/read.rs`
- Modify: `crates/tools/src/lib.rs`
- Modify: `crates/agent/src/policy.rs`
- Test: `crates/tools/src/bash/tests.rs`
- Test: `crates/tools/src/fs/tests.rs`

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn bash_tool_uses_configured_timeout_limits() { /* ... */ }

#[test]
fn read_file_tool_uses_configured_line_limits() { /* ... */ }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p defect-tools bash_tool_uses_configured_timeout_limits read_file_tool_uses_configured_line_limits`
Expected: FAIL because tool constructors are still hard-coded.

- [ ] **Step 3: Write minimal implementation**

```rust
let tools: Arc<dyn ToolRegistry> = Arc::new(
    StaticToolRegistry::builder()
        .insert(Arc::new(BashTool::from_config(&config.effective.tools.bash)))
        .insert(Arc::new(ReadFileTool::from_config(&config.effective.tools.fs)))
        .build(),
);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p defect-tools bash_tool_uses_configured_timeout_limits read_file_tool_uses_configured_line_limits`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/cli/src/main.rs crates/tools/src/bash/mod.rs crates/tools/src/fs/read.rs crates/tools/src/lib.rs crates/agent/src/policy.rs crates/tools/src/bash/tests.rs crates/tools/src/fs/tests.rs
git commit -m "feat: wire runtime to effective config"
```

### Task 5: End-To-End Verification

**Files:**
- Modify: `crates/config/src/loader/test.rs`
- Modify: `crates/cli/src/main.rs`

- [ ] **Step 1: Add any remaining regression tests**

```rust
#[test]
fn env_backed_provider_override_matches_cli_override() { /* ... */ }
```

- [ ] **Step 2: Run focused package tests**

Run: `cargo test -p defect-config && cargo test -p defect-tools`
Expected: PASS

- [ ] **Step 3: Run workspace lint**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: PASS

- [ ] **Step 4: Run formatter**

Run: `cargo fmt`
Expected: exit 0 with no formatting diff afterwards.

- [ ] **Step 5: Commit**

```bash
git add crates/config/src/loader/test.rs crates/cli/src/main.rs
git commit -m "test: finish config p1 verification"
```
