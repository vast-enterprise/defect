# `search` 内置工具设计

`search` 是 defect 的本地 [`Tool`]：在工作区里检索文件名（glob 形态）或文件内容（grep 形态），统一成一个工具，由参数 `mode` 切两种语义。本工具是**纯本地工具**，**不进 capability 层**——hosted 互联网检索是另一件事，叫 `web_search`，由 `[capabilities.web_search]` 单独管理（[capabilities.md](./capabilities.md)）。本文沉淀本工具的形状、与 ACP / 工作区边界的对位、配置入口，以及 P1 暂不做的部分。

设计原则按依赖顺序：

1. **一个工具，两种 mode**——schema 上以 `mode: "content" | "files"` 切「找文本」与「找文件」；理由见 §2。统一在裸名 `search` 下，[`ToolRegistry`] 上只占一个位置。
2. **以 ACP 为导向**——产出的字段直接对位 [`ToolCallUpdateFields`] / [`ToolCallContent`]，复用 [`ToolKind::Search`]。
3. **纯 Rust 实现**——不 spawn `rg` 二进制；用 [`ignore`] + [`grep-searcher`] + [`regex`] + [`globset`] 在进程内跑。理由见 §6.1。
4. **gitignore 默认尊重**——[`ignore::WalkBuilder`] 默认开启 gitignore / `.ignore` / hidden-file 过滤，与 ripgrep 同款行为；用户能用 `respect_gitignore = false` 显式关掉。
5. **不留坑**——大仓库 / 大文件按硬上限截断，超出走 `truncated = true` 而非静默丢；二进制文件 `grep-searcher` 默认跳过；fail loud 优先于 silently-wrong。

[`ToolCallUpdateFields`]: https://docs.rs/agent-client-protocol-schema/0.13/agent_client_protocol_schema/v1/struct.ToolCallUpdateFields.html
[`ToolCallContent`]: https://docs.rs/agent-client-protocol-schema/0.13/agent_client_protocol_schema/v1/enum.ToolCallContent.html
[`ToolKind::Search`]: https://docs.rs/agent-client-protocol-schema/0.13/agent_client_protocol_schema/v1/enum.ToolKind.html
[`Tool`]: ./tool-trait.md
[`ToolRegistry`]: ./session.md
[`ignore`]: https://crates.io/crates/ignore
[`ignore::WalkBuilder`]: https://docs.rs/ignore/latest/ignore/struct.WalkBuilder.html
[`grep-searcher`]: https://crates.io/crates/grep-searcher
[`regex`]: https://crates.io/crates/regex
[`globset`]: https://crates.io/crates/globset

---

## 1. P1 实装现状

| 部件 | 状态 |
|---|---|
| `[tools.search]` 段解析 + `SearchToolConfig` schema | ✅ |
| `Tool` 实现（`crates/tools/src/search/`） | ✅ |
| 注册到 [`ToolRegistry`]（`tools.search.enabled = true` 时）+ CLI 装配 | ✅ |
| ACP `ToolKind::Search` 路径 | ✅ |
| MCP 同名 `search` 命名空间化（`mcp.<server>.search`） | ✅（与所有 MCP 工具一致，[capabilities §6.2](./capabilities.md)） |

本工具与 capability 层**完全独立**：是否注册仅看 `[tools.search].enabled`，不与 `[capabilities.web_search]` 交叉判断。两者可同时启用，LLM 同时持有 hosted `web_search`（搜外网）与本地 `search`（搜代码库），各司其职。

---

## 2. 为什么是「一个 search 工具 + mode」而不是「grep + glob 两个」

主流实现的两条路：

| 项目 | 形态 | 备注 |
|---|---|---|
| Opencode | `grep` + `glob` 两个独立工具 | grep 走 `rg`，glob 走 `rg --files` |
| Claw-code (rust) | `GrepSearch` + `GlobSearch` 两个 input 类型 | 纯 Rust（`walkdir` + `regex` + `glob`） |
| Codex `file-search` crate | 单一文件名模糊搜索（`nucleo` + `ignore`） | 不做内容 grep |

defect 选**单工具**，理由如下：

1. **裸名占位最小化**——[`ToolRegistry`] 上一个工具就够了，没必要把同一类「在工作区找东西」的需求拆成两个裸名（`grep` + `glob`）。LLM 心智上「我要找东西」是一个动作，参数维度是「找文本还是文件」，恰好对位 `mode` 字段。
2. **MCP 命名空间化已经把"占名"问题压到一处**——[capabilities §6.2](./capabilities.md) 让所有 MCP 工具走 `mcp.<server>.<name>` 前缀，本地裸名不会与 MCP 撞。本地 `search` 占一个位足以；引入两个本地工具反而扩大裸名表面。
3. **mode 让 LLM 一目了然**——`{ "mode": "content", "pattern": "TODO" }` 与 `{ "mode": "files", "pattern": "**/*.rs" }` 是两段非常不同的描述；同 schema 下不同 mode 的字段集自然分开（§3），不会产生"一个工具承载了两件无关事"的问题。Codex 的 `file-search` 也只做 files——内容检索是另一个层次的能力，我们把两件事用 `mode` 显式标注。

> **不要把本工具与 hosted `web_search` 混淆**：hosted `web_search` 搜的是互联网，由 provider 在 wire 上自己执行，走 capability 层（[capabilities.md](./capabilities.md)）；本工具搜的是工作区文件，走 [`Tool`] trait，纯本地实现。两者语义不同、配置不同，且**可以同时启用**——一个搜外网，一个搜代码库。

代价：

- **schema 略大**：properties 多了 `mode`、`before/after/context/multiline` 这些只在 content mode 用的字段。文档要在描述里明确写"仅 content mode 生效"。这是 LLM 易混淆点；§3 的 description 字段把"仅 content"显式写明（schema 描述比 enum 标签更管用）。
- **如果将来真要拆成两个工具**：把当前 `mode` 字段提升为顶层 tool name 即可，runtime 行为完全不变。这是干净的延展，与现有设计无冲突。

> **决策**：P1 固化为单工具 `search` + `mode`。如果遇到「LLM 在两种 mode 间反复犯错」的实测信号，把 `mode` 拆成 tool name；schema 内字段不变。

---

## 3. 工具名片

```rust
ToolSchema {
    name: "search".to_string(),
    description: "Search the workspace. \
                  In `content` mode (default) runs a regex over file contents, \
                  returns matching lines with file path + line number; \
                  in `files` mode lists workspace files matching a glob pattern. \
                  Respects .gitignore by default; binary files are skipped in content mode. \
                  Results are sorted by modification time (newest first), truncated at `head_limit`.".to_string(),
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
                "maximum": 1000,
                "description": "Maximum number of files (files mode) or matches (content mode) to return. \
                                Defaults to 100; results sorted newest-first by mtime."
            },
            "respect_gitignore": {
                "type": "boolean",
                "description": "When true (default) honors .gitignore / .ignore / hidden-file rules. \
                                Set to false to search the full tree."
            }
        },
        "required": ["pattern"]
    }),
}
```

字段取舍：

- **没有 `output_mode` (claw-code 的 `files_with_matches | count | content`)**——P1 单一输出形态：返回带 `path:line: match` 的文本块（content mode）或纯文件列表（files mode）。LLM 想"只要文件名"在 content 模式下也能从输出里抽。多 mode 的 `output_mode` 是真实压力大（context 爆炸）时再加的优化，P1 不做。
- **没有 `file_type` (`type = "rust"` 这类)**——`path_glob` 字段足够覆盖（`**/*.rs`）。`file_type` 是 ripgrep 的语法糖，自己实装语言到扩展名的映射对 P1 是噪声。LLM 想限制语言写 glob 就行。
- **没有 `offset`**——`offset + limit` 是 paging 形态；P1 只做 truncate。LLM 想翻页是较少见的真实场景；有真实压力再加。
- **没有 `replace_all` / 修改类参数**——`search` 是只读检索，不做替换。「批量替换」走 `bash("sd ..." / "sed ...")` 或 LLM 显式 `read_file → edit_file`。
- **`pattern` 字段是必填**——空 pattern 没有合理语义。`mode = files` + `pattern = "**/*"` 即"列出所有文件"，LLM 想这么干就显式写 glob。
- **`path` 受 [`workspace_root`] 约束**——同 [`fs`](./tools-fs.md) 工具家族；越界 → `Failed(InvalidArgs)`。
- **`head_limit` 默认 100、硬上限 1000**——参考 opencode / claw-code 的 100；hard cap 防 LLM 输入 `head_limit = 100000` 灌满 context。
- **`before / after` 上限 50**——单次 match 最多带 100 行上下文；防 LLM 写 1000 让单 match 撑爆 content。schema 不暴露 `context`（`-C`）——LLM 想要对称上下文写 `before = after` 即可，避免「同时给了 context 与 before/after 谁优先」的歧义。

[`workspace_root`]: ./tools-fs.md

---

## 4. 安全等级（`safety_hint`）

```rust
fn safety_hint(&self, _args: &Value) -> SafetyClass {
    SafetyClass::ReadOnly
}
```

**一律返回 `ReadOnly`**——`search` 不写本地状态，不出网，副作用在 fs 边界内严格只读。理由与 [`fs::read_file`](./tools-fs.md) 同：本地世界不变、配合 [`ReadOnlyPolicy`] 让"只读模式"用户能跑这个工具。

哪怕 `path` 越界，落地仍走 `InvalidArgs`（schema 拒绝），不升级 `safety_hint`——`safety_hint` 是「这次调用按设计该被怎样审批」，不为越权 args 提前升级（[`SandboxPolicy`] 才是最终守门员）。

[`ReadOnlyPolicy`]: ./sandbox-policy.md
[`SandboxPolicy`]: ./sandbox-policy.md

---

## 5. `describe(args)`：UI 自描述

```rust
ToolCallUpdateFields {
    title: Some(format_title(mode, pattern, path)),
    kind:  Some(ToolKind::Search),
    locations: locations(path),       // 给 follow-along 一个 hint
    content:   None,                  // 执行期填
    raw_input: None,                  // 主循环填
    raw_output: None,                 // 终态填
    status:    None,                  // 主循环填
}
```

`format_title` 形态：

| mode | 标题 |
|---|---|
| `content` | `Search "<pattern>"` 或 `Search "<pattern>" in <relpath>`（pattern 与 relpath 各自截断） |
| `files` | `Find <pattern>` 或 `Find <pattern> in <relpath>` |

`locations`：

- 当 `path` 是绝对/相对到一个**目录**时，给一个 `ToolCallLocation { path: <abs dir>, line: None }`——客户端能 follow 到搜索范围。
- 当未指定 `path` 时，不填 `locations`（默认是 workspace_root，全局搜索没必要给一个 root location）。
- describe 阶段**不**做 IO（不读盘），与 [`bash`](./tools-bash.md) / [`fetch`](./tools-fetch.md) 一致。`path` 是否真实存在留给 execute 阶段验。

---

## 6. `execute`

```text
                           parse args + validate path / pattern
                                       │
                  ┌────────────────────┴────────────────────┐
                  │ mode = content                          │ mode = files
                  ▼                                         ▼
         build regex (case_insensitive,                     build globset
                      multiline)                                  │
                  │                                               ▼
                  ▼                                       walk(workspace_root, path)
         walk(workspace_root, path)                              │
              with ignore + globset                              │
                  │                                              │
                  ▼                                              ▼
         for each file:                                  for each file:
            read with grep-searcher                          glob match
            collect (path, line, text, ctx)                  collect (path, mtime)
                  │                                              │
                  └────────────────────┬─────────────────────────┘
                                       ▼
                           sort by mtime desc, truncate at head_limit
                                       │
                                       ▼
                           render text content + raw_output
                                       │
                          ┌────────────┼─────────────┐
                          ▼            ▼             ▼
                       Completed    cancel       walker / regex
                                       │            error
                                       ▼            │
                                     Failed(Canceled)
                                                    ▼
                                         Failed(Execution(...))
```

P1 只发**一帧** `Completed`（不发中间 `Progress`）——结果聚合在内存中再一次性吐。理由与 [`bash`](./tools-bash.md) / [`fetch`](./tools-fetch.md) 同款：ACP `ToolCallUpdateFields::content` 是 *replace* 语义，多帧 Progress 在大结果集上是 `O(N²)` 字节。

### 6.1 后端：纯 Rust，不 spawn `rg`

候选：

| 方案 | 利 | 弊 |
|---|---|---|
| spawn `rg` 二进制 | 速度极快、跟 ripgrep 行为完全一致 | 需要打包 / 检测 / 下载 `rg`（opencode 是 auto-download，发布形态复杂）；版本飘移；exec 边界又开一个 shell-like 入口 |
| **纯 Rust：`ignore` + `grep-searcher` + `regex` + `globset`** | 单二进制即可发布；打盘语义透明可测；与 [`fs`](./tools-fs.md) / [`bash`](./tools-bash.md) 统一在 in-process IO | 比 `rg` 慢约 1.5–3×（仍比 `walkdir + regex` 快很多，因为 `grep-searcher` 是 ripgrep 同家族） |
| 纯 Rust：`walkdir` + `regex` | 实现最简单 | 不尊重 gitignore；要自己写忽略规则 |

**P1 决策：纯 Rust，用 [`ignore`] + [`grep-searcher`] + [`regex`] + [`globset`]。**

- [`ignore`] 是 ripgrep 自己拆出来的 walker crate，[`WalkBuilder`] 默认行为就是 ripgrep 的"尊重 .gitignore / .ignore / hidden / 全局 ignore"。
- [`grep-searcher`] + [`grep-matcher`] + [`grep-regex`] 是 ripgrep 的核心搜索引擎，独立可用；不需要 spawn 也能拿到接近 `rg` 的速度。
- [`globset`] 用于 glob 模式（同 ripgrep）。

`Cargo.toml` 在 `crates/tools` 增加（不是 workspace 级——这些 crate 只有 `search` 工具用到）：

```toml
ignore = "0.4"
grep-searcher = "0.1"
grep-matcher = "0.1"
grep-regex = "0.1"
globset = "0.4"
regex = { workspace = true }   # 已有
```

`regex` 复用 workspace 里已有的版本。

### 6.2 walker 装配

```rust
let mut builder = WalkBuilder::new(start);
builder
    .standard_filters(args.respect_gitignore.unwrap_or(true))
    .require_git(false)              // 非 git 仓也能走
    .max_filesize(Some(MAX_GREP_FILE_SIZE))   // §6.5
    .threads(1);                     // P1 单线程，简化取消语义
if let Some(g) = &globset {
    builder.filter_entry(move |e| matches_glob(e, g));
}
let walker = builder.build();
```

要点：

- **`standard_filters`** 同时控制 hidden / .ignore / .gitignore；`true` 时与 ripgrep 默认一致；`false` 时全开。
- **`require_git(false)`**：codex 的 `file-search` 设为 `true` 强制要 git 仓——我们不强制（在 git 仓外的临时目录也要能用）。
- **`max_filesize`**：[`grep-searcher`] 自己也有大小限制，但 walker 层先挡掉 > 16 MiB 的文件，少加载 mmap。
- **单线程**：P1 避免并行带来的取消复杂度。压测如果发现 100k 文件仓库慢得不可接受再加并发（[`WalkBuilder::threads`] 直接调）。
- **`filter_entry`** 仅在 directory entry 上跑，文件级 glob 匹配在迭代时再跑一次（globset 太轻量）；这两次匹配函数相同。

### 6.3 content mode 的搜索

```rust
let matcher = grep_regex::RegexMatcherBuilder::new()
    .case_insensitive(args.case_insensitive.unwrap_or(false))
    .multi_line(args.multiline.unwrap_or(false))
    .build(&args.pattern)?;          // 编译失败 → InvalidArgs

let mut searcher = grep_searcher::SearcherBuilder::new()
    .binary_detection(BinaryDetection::quit(0))   // 二进制即停
    .before_context(args.before.unwrap_or(0))
    .after_context(args.after.unwrap_or(0))
    .multi_line(args.multiline.unwrap_or(false))
    .build();

for entry in walker {
    if cancel.is_cancelled() { return Failed(Canceled); }
    let path = entry?.into_path();
    if !path.is_file() { continue; }
    searcher.search_path(&matcher, &path, &mut sink)?;
    if sink.match_count >= effective_head_limit { break; }
}
```

要点：

- **`BinaryDetection::quit(0)`**：发现二进制就停止扫这个文件（不往 LLM 喷 binary garbage）。`grep-searcher` 自己的默认。
- **`before / after`** 由 searcher 配置；schema 不暴露 `context`（§3）。
- **`multi_line`**：matcher 与 searcher 都得开（`grep-searcher` 要求），否则跨行 `.` 匹配不到。
- **`sink`** 是自定义 [`grep_searcher::Sink`]，把 `(path, line_number, line_text)` 累积到结果向量。

### 6.4 files mode 的搜索

```rust
let glob_set = build_globset(&args.pattern)?;     // pattern 即 glob；无 args.glob
let mut hits: Vec<FileHit> = vec![];
for entry in walker {
    if cancel.is_cancelled() { return Failed(Canceled); }
    let entry = entry?;
    let path = entry.path();
    if !path.is_file() { continue; }
    if !glob_set.is_match(path) { continue; }
    let mtime = entry.metadata().and_then(|m| m.modified()).ok();
    hits.push(FileHit { path: path.to_path_buf(), mtime });
}
hits.sort_by_key(|h| Reverse(h.mtime));
hits.truncate(effective_head_limit);
```

`build_globset` 处理大括号展开：`src/**/foo.{ts,tsx}` → 两个 glob。[`globset`] 的 `Glob::new` **不**自动展开大括号，得自己拆——claw-code 同款 `expand_braces`。

> **注**：files mode 不消费 `args.path_glob`——`pattern` 已经是 glob，多一个文件筛选 glob 字段语义重叠。schema 描述上明确"`path_glob` 仅 content mode 生效"。

### 6.5 大小 / 取消 / 上限

| 维度 | 上限 | 行为 |
|---|---|---|
| 单文件大小（content mode） | `MAX_GREP_FILE_SIZE = 16 MiB` | 直接跳过该文件，不计入结果 |
| 总 match 数（content） | `head_limit`（默认 100，最大 1000） | 达到即停止扫描 |
| 总 file 数（files） | 同上 | 收集完再 sort + truncate |
| 单 match line 长度 | `MAX_MATCH_LINE = 4 KiB` | 长行尾部加 `…` 截断 |
| 总 content payload | `MAX_RESULT_BYTES = 256 KiB` | 累积超过即停止追加新 match，标记 `truncated = true` |
| walker 总 file 数 | `MAX_WALK_FILES = 100_000` | 达到即停止扫描，标记 `truncated = true` |
| 取消 | `ctx.cancel.cancelled()` | walker 循环每步检查，drop walker 立即终止 |

理由：

- **`MAX_GREP_FILE_SIZE`**：generated minified JS / vendored bundles 经常 > 5 MiB，搜命中率低、token 浪费高；16 MiB 给真实代码留余量。
- **`MAX_RESULT_BYTES = 256 KiB`**：LLM context 单 message 不该超几十万字符；即便 head_limit 没满也要兜底。
- **`MAX_WALK_FILES = 100_000`**：超大仓（chromium 量级）防爆。
- **`MAX_MATCH_LINE`**：minified 单行可达几 MB，截到 4 KiB 给 LLM 看个大概。

### 6.6 终态

```rust
struct SearchOutput {
    mode: SearchMode,                    // "content" | "files"
    files_scanned: u64,                  // 走到 grep-searcher 的文件数（content）；走 path_glob 检测的（files）
    files_matched: u32,                  // 至少有一个 match 的文件数（content）；命中 pattern glob 的（files）
    matches_total: u32,                  // 全部 match 行数（content 模式才有意义）
    truncated: bool,
    elapsed_ms: u64,
    /// 实际生效的 head_limit（clamp 后）
    head_limit: u32,
}
```

`raw_output = serde_json::to_value(SearchOutput { ... })`。

终态映射规则：

| 退出形态 | event | content 形态 |
|---|---|---|
| 找到结果 | `Completed` | content mode：`<file>\n    L<line>: <text>\n...`；files mode：每行一个 path |
| 没找到任何 match | `Completed` | content = `"(no matches)"`；raw_output.matches_total = 0 |
| `pattern` 不是合法 regex（content） / 不是合法 glob（files） | `Failed(InvalidArgs(...))` | LLM 改 args 重试 |
| `path` 越界 | `Failed(InvalidArgs(...))` | LLM 改 args 重试 |
| walker 中途 IO 失败（permission denied 等） | 跳过该路径，继续走 | 只有"完全无法启动 walker"才算 `Failed(Execution)` |
| `ctx.cancel` 触发 | `Failed(Canceled)` | — |

### 6.7 输出渲染

content mode 的渲染（与 opencode 一致，便于 LLM 引用）：

```text
<workspace-relative-path>
    L42: matched line text
    L43: another match in same file

<another-path>
    L8: ...
```

- 显示路径用 workspace-relative。
- 行号前缀 `L`，匹配 opencode；无 line-number 选项（永远展示）。
- before/after 上下文用同样的 `L<n>: ...` 格式，但行号不带高亮（P1 不做颜色 / markdown 高亮——客户端可以二次渲染）。
- match 之间用空行分隔，文件之间用空行 + 路径标题分隔。
- 末尾追加 `\n[truncated; showing N of M matches]` 当 `truncated = true`。

files mode 的渲染：

```text
<path-1>
<path-2>
...
[truncated; showing N of M files]   # 仅当 truncated
```

- 一行一个 workspace-relative 路径。
- 不带额外元信息（mtime 等）；LLM 想看 mtime 会自己 `bash("ls -lt ...")` 或 `read_file`。
- 末尾追加 truncated 提示同 content mode。

---

## 7. 配置入口

```toml
[tools.search]
enabled = true
default_head_limit = 100
max_head_limit = 1000
max_file_size_bytes = 16777216    # 16 MiB
max_result_bytes = 262144         # 256 KiB
max_walk_files = 100000
respect_gitignore_default = true
```

对位 `SearchToolConfig`：

```rust
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchToolConfig {
    pub enabled: bool,
    pub default_head_limit: u32,
    pub max_head_limit: u32,
    pub max_file_size_bytes: u64,
    pub max_result_bytes: u64,
    pub max_walk_files: u64,
    pub respect_gitignore_default: bool,
}

impl Default for SearchToolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_head_limit: 100,
            max_head_limit: 1000,
            max_file_size_bytes: 16 * 1024 * 1024,
            max_result_bytes: 256 * 1024,
            max_walk_files: 100_000,
            respect_gitignore_default: true,
        }
    }
}
```

字段语义：

- **`enabled`**——`false` 时本地 `search` 工具不注册。这是**唯一**的注册条件，与 `[capabilities.web_search]` 完全无关。MCP `search` 仍走 `mcp.<server>.search` 命名空间，不受影响（[capabilities §6.2](./capabilities.md)）。
- **`default_head_limit` / `max_head_limit`**——LLM 不传 `head_limit` 时用 default；传值大于 max 时 clamp 到 max（不报错；与 [`tools-fetch.md` §7](./tools-fetch.md) 同款）。
- **`max_file_size_bytes` / `max_result_bytes` / `max_walk_files`**——硬上限，由 §6.5 描述。
- **`respect_gitignore_default`**——LLM 不传 `respect_gitignore` 时使用。

### 7.1 没有 per-provider 覆写

P1 **不**支持 `[providers.<p>.tools.search]`。`search` 是全局本地工具，per-provider 启停或参数差异在当前没有真实需求——同一份 `search` 在 Anthropic 与 OpenAI 下行为应一致。

真出现 per-provider 差异时再开口子（缺省值不变，不算 breaking）。同 [`tools-fetch.md` §7.1](./tools-fetch.md)。

### 7.2 与 `[capabilities.web_search]` 的关系

**没有关系。** 本工具是工作区内的 grep/glob，hosted `web_search` 是互联网检索，两者是相互独立的能力。`[tools.search]` 段无论何时都会被解析、`enabled = true` 时无条件注册本地 `search` tool；不会因为开/关 hosted `web_search` 而联动行为，也不会因此发 `ConfigWarning`。

历史包袱：早期版本曾把两者塞进同一个 `SearchCapabilityMode` 三态 enum（`Delegate` / `Local` / `Disabled`），导致语义纠缠。现已分家——hosted 那部分搬去 `[capabilities.web_search]`，本工具单独走 `[tools.search].enabled`。

---

## 8. 与 [`bash`] / [`fs`] / [`fetch`] / 其他工具的边界

| 操作 | 用哪个工具 | 理由 |
|---|---|---|
| "搜一下 TODO" | `search { mode: content, pattern: "TODO" }` | 本工具核心场景 |
| "找出所有 .rs 文件" | `search { mode: files, pattern: "**/*.rs" }` | files mode 直接覆盖 |
| `rg -n "fn main" -t rust` | `search { mode: content, pattern: "fn main", path_glob: "**/*.rs" }` | 同上 |
| 跨进程 / 复杂 ripgrep flags（`-S`, `-w`, `-F`） | `bash("rg ...")` | P1 schema 不暴露这些；走 shell |
| 替换文本 | `bash("sd ..." / "sed ...")` 或 LLM `read_file → edit_file` | search 是只读 |
| 打开 / 读单个文件内容 | `fs.read_file` | search 给 path，read_file 给内容 |
| 拉一个网络 URL 内容 | `fetch` | search 不出网 |
| 检索远程文档（"搜 React 18 release notes"） | hosted `web_search`（`[capabilities.web_search] mode = "delegate"`） | 本工具是本地实现；hosted web 检索走 [capability 层](./capabilities.md) |

`search` 与 `fetch` 不重叠：search 是工作区检索，fetch 是网络读取。`search` 与 hosted `web_search` 也不冲突——前者搜代码库、后者搜外网，两者可同时启用，LLM 用 schema 描述区分。

[`bash`]: ./tools-bash.md
[`fs`]: ./tools-fs.md
[`fetch`]: ./tools-fetch.md

---

## 9. 落地节奏（已完成，留档）

1. **`crates/config/`**——`[tools.search]` 强 schema：
   - `types.rs`：`SearchToolConfig` + `Default` impl + `ToolsConfig.search` 字段。
   - `loader.rs`：解析 `[tools.search]` 段，落到 `EffectiveConfig.tools.search`；`is_known_config_key` 枚举具体子键。
   - `loader/test.rs`：默认值、字段合并、clamp 测试。
2. **`crates/tools/Cargo.toml`**——`ignore` / `grep-searcher` / `grep-matcher` / `grep-regex` / `globset` 依赖。
3. **`crates/tools/src/search/`**—— `SearchTool` 实装：
   - `search.rs`：`pub struct SearchTool`，实现 [`Tool`]，dispatcher。
   - `search/content.rs`：content mode 的 `Sink` + 渲染（§6.3 / §6.7）。
   - `search/files.rs`：files mode 的 walker + sort + 渲染（§6.4 / §6.7）。
   - `search/glob.rs`：`build_globset` + 大括号展开（§6.4）。
   - `search/tests.rs`：§10 用例。
   - `lib.rs`：`pub mod search; pub use search::SearchTool;`。
4. **`crates/cli/src/main.rs`**——`build_process_tools` 在 `tools.search.enabled = true` 时无条件注册 `SearchTool`：
   - 装配点：与 `fetch` 同款的 `if config.effective.tools.fetch.enabled` 形态。**不**与 `[capabilities.web_search]` 交叉判断。
   - 注册名：裸名 `"search"`（不加前缀；MCP 永远走 `mcp.<server>.search` 不冲突）。
5. **`TODO.MD`** / **`capabilities.md`**——已同步更新到 ✅。

---

## 10. 测试矩阵（落地时）

每条都写成 `#[tokio::test]`，放在 `crates/tools/src/search/tests.rs`。fixture 用 `tempfile::TempDir` 构造小型仓库（含 `.gitignore` / 隐藏文件 / 二进制文件）。

| # | 场景 | 验证 |
|---|---|---|
| 1 | content mode：`pattern = "TODO"`，命中 3 行分布在 2 个文件 | event = Completed；content 含 2 段路径 + 3 个 `L<line>:`；raw_output.matches_total = 3；files_matched = 2 |
| 2 | content mode：未命中 | event = Completed；content = "(no matches)"；matches_total = 0 |
| 3 | content mode：`pattern = "[invalid"`（regex 编译错误） | event = Failed(InvalidArgs)，错误信息含 "regex" |
| 4 | content mode：`case_insensitive = true` 时命中大小写不同 | matches_total > 0；不开时 = 0 |
| 5 | content mode：`before = 1, after = 1` | content 对每个 match 都有前后一行（带 `L<n>:`） |
| 6 | content mode：`multiline = true` 命中跨行 pattern `(?s)foo\n.*bar` | matches_total ≥ 1；关掉时 = 0 |
| 7 | content mode：`glob = "**/*.rs"` 限定文件 | 所有命中文件都以 `.rs` 结尾 |
| 8 | content mode：仓库含 `.gitignore` 排除 `vendor/` | 默认不搜 `vendor/`；`respect_gitignore = false` 时能搜到 |
| 9 | content mode：仓库含二进制文件（`\0` 字节） | 不出现在结果里；files_scanned 不计 |
| 10 | content mode：单文件 > `max_file_size_bytes` | 跳过；不在结果中 |
| 11 | content mode：命中数 ≥ `head_limit` | event = Completed；raw_output.truncated = true；content 末尾含 `[truncated]` |
| 12 | content mode：单 match 行 > `MAX_MATCH_LINE` | 行尾 `…` 截断；raw_output.truncated 不变（line 截断不算总 truncate） |
| 13 | content mode：`MAX_RESULT_BYTES` 触发 | 在 head_limit 之前停；truncated = true |
| 14 | content mode：取消 | event = Failed(Canceled) |
| 15 | files mode：`pattern = "**/*.rs"` | 文件列表全是 `.rs`；按 mtime desc |
| 16 | files mode：`pattern = "src/**/foo.{ts,tsx}"` 大括号展开 | 同时命中 `.ts` 与 `.tsx` |
| 17 | files mode：`pattern = "[bad-glob"` | event = Failed(InvalidArgs)，错误信息含 "glob" |
| 18 | files mode：未命中 | content = "(no matches)"；files_matched = 0 |
| 19 | files mode：命中数 ≥ `head_limit` | truncated = true；最近 mtime 优先（不是任意顺序） |
| 20 | files mode：`respect_gitignore = false` | 能找到 `.gitignore` 内的文件 |
| 21 | path 越界（`../../etc`） | event = Failed(InvalidArgs)，含 "escapes workspace root" |
| 22 | `path` 指定一个子目录 | locations 含该目录；只搜该目录 |
| 23 | `head_limit` > `max_head_limit` | clamp 到 max；不报错；raw_output.head_limit 是 max |
| 24 | walker 总文件数 > `max_walk_files` | truncated = true；提前停 |
| 25 | `tools.search.enabled = false` | CLI 装配下 `SearchTool` 不注册（LLM tools schema 里没有 `search`） |
| 26 | 真实 e2e：deepseek prompt "搜一下 TODO" → `search` → 总结 | TurnEnded = `EndTurn`；至少一次 ToolCallStarted/Finished kind = Search |

#3 / #17（参数错误）/ #14（取消）/ #21（越界）/ #25（注册 gate）是 §1 设计原则 5「不留坑」与 §7 配置生效条件的回归基线。

---

## 11. P1 不做（演进口子）

下列每条都是诚实的「feature gap」，当前要么 fail loud（schema 拒绝）要么走 [`bash`]，**不会**静默走错路径。

- **`output_mode` (`files_with_matches` / `count` / `content` 三选)**——P1 一律 `content`。LLM 想要 file 列表用 `mode = files`；想要 count 自己 `bash("rg -c ...")` 或在 content 输出上数。引入时机：用户出现「context 被多匹配灌满」的明确反馈；schema 加 `output_mode`，content / count / files_with_matches 共享同一份扫描结果，渲染层切换。
- **paging（`offset` 字段）**——P1 只做 truncate。引入需要先确定 paging 是 stateless（每次重扫，offset 进 args）还是 stateful（session 内带状态）；前者扩展性差，后者要绑 session lifecycle。等真有压力再设计。
- **`file_type` 语言别名**——P1 用 `glob`。引入时机：LLM 反复犯 glob 语法错（少见，因为 LLM 通常熟 ripgrep flags）；届时把 `file_type` 与 ripgrep 的 type definitions 对齐，需要打包一份 type → globs 表。
- **fuzzy 文件名搜索**——codex 的 `file-search` 用 `nucleo` 做 fuzzy file 搜索（"找类似 `usercontroler` 的文件"）。P1 走 `mode = files` + glob——不做 fuzzy。引入时机：LLM 反复猜不中精确文件名；届时加 `mode = "fuzzy"` 第三态。
- **多线程 walker**——P1 单线程。`ignore::WalkBuilder::threads(N)` 一行代码可调，但取消语义在多 receiver 上更复杂；待大仓压测出现真实瓶颈再开。
- **结构化 patch / hunk 输出**——P1 只输出 `path:line: text`。客户端 UI 想做 grep-style 高亮要自己 parse；ACP 后续若引入 hunk 类型再升级。
- **hosted `web_search`**——`[capabilities.web_search] mode = "delegate"` 走 provider hosted（[`capabilities.md`](./capabilities.md) §3 / §4）。P1 hosted wire 编解码尚未落地（adapter 都返回 `web_search: false`）；本工具与 hosted 路径完全独立，**两者可同时启用**。
- **替换 / write back**——P1 只读。批量替换走 `bash("sd ...")`；单点修改走 `edit_file`。

---

## 12. 决议

1. 单一 `search` 工具 + `mode: "content" | "files"`，纯本地，**不进 capability 层**（hosted 互联网检索是另一件事，叫 `web_search`）
2. 纯 Rust 实现：`ignore` + `grep-searcher` + `grep-regex` + `globset`，不 spawn `rg`
3. gitignore 默认尊重；`respect_gitignore = false` 显式关
4. mtime desc 排序，head_limit 默认 100 / 上限 1000；越界 / 大文件按硬上限 truncate
5. 一律 `SafetyClass::ReadOnly`；不出网、不写盘
6. 注册条件：仅 `[tools.search] enabled = true`（与 `[capabilities.web_search]` 完全独立）；MCP 同名走 `mcp.<server>.search` 命名空间
7. `[tools.search]` 段为强 schema `SearchToolConfig`
