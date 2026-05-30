# Skill 系统设计

`Skill` 是用户为 `defect-agent` 配置的可复用提示片段——一段 markdown 文本，加上可选的脚本 / 参考文件，按规则注入到 LLM 的 system prompt，或在模型按需要时通过 `skill` 工具拉取完整内容。

本文沉淀 skill 的文件形态、加载/匹配语义、与 [`hooks.md`](./hooks.md) / [`tool-trait.md`](./tool-trait.md) 的关系。具体配置语法 / 路径覆盖见 [`config.md`](./config.md) 演进版本。

> **实现现状（v0 已落地）**：本文成文于 subagent 系统之前，§2.2 当时写"defect 没有 sub-agent 概念"已过时——`spawn_agent` / `ProfileSpec`（`crates/config/src/profiles.rs`）随后落地，成了 skill 的同构姊妹特性。落地时按那套更被验证的约定，对本文做了几处**有意偏离**，以实现为准：
> - **发现逻辑在 `defect-config`**（`crates/config/src/skills.rs`，镜像 `profiles.rs`），不在 `defect-agent`——保持 config→agent 的单向依赖。产出 `SkillSpec`，CLI 装配期投影成 agent 侧 `SkillEntry`。
> - **`SkillTool` 在 `crates/agent/src/tool/skill.rs`**（紧挨 `spawn_agent.rs`），注册进 `process_tools`（不是 §6.2 设想的 session-level factory）——即随 `AgentCore` 实例、被该 core 的各 session 共享一份。**不是进程全局单例**：把 defect 当库引用、一个进程里装配多个 `AgentCore` 时，各 core 各持自己的 skill 索引。
> - **加载错误一律 hard fail**（缺 frontmatter / 缺字段 / `name`≠目录名 / 未知键），不是 §3.4 表的 warn-and-skip——与 `profiles.rs` 和 [[feedback-no-wrong-v0-impls]] / [[feedback-minimize-no-paternalistic-guards]] 一致。
> - **L1 清单优先走 `skill` 工具自己的 description catalog**（与 `spawn_agent` 把 profile catalog 编进 description 同款，零配置、单一真相源）；§6.1 的 `SkillManifestHook` 仍实现为**可选** builtin（名 `skill-manifest`，CLI 用捕获 skill 索引的闭包注册），用户可在 `[[hooks.session_start]]` 显式挂上让清单同时进 system prompt。
> - **frontmatter 同时支持 `+++`(TOML) / `---`(YAML)**（与单文件 profile 共用 `crates/config/src/frontmatter.rs`），比 §3.2 只画 YAML 更宽。
> - **`always` / `triggers` / `allowed_tools` 字段 v0 已显式占位**（"解析但不消费"，对齐 §3.2 表）：`SkillManifestToml` 保留 `deny_unknown_fields` 抓必填项 typo，同时把这三个 open-standard 字段显式列出（`allowed_tools` 兼收 Anthropic 的连字符 `allowed-tools`），所以带这些字段的上游 skill 扔进来不报错、值被忽略；v1 接入时从"被忽略"变"被消费"，对用户文件向后兼容。§4.3 / §5.1 的运行期行为仍属未来计划。

设计前提：
- 三家参考实现（Anthropic Claude Code、OpenAI Codex、opencode）都把 skill 当成"提示片段 + 可选脚本"的组合体，但**激活策略**与**注入靶点**分歧很大；详见 §2 对照表。defect 选定的组合在 §3。
- skill 系统**不**自己造加载点——`SessionStart` / `UserPromptSubmit` 是 [`hooks.md`](./hooks.md) §2 已定义的 Sync 拦截事件，skill 走 builtin handler 接进去（见 §6.1）。
- skill 工具（模型主动加载）**不**是 hook，也不是新一类抽象——它是 [`tool.rs`](../../crates/agent/src/tool.rs) 里的普通 `Tool` 实现（见 §6.2）。

## 1. 定位与术语

| 概念 | 含义 |
| --- | --- |
| **Skill** | 一个 markdown 文件 + 同目录下的可选 scripts / refs，由用户编写或仓库携带 |
| **Skill descriptor** | skill 的 frontmatter 解析结果，**不**含 body —— 用于"清单注入"时只占少量 token |
| **Skill body** | `SKILL.md` 去掉 frontmatter 后的全文，按需在第二阶段加载 |
| **Skill registry** | 内存里的 skill 索引（随 `AgentCore` 实例、跨该 core 的 session 共享）：扫描目录 / 解析 frontmatter 的产物，持有所有可用 skill 的 descriptor + body 路径 |
| **Skill loader** | 把 registry 接给 hook builtin 与 `skill` tool 的胶水层 |

Skill **不是**：
- **第二条 prompt 拼接路径**——skill body 的注入靶点统一是 system prompt suffix，与 hook outcome 的 `append` 走同一条路（见 §5）。
- **绕过 hooks 的私有事件**——skill 加载时机就是 `SessionStart` / `UserPromptSubmit` 这两个 hook 事件；不在主循环里另开"skill emit"。
- **MCP 服务**——skill 是仓库本地静态文件，没有 transport / 握手；MCP 服务器走 [`config.md`](./config.md) 的 `[mcp.servers]` 段。

### 1.1 三阶段 progressive disclosure（参考 Anthropic）

模型上下文里的 skill 内容分三层加载——这不是 defect 发明的，而是 [Anthropic Skills](https://platform.claude.com/docs/en/docs/agents-and-tools/agent-skills/overview) 与 [Codex Skills](https://developers.openai.com/codex/skills) 共同遵循的"progressive disclosure"模式：

| 层级 | 内容 | 何时进入 context | Token 量 |
| --- | --- | --- | --- |
| L1 — 清单 | `name + description` of all available skills | `SessionStart` hook 拼进 system prompt suffix | 每条 ~100 token，全表预算 ~1% context |
| L2 — body | 单个 skill 的 `SKILL.md` 全文 | 模型调 `skill` 工具拉取，或被 `UserPromptSubmit` matcher 命中（v1） | 单条几百到几千 token |
| L3 — 附件 | skill 目录下的 `scripts/*.py` / `refs/*.md` 等 | 模型自己用 `bash` / `fs.read` 工具读 | 不计入 skill 系统预算 |

defect v0 仅承诺 L1 + L2 + L3 的接入点；token budget 算法（清单截断 / compact 后重附）留给 v1。

## 2. 三家参考实现对照

| 维度 | Anthropic Claude Code | OpenAI Codex | opencode | **defect 选定** |
| --- | --- | --- | --- | --- |
| 用户概念 | `Skill` (`.claude/skills/<name>/SKILL.md`) | `Skill` (`.agents/skills/<name>/SKILL.md`) | `agent` / `command` (单文件) | **`Skill`**（命名跟 Anthropic/Codex 对齐） |
| 文件粒度 | dir-per-skill + bundled scripts | dir-per-skill + bundled scripts | 单文件 `<name>.md` | **dir-per-skill** |
| 必须字段 | `name`, `description` | `name`, `description` | `description` | **`name`, `description`**（同 §3.1） |
| 项目位置 | `.claude/skills/` | `.agents/skills/` | `.opencode/agents` 或 `.opencode/commands` | **`.defect/skills/`** |
| 用户位置 | `~/.claude/skills/` | `$CODEX_HOME/skills`, `~/.agents/skills/` | `~/.config/opencode/agents/` | **`$XDG_CONFIG_HOME/defect/skills/`** |
| 激活默认 | LLM-decided（按 description 匹配） | 同 Anthropic | 人工 `/cmd` 触发 | **L1 always-on 清单 + L2 模型 tool 拉取**（同 Anthropic/Codex） |
| 注入靶点 | 清单进 system prompt；body 进 tool result | 同 Anthropic | command body 作为 user message 模板 | **统一 system prompt suffix**（与 [hooks §3.1] `append` 同条路） |
| 模型可调用工具 | `Skill` | `skill_load`-类工具（同 open standard） | 无 | **`skill`**（snake_case，对齐 §6.2） |
| 用户 `/cmd` 触发 | 是（`/skill-name`） | 是（`$skill-name`） | 是 | **v0 不做**（留给 ACP `slash_commands` 接入；见 §9） |
| 热加载 | 是（FS watcher） | 否 | 否 | **v0 否**（启动期全扫描；见 §7.4） |

[hooks §3.1]: ./hooks.md#31-hookoutcome可组合结构

### 2.1 defect 与 Anthropic / Codex 的对齐点

defect 的 skill 形态尽量遵循 Anthropic 的 [Agent Skills open standard](https://platform.claude.com/docs/en/docs/agents-and-tools/agent-skills/overview)：

- `SKILL.md` 必带 YAML frontmatter
- frontmatter 至少有 `name` / `description`（短描述用于 L1 清单）
- skill 目录内可同级放 `scripts/` `refs/` 子目录，模型用普通工具读
- 模型用一个名为 `skill` 的工具按 name 拉取 body

对齐的好处：用户已经写好的 Anthropic-format skill，扔进 `.defect/skills/` 就能用；维护两套格式没价值。

### 2.2 与三家的明确分歧

- **不模仿 opencode 的 user-only 触发**——defect 走 LLM-decided（L1 清单 + 模型 tool 拉取）作为默认。理由：opencode 模式假设有 TUI 让用户按 `Tab`/`/`，而 defect 是 ACP server 跑在 Zed / headless 等多种宿主里，宿主不一定有 slash command UI；让模型主动决定加载更可移植。
- **不复刻 Anthropic 的 `paths:` glob 网关**——v0 不在 frontmatter 上做"按文件路径自动激活"的字段。文档 §4.1 的 frontmatter schema 留出 `triggers` 字段打桩，但匹配实现 v0 不做（见 §4.3）。
- **不接入 Anthropic 的 `context: fork` / sub-agent 路由**——defect 没有 sub-agent 概念（参考 [`session.md`](./session.md) 的单 session 模型），skill 不能 fork 出新 session。

## 3. 文件形态

### 3.1 目录结构

```
.defect/skills/
├── code-review/                    # skill 名 = 目录名
│   ├── SKILL.md                    # 必有
│   ├── scripts/
│   │   └── lint-rust.sh            # 模型按需 bash 运行
│   └── refs/
│       └── style-guide.md          # 模型按需 fs.read
├── debug-rust/
│   └── SKILL.md
└── ...
```

约束：
- 目录名 = skill name；只允许 `[a-z0-9_-]+`，长度 ≤ 64 字符。约束理由是 skill name 会出现在 L1 清单与 `skill` 工具的 args 里，避免转义麻烦。
- `SKILL.md` 必须存在；其他文件 / 子目录形态不约束。
- 嵌套子 skill（`<name>/sub-skill/SKILL.md`）**不识别**——v0 走单层目录。

### 3.2 SKILL.md frontmatter

```markdown
---
name: code-review
description: 用 cargo / clippy 风格审查 Rust 代码，命中常见反模式时给出修复片段。
# 以下字段 v0 解析但不消费——见 §4.3
always: false
triggers:
  globs: ["**/*.rs"]
  keywords: ["clippy", "lint", "review"]
allowed_tools: ["bash", "fs.read"]
---

（body 任意 markdown 内容，会作为 L2 加载时的完整 SKILL.md 文本喂给模型）
```

| 字段 | 类型 | 必填 | v0 行为 |
| --- | --- | --- | --- |
| `name` | string | ✓ | 与目录名一致；不一致时配置加载期 fail-fast（避免清单展示与 `skill` 工具入参对不上） |
| `description` | string | ✓ | 进 L1 清单；建议 ≤ 200 字符（不强校验，仅 warn） |
| `always` | bool | ✗ | 见 §5.1 —— v0 仅打桩，运行时一律按 `false` 处理 |
| `triggers.globs` | `Vec<String>` | ✗ | 见 §4.3 —— v0 解析后忽略 |
| `triggers.keywords` | `Vec<String>` | ✗ | 同上 |
| `allowed_tools` | `Vec<String>` | ✗ | v0 解析后忽略；v1 用于让 ACP 客户端做 tool gating（参考 Anthropic `allowed-tools`） |

> **实现注**：上表的 `always` / `triggers` / `allowed_tools` 在 v0 已**显式占位**（解析但不消费）。与本节原设想"未识别字段一律 warn 并忽略"不同，实现保留 `deny_unknown_fields`：**已知占位字段**（含连字符 `allowed-tools` 别名）接受，**真正未知的键**（如 `tirggers` typo）hard error。理由见顶部"实现现状"块——既吃得下上游 open-standard skill，又不放过必填项拼写错。

### 3.3 body 形态

`SKILL.md` 去掉 frontmatter 后整段都视作 markdown body。defect 不解析 body 里的任何特殊语法（`!`shell``, `@file`, `$ARGS` 等三家都有的 substitution）；这些 v0 不做：

- `$ARGUMENTS` 类 substitution 需要先有"用户 `/cmd skill-name args` 入口"——v0 没接入这条路径（见 §2 对照表）
- `!`shell`` 在加载期跑命令是可执行性 / 沙盒性都很重的功能，留给后续 PR 讨论时机
- `@filename` embed 与 `fs.read` 工具语义重合；模型自己读就行

body 直出原文，不做任何 templating——保持 v0 简单可推理。

### 3.4 加载错误的降级

skill 加载在 SessionStart 之前、配置加载之后。错误分级：

| 错误类型 | 行为 |
| --- | --- |
| frontmatter 解析失败（YAML 语法错） | warn，**不**注册该 skill；其他 skill 继续 |
| 必填字段缺失（`name` / `description`） | 同上 |
| `name` ≠ 目录名 | 同上 |
| 未识别 frontmatter 字段 | warn，字段值忽略；其他字段照常使用 |
| `SKILL.md` 不存在但目录存在 | warn，目录视为非 skill |
| 整个 `.defect/skills/` 目录不存在 | 静默；空 registry |

**永不阻塞 session 启动**——这条与 [hooks §3.5] 的 `SessionStart` 失败降级语义一致。

[hooks §3.5]: ./hooks.md#35-hookerror-与降级

## 4. SkillRegistry

```rust
pub struct SkillRegistry {
    skills: BTreeMap<String, SkillEntry>,
}

pub struct SkillEntry {
    pub descriptor: SkillDescriptor,
    /// SKILL.md 全文（去 frontmatter 后的 body）
    pub body: String,
    /// skill 目录绝对路径，供 `skill` 工具回填给模型
    pub root: PathBuf,
}

#[non_exhaustive]
pub struct SkillDescriptor {
    pub name: String,
    pub description: String,
    pub source: ConfigSource,         // user / project / project-local，沿用 hooks 的来源标签
    pub triggers: SkillTriggers,      // v0 解析后不消费
    pub always: bool,                 // v0 解析后不消费
    pub allowed_tools: Vec<String>,   // v0 解析后不消费
}

#[derive(Default)]
pub struct SkillTriggers {
    pub globs: Vec<String>,
    pub keywords: Vec<String>,
}
```

### 4.1 路径覆盖

skill 来源层与 hooks 一致（详见 [hooks §5.1]）：

```
$XDG_CONFIG_HOME/defect/skills/    # ConfigSource::User
<repo>/.defect/skills/             # ConfigSource::Project
<repo>/.defect/skills.local/       # ConfigSource::ProjectLocal（默认禁用，见 §8）
```

[hooks §5.1]: ./hooks.md#51-文件位置

下层覆盖上层 = **同名 skill 替换**，不 merge。理由：与 hooks 的 append+dedupe 不同——skill body 是不可分的 markdown，按字段合并没有自然语义（你不能把两段说明拼一起说"这就是 code-review skill 了"）。

`SkillRegistry::load` 返回单个有效 skill 列表 + 一份覆盖记录（warning），让用户能看到"project 的 code-review 覆盖了 user 的"。

### 4.2 装配位置

`SkillRegistry` 由 CLI 入口在装配 `DefaultAgentCore` 时构造一次，注入：

```rust
DefaultAgentCoreBuilder::new(...)
    .hook_engine(hook_engine)
    .skill_registry(Arc::new(skill_registry))   // 新增
    .build()
```

后续 SessionStart hook builtin 与 `skill` tool 都从这份 registry 读——单个进程的 skill 状态唯一权威源。

### 4.3 Trigger / always 的桩位置

frontmatter 字段在 `SkillDescriptor` 上**结构化保留**，但 v0 主循环不消费它们：

- `always: true` —— v1 接入后效果是 SessionStart hook 把这条 skill 的 body **直接**拼进 system prompt（不只是清单），等价 Anthropic 的 always-on
- `triggers.globs` —— v1 接入 `UserPromptSubmit` hook 时按用户 cwd 文件树 / 当前 prompt 提及的路径做 glob 匹配
- `triggers.keywords` —— v1 接入时按 prompt 文本 substring 匹配

打桩 = 让 v0 的 schema 与 v1 兼容；用户已经写好的 frontmatter 升级时不用改文件。

## 5. 注入靶点：统一 system prompt suffix

skill 内容进入模型 context 的所有路径都汇到**同一条出口**：通过 hook outcome 的 `append: Vec<ContentBlock>` 字段拼到 session 的 system prompt 后缀。

```text
[base prompt]
[provider/model overlay]
[skill L1 清单]            ←  SessionStart hook builtin 注入
[skill body of always:true] ←  SessionStart hook builtin 注入（v1）
[skill body matched by trigger] ← UserPromptSubmit hook builtin 注入（v1）
```

`skill` 工具（§6.2）的返回路径不走 system prompt——它是工具调用的常规返回，body 出现在 tool result content 里。但 UI / token 审计上**两条路径加起来构成"模型看到的全部 skill 信息"**。

### 5.1 为什么不让 always:true 直接走"读文件 → 拼 base prompt"

技术上等价（base prompt + 文件内容拼接也能塞进 system prompt）。选择走 hook 的理由：

- **观察口子统一**：所有 system prompt 注入都过 hook engine，`tracing-audit` builtin（[hooks §4.1]）能看到 skill 注入；如果 skill 走独立路径，审计要再写一遍。
- **覆盖与去重**：hook engine 已经处理"同 handler 多 layer 注册"的去重；skill 走 hook 自动继承。
- **未来 trust gating**（[hooks §6]）也只需写一次。

[hooks §4.1]: ./hooks.md#41-builtin
[hooks §6]: ./hooks.md#6-信任模型

## 6. 与 hooks / tool 的接口

### 6.1 SessionStart hook builtin：清单注入

`crate::hooks::builtin::SkillManifestHook`：

```rust
pub struct SkillManifestHook {
    registry: Arc<SkillRegistry>,
}

impl HookHandler for SkillManifestHook {
    fn capability(&self) -> HookCapability { HookCapability::Intercept }

    fn handle<'a>(
        &'a self,
        event: &'a HookEvent<'a>,
        _ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<HookOutcome, HookError>> {
        Box::pin(async move {
            let HookEvent::SessionStart { .. } = event else {
                return Ok(HookOutcome::default());  // 其他事件跳过
            };
            let manifest = render_manifest(&self.registry);  // "可用 skills:\n- code-review: ..."
            Ok(HookOutcome {
                append: vec![ContentBlock::text(manifest)],
                ..Default::default()
            })
        })
    }
}
```

注册形式（[hooks §4.1]）：

```toml
[[hooks.session_start]]
handler = { type = "builtin", name = "skill-manifest" }
```

清单形态（v0 简单格式）：

```text
## 可用 Skills

可通过 `skill` 工具按名称加载完整内容：

- **code-review**: 用 cargo / clippy 风格审查 Rust 代码...
- **debug-rust**: ...
```

token 预算（v0）：清单整体不截断；只在 §10 留 `// TODO(v1)` 标记后续按 `name + description` 长度做 1% context 限制（参考 Anthropic 的 `skillListingBudgetFraction`）。

### 6.2 `skill` tool：模型按需加载 L2

普通 `Tool` 实现，注册到 session 的 tool registry：

```rust
pub struct SkillTool {
    registry: Arc<SkillRegistry>,
}

impl Tool for SkillTool {
    fn schema(&self) -> &ToolSchema {
        // {
        //   "name": "skill",
        //   "description": "Load the full body of an available skill by name. Use the names listed in the system prompt.",
        //   "input_schema": {
        //     "type": "object",
        //     "properties": { "name": { "type": "string" } },
        //     "required": ["name"]
        //   }
        // }
        &self.schema
    }

    fn safety_hint(&self, _args: &Value) -> SafetyClass { SafetyClass::ReadOnly }

    fn describe(&self, args: &Value) -> ToolCallDescription {
        // title = "Loading skill {name}"
        // kind = ToolKind::Other
    }

    fn execute(&self, args: Value, _ctx: ToolContext<'_>) -> ToolStream {
        // 1. parse {name: String}
        // 2. registry.lookup(&name) → SkillEntry
        // 3. 输出 ToolEvent::Completed with content =
        //      "Skill: {name}\nLocation: {root}\n\n{body}"
        // 4. 找不到时输出 ToolEvent::Failed("unknown skill: {name}; available: ...")
    }
}
```

要点：
- `safety_hint = ReadOnly`：不写盘 / 不出网，普通 sandbox policy 一律放行。
- `Location: {root}` 行让模型知道 skill 目录绝对路径——后续要读 `scripts/lint-rust.sh` 时可以拼绝对路径喂给 `bash` / `fs.read`。
- 工具名 `skill`（snake_case 与项目其他工具一致），input schema 单字段 `name: string`——足够覆盖 v0；v1 接 `arguments` 时按 [Tool trait `input_schema`](./tool-trait.md#1-toolschema) 的演进规则加字段。

### 6.3 与 capability mode 的关系

`skill` tool 是本地工具——`capabilities` 系统不接入这条路径（[`capabilities.md`](./capabilities.md) 的 search 模式仅控制 search 这一类）。skill 系统本身没有 hosted 形态；provider-side 没 skill 概念。

如果未来某 provider 提供"hosted skill repository"，可以考虑加 `capabilities.skills.mode = local | delegate | disabled`——但目前所有 provider 都没这功能，v0 不留这个旋钮。

## 7. 加载时机

### 7.1 启动期全扫描

`SkillRegistry::load(opts)` 在 CLI 入口 `defect_cli::run()` 早期、`DefaultAgentCore` 装配前调用：

1. 解析三层目录路径（user / project / project-local）
2. 每层 `read_dir` → 对每个子目录尝试读 `SKILL.md`
3. 解析 frontmatter + body，按 §3.4 表降级
4. 构造 `SkillRegistry`，按 §4.1 覆盖语义合并

### 7.2 不做 lazy load

理由：
- skill 数量预期 < 100 / repo（参考 Anthropic / Codex 的实战配置规模），全量扫描在 SSD 上 < 50ms
- lazy load 与 `SkillManifestHook` 在 SessionStart 期就要生成清单的需求矛盾——清单总要全量列名
- frontmatter 解析失败不阻塞启动（§3.4），lazy 模式下错误会变成"模型调 `skill` 工具时才报"，调试体验更差

### 7.3 不做 ad-hoc 重载

`/skill reload` 类命令 v0 不接入。重启 session 即可；v1 再考虑 ACP `notification` 触发的"重新装配 registry + 重发 SessionStart 拼新清单"——这条要先看 hook engine 热加载（[hooks §8] `DefaultHookEngine::reload`）落地后什么形态再决定。

[hooks §8]: ./hooks.md#8-hookengine-trait

### 7.4 不做 FS watcher

Anthropic 有 watcher（mid-session 改 SKILL.md 立即生效）；defect v0 不做。理由：
- watcher 涉及跨平台 inotify / fsevents，与"纯 Rust 单二进制 / 跑在 WASM 也能用"的设计前提（[hooks §0]）冲突
- skill 内容稳定后修改频率低，重启 session 是可接受的代价

## 8. 信任模型

复用 [hooks §6] 的信任语义——`.defect/skills.local/` 默认**禁用**，需要走 `defect skills trust <hash>` 显式信任（CLI 入口与 hooks 共用）。

理由：克隆仓库时若 `.defect/skills.local/` 已经存在（即使 .gitignore 通常排除它），加载未审查的 skill 会把陌生 markdown 拼进系统 prompt，等于让攻击者控制模型的隐性指令。这条与 hook 的攻击面对等。

`<hash>` 算法对单个 skill 取 frontmatter + body 的 SHA-256 截断；user / project 层的 skill 默认信任（与 hooks 同款隐式信任）。

### 8.1 与 hooks trust 的合并

不分两套子命令——`defect skills list` 仅展示 skill 列表与状态，`defect skills trust <hash>` 与 `defect hooks trust <hash>` 走相同的 user-level config 段：

```toml
# user-level config
[[trust.entries]]
kind = "skill"      # or "hook"
hash = "abc123..."
```

具体落地见 [hooks §6] 演进版 / `defect-config` PR；本节仅承诺"skill 与 hook 共用一套信任注册表"。

## 9. 演进路径

### 9.1 v1 待办

- 接入 `triggers.globs` / `triggers.keywords`：在 `crate::hooks::builtin::SkillRouterHook` 内实现 [hooks §9.2] 草稿的匹配逻辑
- 接入 `always: true`：扩展 `SkillManifestHook` 把 always 项的 body 直接拼进 system prompt
- L1 清单 token 预算：参考 Anthropic 的 1% context 限制
- `defect skills list` / `defect skills trust` CLI

### 9.2 v2 候选

- ACP `slash_commands` 接入：把"用户主动 `/skill-name`"暴露给 IDE 客户端（与 [hooks §7.4] PermissionAsk 升级走同一思路——v0 enum 已留，靠后续真实需求驱动）
- skill body 的 `$ARGUMENTS` substitution（依赖 slash_commands 提供 args）
- `allowed_tools` 强制：在 skill body 进 system prompt 时同步告知 sandbox policy 仅允许该 skill 列出的工具——这条与 [`sandbox-policy.md`](./sandbox-policy.md) 的 per-call policy 接口要先对齐

### 9.3 不在路线图上

- `!`shell`` / `@filename` substitution（与 sandbox policy 冲突，无法静态分析风险）
- FS watcher 热加载（§7.4）
- skill 之间的 `depends_on` / `includes`（YAGNI，Anthropic / Codex 都没做）
- LLM-as-router 的 skill 选择（与 [hooks §4.3.2] "PreToolUse 上跑 prompt = 时延翻倍"同款理由——`UserPromptSubmit` 上跑 LLM 同样致命）

## 10. 落地节奏

按下列顺序：

1. 新建 `crates/agent/src/skill.rs`（`SkillRegistry` / `SkillEntry` / `SkillDescriptor` 类型 + frontmatter 解析）
2. 新建 `crates/agent/src/skill/loader.rs`（目录扫描 + §3.4 错误降级 + §4.1 覆盖语义）
3. `DefaultAgentCoreBuilder::skill_registry(Arc<SkillRegistry>)` 注入位
4. `crate::hooks::builtin::SkillManifestHook` 实现（依赖 hooks Phase D engine 已就绪）
5. `defect-tools` 新增 `SkillTool`，注册到 session tool registry
6. CLI 入口 `defect_cli::run()` 装配：扫描 → 构造 registry → 注入 builder
7. e2e：mock provider 跑一遍"清单出现在 system prompt → 模型调 `skill` → body 进 tool result"

测试策略：复用 [`docs/testing/e2e.md`](../testing/e2e.md) 的 mock 框架；frontmatter 解析 / 覆盖语义在 `skill::loader` 模块单测覆盖。

依赖关系：
- 步骤 1-3 不依赖 hooks；可与 hooks Phase D 并行
- 步骤 4 依赖 hooks Phase D（builtin registry + engine 调度）
- 步骤 5-7 依赖步骤 1-4 全部就绪

> v0 仅承诺 §3 / §4 / §5 / §6 的形状与接口；§7-§9 的具体调参在落地 PR 再细化。
