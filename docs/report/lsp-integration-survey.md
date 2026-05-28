# LSP 接入调研报告

## 1. 目标

本文调研三类参考实现的 LSP 管理方式，并结合 defect 当前架构，给出适合本项目的实现建议。

调研对象：

- `docs/coding-reference/codex`
- `docs/coding-reference/opencode`
- `docs/coding-reference/claw-code`

重点关注：

- LSP 生命周期管理
- 状态归属与缓存位置
- 对外暴露方式
- 与主循环/工具系统的耦合方式
- 诊断同步与文档同步策略
- 对 defect 的可借鉴点

---

## 2. 结论摘要

先给结论：

1. **Codex**：公开代码里几乎看不到完整 LSP 产品层实现，至少在当前本地参考副本中，**没有形成清晰的 agent 内建 LSP 管理层**。更像是依赖 IDE/编辑器侧，或相关实现不在本次可见代码范围内。
2. **OpenCode**：有一套**最完整、最值得参考**的 LSP 架构：
   - 内建 LSP 服务层
   - server 自动发现/自动启动
   - client 生命周期管理
   - diagnostics push/pull 混合处理
   - 通过 tool 暴露能力
   - 通过 API / UI 暴露状态
3. **Claw Code**：当前副本里已经有 **LSP registry + tool 接口**，但整体成熟度明显低于 OpenCode，属于**“有接口骨架，真实 server 进程编排和完整协议栈还比较弱/占位”**的状态。
4. **对 defect 最合适的路线**：
   - **第一阶段不要学 Codex 的“不可见实现”**
   - **也不要直接照搬 OpenCode 的整套大系统**
   - **最合适的是：以 OpenCode 为主参考，以 Claw Code 为次参考，先做 session 级 LSP manager + tool 暴露 + diagnostics 缓存的中间方案**

一句话总结：

> 对 defect 来说，最优路径不是“先把 LSP 当成几个零散工具”，也不是“一步到位做成 OpenCode 那种完整 LSP 平台”，而是先做一个 **session 级 LSP 子系统**，再把能力以工具形式暴露给主循环。

---

## 3. Codex：当前可见代码中的 LSP 情况

### 3.1 直接观察

在本地参考副本中，能看到的 LSP 线索主要是：

- `.vscode/settings.json` 中的开发者本地 `rust-analyzer` 设置
- `Cargo.lock` 中存在 `lsp-types`
- 大量与 `diagnostics` 相关的内容，但多数属于：
  - config diagnostics
  - doctor diagnostics
  - TUI / health diagnostics
  - 非代码语义 LSP 诊断

**没有检索到一个明确的、对 agent 主流程开放的 LSP 子系统入口**，比如：

- `src/lsp/...`
- LSP tool
- workspace symbol tool
- definition/references tool
- server lifecycle manager

### 3.2 可推断结论

因此只能做一个保守判断：

- **Codex 当前公开仓库里，并未暴露出可直接借鉴的 LSP 管理实现**；
- 如果真实产品里有更深的 LSP 能力，可能在：
  - IDE 扩展层
  - app-server 协议层
  - 仓库之外的私有/未开源部分

### 3.3 对 defect 的启示

Codex 在本次调研里对 defect 的主要价值不是“LSP 怎么做”，而是一个反例提醒：

- **不要假设产品一定已经有 LSP，只因为它很强**；
- 对公开可维护项目来说，**LSP 能力若要成为系统一等公民，源码里应该能看到清晰的生命周期与状态边界**；
- 如果看不到，那就不能把它当作我们设计的主要参考。

所以：

> Codex 不能作为本项目 LSP 接入方案的主参考样本。

---

## 4. OpenCode：完整的 LSP 内建子系统

OpenCode 是这次调研里最有价值的样本。

---

### 4.1 总体架构判断

从这些路径可以看出它不是“几个 LSP 工具函数”，而是一整套子系统：

- `packages/opencode/src/lsp/lsp.ts`
- `packages/opencode/src/lsp/client.ts`
- `packages/opencode/src/lsp/server.ts`
- `packages/opencode/src/tool/lsp.ts`
- `packages/opencode/src/config/lsp.ts`
- `packages/opencode/src/server/routes/.../file.ts`
- `packages/opencode/src/server/routes/.../instance.ts`
- `packages/app/src/context/...` 中的 `lsp.updated`
- `packages/sdk/openapi.json` 中的 `/lsp`

这说明 OpenCode 的 LSP 不是单纯工具，而是：

- **有独立服务层**
- **有独立 client 层**
- **有 server 定义/发现层**
- **有 config 层**
- **有 UI 事件层**
- **有 HTTP API 暴露层**

这是一个非常清晰的分层方案。

---

### 4.2 状态归属：instance/session 级，而不是进程全局裸单例

`lsp.ts` 中核心状态：

- `clients: LSPClient.Info[]`
- `servers: Record<string, LSPServer.Info>`
- `broken: Set<string>`
- `spawning: Map<string, Promise<LSPClient.Info | undefined>>`

并通过 `InstanceState` 挂载。

这说明：

- LSP 状态属于**当前实例上下文**
- 不是一个简单的全局静态表
- 其粒度接近于 **workspace / instance / session**

这个选择非常重要，因为 LSP 天然依赖：

- root path
- workspace 文件类型
- 当前实例目录
- 当前上下文配置

### 对 defect 的启示

defect 目前最自然的挂点也不是进程级全局，而应当是：

- **session 级**（首选）
- 或将来演进为 workspace-instance 级

而不是把 LSP registry 塞成一个简单 `static OnceLock<HashMap<...>>`。

---

### 4.3 生命周期：按文件懒发现、按 root 启动、去重复用

`lsp.ts` 里最核心的一段逻辑是 `getClients(file)`：

它会：

1. 根据文件扩展名筛 server
2. 调用每个 server 的 `root(file, ctx)` 决定是否适用、根目录在哪
3. 如果已有同 `(root, server.id)` 的 client，就直接复用
4. 如果没有，就走 `spawning` 去重并发启动
5. 启动成功后加入 `clients`
6. 启动失败则写入 `broken`
7. 成功后发 `lsp.updated` 事件

这是一个非常成熟的模式。

#### 它解决了什么问题

- **同一个语言 server 不会重复启动**
- **同一个 root 下多个文件会复用同一 client**
- **并发请求不会把同一 server 拉起多次**
- **失败 server 有 broken 标记，避免疯狂重试**

#### 对 defect 的启示

这几条几乎应该照抄为原则：

1. **按 root + language 去重**
2. **按需懒启动，而不是 session 创建时全量拉起所有 server**
3. **要有 inflight/spawning 去重表**
4. **要有 broken/backoff 状态，避免死循环启动失败**

这意味着 defect 的实现里至少需要：

- `LspManager`
- `ClientKey { root, server_id }`
- `active_clients`
- `spawning_clients`
- `broken_clients`

---

### 4.4 server 层：内建 server catalog + root 规则 + 自动下载能力

`server.ts` 是 OpenCode 的另一个关键点。

每个 server 都有统一形状：

- `id`
- `extensions`
- `root(file, ctx)`
- `spawn(root, ctx, flags)`

而且内置了大量语言：

- TypeScript
- Rust
- Go
- Python (`pyright` / `ty`)
- C/C++ (`clangd`)
- Vue
- Svelte
- Astro
- Lua
- Terraform
- YAML
- Dockerfile
- Kotlin
- Java
- C#
- Razor
- F#
- Julia
- Haskell
- Bash
- Nix
- 等等

#### 特别值得注意的点

##### 1. `root()` 是每个 server 自己决定的
比如：

- Rust：优先找 workspace root / Cargo.toml
- TS：看 lockfile / package manager markers
- Go：看 `go.work` / `go.mod`
- Python：看 `pyproject.toml` / `requirements.txt`

这说明 OpenCode 没有用一个统一的“项目根目录”硬套所有语言，而是**server 自己决定 attach root**。

##### 2. `spawn()` 不只是启动命令
它还能：

- 找本地命令
- 找 npm 包
- 下载 release binary
- 安装缺失依赖
- 注入 initialization options

这是很重的产品能力。

#### 对 defect 的启示

这里要分两层看：

##### 值得借鉴
- server catalog 抽象
- 每个 server 自定义 root 规则
- `spawn()` 封装启动细节

##### 不建议第一阶段照搬
- 自动下载语言服务器
- 多生态安装器（npm/go/dotnet/gem/github release）

因为 defect 现在是：

- headless agent
- Rust 核心较轻
- 当前代码库强调紧凑和可维护

如果一开始就把 LSP 接成“自动下载安装各种 server”，复杂度会暴涨。

所以 defect 第一阶段更适合：

> **只支持“显式配置 server command” + 少量 builtin 约定”，不做自动安装器。**

---

### 4.5 client 层：真实 LSP 协议、文档同步、diagnostics push/pull 混合

`client.ts` 是 OpenCode 最强的一部分。

能看到它使用：

- `vscode-jsonrpc/node`
- `createMessageConnection`
- `StreamMessageReader/Writer`
- `initialize` / `initialized`
- `textDocument/didOpen`
- `textDocument/didChange`
- `workspace/didChangeWatchedFiles`
- `textDocument/publishDiagnostics`
- `textDocument/diagnostic`
- `workspace/diagnostic`
- `workspace/configuration`
- `workspace/workspaceFolders`

这意味着它是真的在做**完整 LSP client**，而不是 registry 占位。

#### 关键能力 1：文件 touch/open/change
OpenCode 提供：

- `touchFile(file, diagnostics?: "document" | "full")`

它会：

- 把文件读出来
- 如果没打开过：走 `didOpen`
- 如果已打开：走 `didChange`
- 支持增量/全量同步策略

这很关键，因为它保证了：

- agent 发起 LSP 查询前，server 至少看过最新文件内容
- diagnostics 能围绕“当前文件状态”工作

#### 关键能力 2：diagnostics 双通道
OpenCode 同时维护：

- `pushDiagnostics`
- `pullDiagnostics`

并做 merge。

原因是：

- 有的 server 主动 `publishDiagnostics`
- 有的 server 需要 client 主动 request `textDocument/diagnostic`
- 有的 server 两者都支持

这是非常工程化的现实处理。

#### 关键能力 3：等待 diagnostics 成熟
它不只是“发个 didOpen 然后立即问结果”，而是：

- 监听发布事件
- 追踪版本
- 追踪注册能力变化
- 用 timeout / debounce 等待“足够新”的 diagnostics

这套机制比“调用一次然后拿缓存”稳很多。

### 对 defect 的启示

这是 OpenCode 最值得借鉴、同时也最容易超出第一阶段边界的地方。

对 defect 来说：

#### 第一阶段建议借鉴
- 真实 LSP client 抽象，而非假 registry
- 文档至少支持 `didOpen` / `didChange`
- diagnostics 要有缓存
- 查询前先 `touch_file`

#### 第一阶段不必完全照搬
- push/pull 诊断全覆盖
- dynamic registration 的完整支持
- full/document diagnostics wait orchestration

也就是说 defect 可以先做一个“缩水但真实”的版本：

- 先支持 push diagnostics
- pull diagnostics 作为第二阶段
- 先支持 `didOpen` / 全量 `didChange`
- 先不做复杂 debounce / registration refresh

---

### 4.6 tool 层：LSP 作为统一工具暴露

`tool/lsp.ts` 显示 OpenCode 最终把 LSP 能力暴露成一个统一 tool：

参数大致是：

- `operation`
  - `goToDefinition`
  - `findReferences`
  - `hover`
  - `documentSymbol`
  - `workspaceSymbol`
  - `goToImplementation`
  - `prepareCallHierarchy`
  - `incomingCalls`
  - `outgoingCalls`
- `filePath`
- `line`
- `character`
- `query`

执行流程：

1. 路径归一化
2. 权限检查：`permission: "lsp"`
3. 文件存在性检查
4. `lsp.hasClients(file)`
5. `lsp.touchFile(file, "document")`
6. 执行具体 query
7. 返回结构化结果

#### 这说明什么
OpenCode 的核心设计不是“把 LSP 深埋在主循环里自动发生”，而是：

- **LSP 先是一个服务层**
- **再通过工具统一暴露给 agent**

这是 defect 非常应该借鉴的点。

因为这条路线：

- 与当前 `Tool` 抽象天然兼容
- 不会污染 ACP 层
- 调试容易
- 可逐步扩展 operation 集合

---

### 4.7 API / UI 暴露：LSP 还是系统状态的一部分

OpenCode 不只把 LSP 当工具，还把它当状态：

- `/lsp` HTTP endpoint
- `lsp.updated` 事件
- UI status popover 中的 LSP tab
- workspace symbol HTTP route

这说明在 OpenCode 里，LSP 不只是 agent 内部依赖，而是**用户可见子系统**。

### 对 defect 的启示

defect 当前是 headless ACP agent，不做 UI，因此：

- **不需要一开始就做 UI 级 LSP 面板**
- 但可以预留：
  - session 级 `list_lsp_servers()`
  - `lsp status` 类接口
  - event 订阅点

这样 ACP / 外部前端将来若想展示，也有基础。

---

## 5. Claw Code：轻量 registry + tool 暴露，成熟度中等

Claw Code 当前副本中，LSP 已经接进系统，但形态明显比 OpenCode 简化很多。

---

### 5.1 已有内容

关键文件：

- `rust/crates/runtime/src/lsp_client.rs`
- `rust/crates/tools/src/lib.rs`
- `rust/PARITY.md`

在 `PARITY.md` 里，项目自己写的是：

- `runtime::lsp_client` + `tools`
- registry + dispatch for diagnostics, hover, definition, references, completion, symbols, formatting

所以它自我定位是“LSP 已有一套 registry+dispatch”。

---

### 5.2 registry 设计：偏静态、偏缓存、偏占位

`lsp_client.rs` 的核心是：

- `LspRegistry`
- `HashMap<String, LspServerState>`
- 每个 server state 包含：
  - `language`
  - `status`
  - `root_path`
  - `capabilities`
  - `diagnostics`

并且它支持：

- `register(language, status, root_path, capabilities)`
- `find_server_for_path(path)`：按扩展名映射语言
- `add_diagnostics`
- `get_diagnostics`
- `clear_diagnostics`
- `dispatch(action, path, line, character, query)`

#### 但问题是
它的 `dispatch()` 对非 diagnostics 的行为基本是：

- 检查有没有匹配 server
- 检查 server status 是否 connected
- **返回一个“已派发”的结构化占位结果**
- 注释明确写着：
  - “actual LSP JSON-RPC calls would go through the real LSP process here”

也就是说它目前更多是：

- **状态机壳子**
- **能力路由骨架**
- **真实协议调用尚未完整落地**

这点非常重要。

### 对 defect 的启示

Claw Code 的价值不是“能直接照搬完整实现”，而是：

- 它证明了 **LSP 可以先以 registry + tool 形式接入 agent**
- 说明这条演进路线是自然的
- 但也提醒我们：**如果只有 registry 而没有真实 client 生命周期，价值有限**

---

### 5.3 tool 层：统一 LSP 工具入口

在 `tools/src/lib.rs` 里，可以看到它有一个 `LSP` 工具，描述是：

- 查询代码智能（symbols / references / diagnostics）

这和 OpenCode 很像：

- 用一个统一工具承载多种 operation

但 Claw Code 的做法更轻：

- tool 调用 `global_lsp_registry()`
- 然后 `run_lsp()` 转发到 registry dispatch

这里明显能看出它当前是：

- **工具层完成了统一入口**
- **runtime registry 完成了最小状态管理**
- **但还缺 OpenCode 那种真实 client/server 编排深度**

### 对 defect 的启示

这说明 defect 的第一阶段完全可以采取：

- 新建 LSP manager
- 暴露统一 `lsp` 工具
- operation 枚举先做：
  - diagnostics
  - hover
  - definition
  - references
  - document symbols

这个方向是可行的。

---

### 5.4 状态归属问题：global registry 倾向明显

Claw Code 里有 `global_lsp_registry()` 这种接口痕迹，说明它在当前阶段偏向：

- 全局单例
- 进程级共享

对于轻量原型这没问题，但对 defect 来说要谨慎。

#### 为什么 defect 不宜直接照搬全局单例
因为 defect 有：

- 多 session
- per-session cwd
- ACP 委托 fs/shell
- 将来可能 resume/load session

如果把 LSP registry 做成进程级全局：

- root 混杂风险大
- 不同 session 的权限/上下文边界不清楚
- diagnostics 归属不清晰

所以：

> Claw Code 在“tool 暴露形式”上可借鉴，但在“全局 registry”上不适合 defect 直接照搬。

---

## 6. 三家对比

| 维度 | Codex | OpenCode | Claw Code | 对 defect 的参考价值 |
|---|---|---|---|---|
| 公开可见的 LSP 子系统 | 弱/不明确 | 强 | 中 | OpenCode 最高 |
| 生命周期管理 | 不清晰 | 完整 | 较弱 | OpenCode 最高 |
| server catalog | 不明显 | 完整 | 很弱 | OpenCode 最高 |
| 真实 LSP client | 不明显 | 完整 | 较弱/占位 | OpenCode 最高 |
| diagnostics 管理 | 不清晰 | push/pull 混合 | 缓存型 | OpenCode 主参考，Claw 次参考 |
| tool 暴露 | 不明显 | 有 | 有 | OpenCode + Claw 都可参考 |
| API / UI 状态暴露 | 不明显 | 有 | 基本无 | OpenCode 可做远期参考 |
| 适配多 session/workspace | 不清晰 | 较强 | 一般 | OpenCode 更适合 |

总结：

- **主参考：OpenCode**
- **次参考：Claw Code**
- **Codex：本次不作为核心参考**

---

## 7. defect 当前架构下的实现建议

结合当前 defect 代码结构：

- `defect-agent`：核心 trait / session / tool / turn loop
- `defect-tools`：内建工具
- `defect-mcp`：外部工具适配
- `defect-acp`：协议桥接
- `defect-cli`：装配入口

我建议 defect 的 LSP 接入分三阶段。

---

### 7.1 第一阶段：session 级 LSP manager + 统一 lsp tool

#### 核心原则

1. **LSP 是一个 session 级服务，不是 ACP 概念**
2. **LSP 能力先通过 tool 暴露**
3. **不要先做 UI / ACP 扩展 / 自动 prompt 注入**
4. **先做真实 client，不做假 registry**

#### 推荐 crate 结构

新增：

```text
a crates/lsp/
```

建议模块：

```text
crates/lsp/src/
├── lib.rs
├── manager.rs        # session/workspace 级 LSP 管理器
├── catalog.rs        # server 定义、root 规则、配置装配
├── client.rs         # JSON-RPC / stdio client
├── document.rs       # open/change/version 管理
├── diagnostics.rs    # 诊断缓存
├── query.rs          # hover/definition/references/symbols
└── test.rs
```

#### 建议对 agent 增加的能力

不是直接改 ACP，而是增加 session 侧依赖装配能力。

目前已有：

- `SessionToolFactory`
- `ToolRegistry`

建议增加：

- `CompositeSessionToolFactory`

这样 CLI 可以同时装：

- MCP tool factory
- LSP tool factory

#### 第一批 operation

建议只做：

- `diagnostics`
- `hover`
- `definition`
- `references`
- `document_symbols`

先不做：

- rename
- code action
- formatting
- workspace edit
- call hierarchy

#### 推荐工具形式

新增单一工具：

- `lsp`

参数：

- `operation`
- `path`
- `line`
- `character`
- `query`

理由：

- 与 OpenCode 一致
- 比拆成一堆 tool 更稳
- schema 演进简单

---

### 7.2 第二阶段：diagnostics 缓存与自动触发

第一阶段跑通后，再增强：

1. 文件读写工具成功后，通知 LSP manager `touch/open/change`
2. session 内缓存当前 diagnostics
3. 提供：
   - `list_lsp_servers`
   - `lsp_status`
4. 允许在 turn 前主动拉一次 diagnostics

此时仍然**不必修改 ACP 事件模型**。

可以先让模型自己调用 `lsp` 工具拿 diagnostics。

---

### 7.3 第三阶段：将 diagnostics 作为 session 上下文的一部分

只有在前两阶段稳定后，再考虑：

- `AgentEvent::DiagnosticsUpdated`
- prompt 前自动注入关键 diagnostics 摘要
- storage 持久化最近诊断快照
- ACP 前端暴露 LSP 状态

这是远期增强，不建议现在就做。

---

## 8. defect 不建议直接照搬的点

### 8.1 不建议照搬 OpenCode 的自动下载全家桶

OpenCode 为很多语言 server 自动：

- npm 安装
- go install
- dotnet tool install
- gem install
- GitHub release 下载

这非常强，但对 defect 当前阶段不合适：

- 复杂度高
- 失败模式多
- 平台兼容面大
- 与“紧凑、节省资源”的项目目标不完全一致

建议 defect 第一阶段只做：

- 显式配置 command
- 少量常见默认名查找（如 `rust-analyzer`、`gopls`）
- 找不到就报清晰错误

### 8.2 不建议照搬 Claw Code 的全局 registry

defect 更适合：

- session 级 manager
- 每个 session 绑定自己的 cwd/root/client set

而不是进程全局单例。

### 8.3 不建议第一阶段就做 workspace symbol HTTP / UI 状态页

因为 defect 是 headless ACP agent，不是自带 console/web UI 产品。

---

## 9. 推荐设计草案

### 9.1 类型边界建议

```rust
pub trait LspService: Send + Sync {
    fn touch_file(&self, path: &Path) -> BoxFuture<'_, Result<(), LspError>>;
    fn diagnostics(&self, path: Option<&Path>) -> BoxFuture<'_, Result<Vec<LspDiagnostic>, LspError>>;
    fn hover(&self, path: &Path, position: LspPosition) -> BoxFuture<'_, Result<Option<LspHover>, LspError>>;
    fn definition(&self, path: &Path, position: LspPosition) -> BoxFuture<'_, Result<Vec<LspLocation>, LspError>>;
    fn references(&self, path: &Path, position: LspPosition) -> BoxFuture<'_, Result<Vec<LspLocation>, LspError>>;
    fn document_symbols(&self, path: &Path) -> BoxFuture<'_, Result<Vec<LspSymbol>, LspError>>;
}
```

然后由：

- `DefaultSession` 持有 `Option<Arc<dyn LspService>>`
- `LspTool` 调用该 service

### 9.2 生命周期建议

- session 创建时：构造 manager，但不启动所有 server
- 首次对某路径调用 LSP 时：
  - 通过 extension/root rule 找 server
  - 若未启动则懒启动
- session drop 时：统一 shutdown 所有关联 server

### 9.3 diagnostics 策略建议

第一阶段：

- `didOpen`
- 全量 `didChange`
- 接 `publishDiagnostics`
- 内存缓存

第二阶段：

- 按需 `textDocument/diagnostic`
- 聚合 `workspace/diagnostic`

### 9.4 配置建议

在 defect config 中新增：

```toml
[lsp]
enabled = true

[lsp.servers.rust]
command = ["rust-analyzer"]
extensions = ["rs"]
root_markers = ["Cargo.toml", "Cargo.lock"]

[lsp.servers.go]
command = ["gopls"]
extensions = ["go"]
root_markers = ["go.mod", "go.work"]
```

不要一开始把 builtin catalog 写得太满，先从：

- rust
- typescript
- python
- go

开始就够了。

---

## 10. 最终建议

### 推荐结论

defect 的 LSP 接入应当：

1. **以 OpenCode 为主参考样本**
2. **以 Claw Code 为“轻量 agent 中 LSP tool 形式”的补充参考**
3. **不将 Codex 作为本次实现的核心参考**

### 推荐路线

#### v1
- 新增 `defect-lsp` crate
- session 级 LSP manager
- 统一 `lsp` 工具
- 支持 diagnostics / hover / definition / references / document_symbols
- 仅显式 server 配置，不做自动安装

#### v2
- 文档变更自动同步
- diagnostics 缓存增强
- LSP 状态查询接口

#### v3
- diagnostics 注入上下文
- 事件模型集成
- ACP 前端状态暴露

### 为什么这是最适合 defect 的方案

因为它同时满足：

- 与当前 `Tool` / `Session` 架构兼容
- 不污染 ACP 薄桥接层
- 不必一次性承担 OpenCode 的全部复杂度
- 比 Claw Code 的纯 registry 占位更有真实价值
- 允许未来逐步演进到更强的语义系统

---

## 11. 一句话结论

> 如果要给 defect 接 LSP，正确路线是：**参考 OpenCode 的“session/instance 级 LSP 服务 + tool 暴露 + diagnostics 缓存”架构，但在第一阶段收缩范围，不做自动下载安装，也不做全局 registry。**
