# `bash` 内置工具设计

`bash` 是 v0 唯一的命令执行工具：跑一条 shell 命令、流式回写 stdout/stderr、按超时或取消打断。本文沉淀工具的形状、与 ACP 的对位、以及 v0 故意不做的边界。

设计原则按依赖顺序：

1. **以 ACP 为导向**——产出的字段直接对位 [`ToolCallUpdateFields`] / [`ToolCallContent`]，不另造内部结构。
2. **在 [`Tool`] 抽象内做最小工具**——分类、权限、cancel 都已经由 [tool-trait] / [sandbox-policy] 决定；本工具只做"跑命令并把输出 patch 上 wire"。
3. **故意不做沙箱**——v0 的 `bash` 直接 spawn 子进程，依赖 [`SandboxPolicy::ask_writes`] 让用户兜底；OS 级隔离按 [`sandbox-policy.md` §8](./sandbox-policy.md) 演进，不在 v0 实现。

[`ToolCallUpdateFields`]: https://docs.rs/agent-client-protocol-schema/0.13/agent_client_protocol_schema/v1/struct.ToolCallUpdateFields.html
[`ToolCallContent`]: https://docs.rs/agent-client-protocol-schema/0.13/agent_client_protocol_schema/v1/enum.ToolCallContent.html
[`Tool`]: ./tool-trait.md
[tool-trait]: ./tool-trait.md
[sandbox-policy]: ./sandbox-policy.md
[`SandboxPolicy::ask_writes`]: ./sandbox-policy.md#5-v0-内置-policy

## 1. 工具名片

```rust
ToolSchema {
    name: "bash".to_string(),
    description: "Run a non-interactive shell command. \
                  Captures stdout and stderr; returns combined output and exit code. \
                  Times out after `timeout_ms` (default 30s; max 600s).".to_string(),
    input_schema: json!({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute (passed to `sh -c` on unix, `cmd /C` on windows)."
            },
            "workdir": {
                "type": "string",
                "description": "Optional working directory. Must be inside the session cwd; \
                               relative paths resolve against the session cwd. Defaults to the session cwd."
            },
            "timeout_ms": {
                "type": "integer",
                "minimum": 1,
                "maximum": 600000,
                "description": "Per-call timeout in milliseconds. Defaults to 30000."
            }
        },
        "required": ["command"]
    }),
}
```

字段取舍：

- **没有 `args: Vec<String>`**——v0 一律走 `sh -c "<command>"`。给 LLM 解释 argv vs cmdline 是徒增复杂度；让模型自己写 shell 行更自然。代价是命令注入归 LLM 自己负责（这本来就是 LLM tool 调用的现实）。
- **没有 `env`**——子进程继承 agent 进程 env。需要时由 LLM 在 `command` 里 `FOO=bar cmd ...`；同时避免 LLM 通过 env 偷渡凭证。
- **没有 `stdin`**——v0 不支持向命令喂入数据。需要的话先 `printf '...' >/tmp/x` 再 `cmd </tmp/x`。
- **`workdir` 必须在 `cwd` 子树**——上层 [`ToolContext::cwd`] 是 session 唯一可信 root；超出是 v0 不允许的（参见 §5.1）。
- **`timeout_ms` 默认 30s / 上限 10 分钟**——同一 turn 内的工具调用应当短跑；长任务由 LLM 显式拆分（`run_in_background` / 后台脚本）等到后续工具引入。codex 的 `DEFAULT_EXEC_COMMAND_TIMEOUT_MS = 10_000`（10s）更紧，但我们的常见用例是 `cargo test` / `pnpm build` 这种动辄 20s+ 的命令，10s 命中率低，提到 30s 让 LLM 不必每次显式传 timeout。

[`ToolContext::cwd`]: ./tool-trait.md#6-toolcontext

## 2. 安全等级（`safety_hint`）

```rust
fn safety_hint(&self, _args: &Value) -> SafetyClass {
    SafetyClass::Destructive
}
```

**一律返回 `Destructive`**——v0 不解析命令文本去推断"这条 `ls` 是 ReadOnly"；shell 命令的语义无法从字符串可靠分类（`ls > /etc/passwd` 是 Destructive，`rm -rf .` 是 Destructive 且严重，`bash -c '$EVIL'` 不可静态判断）。

代价：用户用 [`AskWritesPolicy`] 时**每条** bash 命令都会触发 Ask。这是 v0 故意付出的代价——比"误判 ReadOnly 让 destructive 静默放行"安全。`AllowAlways` 让用户在第一次 Ask 后 opt-in 到信任模式。

未来演进（**不在 v0**）：codex 走的路是 [`shell-command/parse_command.rs`](../coding-reference/codex/codex-rs/shell-command/src/parse_command.rs) 那种 2500 行命令解析器；我们会复用 codex 的 [`execpolicy`](../coding-reference/codex/codex-rs/execpolicy/) 而不是自己写。轨迹见 §8。

[`AskWritesPolicy`]: ./sandbox-policy.md#5-v0-内置-policy

## 3. `describe(args)`：UI 自描述

```rust
ToolCallUpdateFields {
    title: Some(format!("$ {}", truncate(command, 80))),
    kind:  Some(ToolKind::Execute),
    locations: workdir.map(|p| vec![ToolCallLocation { path: p, line: None }]),
    content:   None,    // 执行期才填
    raw_input: None,    // 主循环填
    raw_output: None,   // 终态填
    status:    None,    // 主循环填
}
```

- `title` 用 `"$ <command>"`，截到 80 字符。客户端 UI 一眼能看出"这是一条 shell 命令"。
- `kind = Execute` 命中 ACP 既有的语义。
- `locations` 仅在用户传了 `workdir` 时填——给 ACP 客户端的 follow-along 一个落脚点。
- `content` / `raw_output` 在执行期 / 终态由 [`execute`](#4-execute) 填，而不是 `describe`。

## 4. `execute`

```text
                tokio::process::Command::new(sh).arg("-c").arg(command)
                    .current_dir(workdir).stdout(Pipe).stderr(Pipe).spawn()
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
       stdout chunks       stderr chunks       wait_with_status
            │                   │                   │
            └────────► merge into one accumulator ◄─┘
                                │   (cap at 1 MiB; overflow tracked)
                                ▼
            ┌───────────── child exited / cancel / timeout ─────────────┐
            ▼                            ▼                              ▼
        exit==0                       exit!=0                       cancelled
            │                            │                              │
            ▼                            ▼                              ▼
   ToolEvent::Completed         ToolEvent::Completed              ToolEvent::Failed(
   ({ content: Text(buf),       ({ content: Text(buf+               ToolError::Canceled
     raw_output: { exit:0 } })    "\n[exit code: N]"),               )
                                  raw_output: { exit:N } })
```

v0 只发**一帧** `Completed`（不发中间 `Progress`），原因见 §4.2。`Failed(Canceled)` 也是终态唯一帧。

### 4.1 进程生成

```rust
#[cfg(unix)]
let mut cmd = Command::new("/bin/sh");
cmd.arg("-c").arg(&command);

#[cfg(windows)]
let mut cmd = Command::new("cmd");
cmd.arg("/C").arg(&command);

cmd.current_dir(&workdir)
    .stdin(Stdio::null())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .kill_on_drop(true);
```

设计点：

- **`stdin = null`**：拒绝交互式命令尝试读 stdin（不会卡死）。`apt install` 这类无法直接跑——LLM 必须加 `-y`。
- **`kill_on_drop`**：drop `Child` 时 tokio 自动 SIGKILL。配合 [`ToolContext::cancel`] 让取消路径无泄漏。
- **不开新 process group / setpgid**：v0 不实现"杀整棵进程树"。代价：`cmd1 | cmd2` 之类的 pipeline 在取消时只杀 `sh`，子进程可能被 `init` 接管。trade-off：避免 v0 引入 `nix::setpgid` / job-control 复杂度。codex 在 `rmcp-client/src/stdio_server_launcher.rs` 用 `Command::process_group(0)` + `kill_process_group()` 已经把演进路径走通，到 v1 时整段抄过来即可（参见 §8）。
- **直接 SIGKILL（不做 SIGTERM 优雅期）**：`kill_on_drop` 等价于立即 SIGKILL。代价：子进程没机会做清理（写盘、删 lockfile、关连接）。opencode 的 `killTree` 是先 SIGTERM、3s 后才升级 SIGKILL（`shell.ts:28-56`），更对得起跑数据库 / 长服务的场景。v0 选 SIGKILL 是因为我们假定 turn 内 bash 都是短命令（编译器、测试、脚本），优雅退出收益小；演进项见 §8。

### 4.2 输出捕获策略

**stdout / stderr 合并成一条文本流**，按 ACP `ToolCallContent::Content(Content::new(text))` 推送（即标准的 `ContentBlock::Text`）。**不用** `ToolCallContent::Terminal`——后者是 ACP `terminal/create` 反向请求建出来的"持久终端"，agent 要先跟客户端协商出 `TerminalId` 才能引用，v0 没接 terminal 协议。等 `terminal` 工具引入（见 §8）时再上 `Terminal` 形态。

合并理由：

- LLM 看到的工具输出几乎都是"shell 命令输出"心智模型——分两条流让模型很难判断时序。
- 客户端 UI 渲染本来就是一个滚动框；分流没有展示意义。
- 真要分流 ACP 没有 `Stderr` content kind；自己造 wire 字段违背 §设计原则。

合并算法：用 [`tokio::io::BufReader::read_until(b'\n')`] 各自读一行，扔进同一个 `mpsc::channel<Vec<u8>>`。`select!` 在 stdout/stderr/timeout/cancel 之间多路选择。

**buffer 上限**：单条 bash 输出限 1 MiB（`MAX_OUTPUT_BYTES`），与 codex `exec.rs:68` 的 `DEFAULT_OUTPUT_BYTES_CAP` 同量级。超过后的字节直接 drop，最终 `Completed` content 末尾追加 `[output truncated; remaining N bytes dropped]`。理由：

- 防止 LLM context 被一条 `find /` 灌满。
- 防止 agent 进程内存被 `cat largefile.bin` 打爆。
- 1 MiB 是个保守上限——LLM 通常只读 head/tail；要更长输出 LLM 自己写 `| head -200`。

**v0 不做 spill-to-disk**：opencode `shell.ts:435-596` 用双缓冲（内存留 tail、超出 spill 到 `/tmp` 临时文件、metadata 暴露文件路径），让 LLM 即便看不到全量也能 grep/tail 后半段。我们 v0 直接 drop 是为了省掉"临时文件生命周期与 turn / session 解耦"这一摊；演进项见 §8。

**v0 单发 `Completed`，不发 `Progress`**：ACP `ToolCallUpdateFields::content` 是 *replace* 语义（每次 update 整段重写），不是 append。要做"流式增量看输出"得每次 Progress 都发整段累积 buffer，对 1 MiB 量级 + 多个 Progress 帧，wire bytes 是 `O(N²)`——`yes` 这种命令直接打爆带宽。v0 选单发：跑完一次性把整段 content 装进 `Completed`。**演进项见 §8**："增量 stream"等到 ACP 给 content 加 append 语义、或我们引入 Terminal 形态。

[`tokio::io::BufReader::read_until(b'\n')`]: https://docs.rs/tokio/latest/tokio/io/trait.AsyncBufReadExt.html#method.read_until

### 4.3 终态

```rust
struct BashOutput {
    exit_code: Option<i32>,        // None when killed by signal
    timed_out: bool,
    truncated_bytes: u64,
}
```

`Completed.raw_output = serde_json::to_value(BashOutput { ... })`——给 LLM / 客户端机读字段。

终态映射规则：

| 退出形态 | event | raw_output.exit_code | content 末尾追加 |
| --- | --- | --- | --- |
| `exit_code == 0` | `Completed` | `0` | （无） |
| `exit_code != 0` | `Completed`（**仍然是 Completed**） | `N` | `\n[exit code: N]` |
| 信号杀掉 | `Completed` | `null`（带 `signal: SIGNAME`） | `\n[killed by signal: SIGNAME]` |
| 超时 | `Completed` | `null`，`timed_out: true` | `\n[timed out after Xms]` |
| `ctx.cancel` 取消 | `Failed(ToolError::Canceled)` | — | — |
| spawn 自身失败 | `Failed(ToolError::Execution(io_err))` | — | — |

为什么"非零退出"是 `Completed` 而不是 `Failed`：

1. shell 命令以非零退出表达**业务结果**（`grep` 没找到 → exit 1，`test -f` 文件不存在 → exit 1）。这不是工具调用失败。
2. ACP wire 上 `status: Failed` 会让客户端 UI 染红——非零退出不该一律染红。
3. LLM 看到 exit code 后能自己决定下一步（重试 / 换命令 / 报告用户），不需要 agent 把它升级成 `ToolError`。

只有"agent 自身没法跑这条命令"（spawn 失败、非法 args、cancel）才走 `Failed`。

### 4.4 取消路径

```rust
tokio::select! {
    biased;
    _ = ctx.cancel.cancelled() => {
        // 1. drop child → kill_on_drop 触发 SIGKILL
        // 2. flush any buffered output as final Progress
        // 3. emit ToolEvent::Failed(ToolError::Canceled)
        return;
    }
    _ = sleep(timeout) => { /* §4.3 timeout 分支 */ }
    status = child.wait() => { /* §4.3 exit 分支 */ }
}
```

`biased` 让取消优先级最高：即便 stdout 一直有输出、`select!` 也优先消费 cancel。这与 `tool-trait.md` §6 "工具实现应在长循环 / await 点检查 cancel" 的契约一致。

## 5. 与 sandbox policy / 路径约束的边界

### 5.1 路径校验

`workdir` 由工具自己校验（不指望 sandbox 兜底——v0 没有 sandbox）。规则：

```rust
fn resolve_workdir(cwd: &Path, requested: Option<&str>) -> Result<PathBuf, ToolError> {
    let target = match requested {
        None => cwd.to_path_buf(),
        Some(s) => {
            let p = Path::new(s);
            if p.is_absolute() { p.to_path_buf() } else { cwd.join(p) }
        }
    };
    let canon = std::fs::canonicalize(&target)
        .map_err(|e| ToolError::InvalidArgs(BoxError::new(e)))?;
    let cwd_canon = std::fs::canonicalize(cwd).unwrap_or_else(|_| cwd.to_path_buf());
    if !canon.starts_with(&cwd_canon) {
        return Err(ToolError::InvalidArgs(BoxError::new(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("workdir {} escapes session cwd {}", canon.display(), cwd_canon.display()),
        ))));
    }
    Ok(canon)
}
```

- **canonicalize 防 symlink 越狱**：`workdir = "/tmp/link"` 而 `link → ../../etc` 时被堵住。
- **跨平台注意**：Windows 上 `canonicalize` 给 `\\?\` 前缀；`starts_with` 仍然正确（同前缀）。
- **越界返回 `InvalidArgs`** 而不是 `Execution`：让主循环把"workdir 越界"送回 LLM 改 args（参见 [tool-trait §7](./tool-trait.md#7-toolerror)）。

`command` 文本本身不做校验——v0 信任 [`SandboxPolicy`] 拦下整次调用。命令注入等问题等 [`execpolicy`](../coding-reference/codex/codex-rs/execpolicy/) 集成（§8）。

### 5.2 与 [`PolicyDecision`] 的协作

工具自己只产 `Destructive`。policy 决定 `Allow` / `Deny` / `Ask`：

- 用户的 `--policy open`（cli 默认 echo provider 时也走这个）→ 一律 Allow，命令直接跑
- 用户 `--policy ask_writes`（默认）→ 每条 bash 都 Ask；用户选 `AllowAlways` 后 policy 内部记 `tool_name="bash"` 加白
- 用户 `--policy read_only` → 一律 Deny（bash 不该跑）；主循环把 "denied by policy" 当 tool_result 喂回 LLM

[`PolicyDecision`]: ./sandbox-policy.md#2-决策模型

## 6. 错误分类映射

| 来源 | ToolError variant | 主循环处理 |
| --- | --- | --- |
| args JSON 反序列化失败 | `InvalidArgs(serde_err)` | 喂回 LLM 让它修 args |
| `workdir` 越界 / 不存在 | `InvalidArgs(io_err)` | 喂回 LLM |
| `spawn` 失败（PATH 上没 `sh`、权限不够） | `Execution(io_err)` | 计入失败、按 policy 决定 retry |
| 子进程跑完（任意 exit code） | **不是错误**，走 `Completed` | LLM 看 exit code 决定下一步 |
| `ctx.cancel` 触发 | `Canceled` | 不计 retry、不报告失败 |

理由见 [tool-trait §7](./tool-trait.md#7-toolerror) 与 §4.3。

## 7. 测试矩阵

每条都写成 `#[tokio::test]`，放在 `crates/tools/src/bash/tests.rs`。

| # | 场景 | 验证 |
| --- | --- | --- |
| 1 | `command="echo hello"` | exit_code=0；content 含 `"hello"`；至少 1 个 Progress + 1 个 Completed |
| 2 | `command="echo err >&2; exit 3"` | exit_code=3；content 同时含 stderr；event 仍是 Completed（不是 Failed） |
| 3 | `command="sleep 5"`，`timeout_ms=100` | content 末尾含 `[timed out after 100ms]`；timed_out=true；event 是 Completed |
| 4 | `command="sleep 5"`，外部 `cancel.cancel()` | event 是 `Failed(Canceled)`；耗时 < 200ms（确认 kill_on_drop 生效） |
| 5 | `command="yes"`（无限输出） | 不爆内存；最终 Completed 的 content 长度 ≤ 1 MiB；末尾有 `[output truncated; ...]` |
| 6 | `workdir="../../etc"` | event 是 `Failed(InvalidArgs)`，没 spawn 子进程 |
| 7 | `workdir="subdir"`（cwd 子目录存在） | spawn 成功，pwd 输出该子目录的 canonical 路径 |
| 8 | spawn 失败（mock：`/bin/sh` 不存在） | event 是 `Failed(Execution)`；source 含 io::Error |
| 9 | `command="cat"`（试 stdin 阻塞） | 立刻 EOF，cat 跑完；exit_code=0；不卡死 |
| 10 | 真实 e2e：deepseek 收到 "list cwd" prompt → 调 `bash` → 看到 ls 输出 → 总结回复 | TurnEnded 是 `EndTurn`；中间至少一次 `ToolCallStarted/Finished` |

#1, #5, #6 是回归基线；#3, #4, #9 是 v0 必须当心的 hang trap；#10 由 `crates/cli/examples/deepseek_e2e.rs` 扩展（与 `echo` 工具并列）。

## 8. v0 不做（演进口子）

下列每条都列出了 v0 的现状、对应风险、以及参考实现里已经把路趟通的位置——v1 时直接抄，不重新设计。

- **进程组清理（process group / kill tree）**：v0 没 `setpgid`，`sh -c "cmd1 | cmd2"` 取消时只杀 `sh`，pipeline 子进程被 init 接管。codex `rmcp-client/src/stdio_server_launcher.rs` 用 `Command::process_group(0)` + 系统调用 `kill_process_group()` 已经趟通跨平台路径；opencode `shell.ts:28-56` 给出 SIGTERM→3s→SIGKILL 的升级模型。v1 整段抄即可。
- **优雅终止（SIGTERM grace period）**：v0 直接 SIGKILL（`kill_on_drop`），子进程没机会写 lockfile / 关连接 / flush。配合上一条做：取消触发后先发 SIGTERM、等 N 秒（默认 3s）、还没退就升级 SIGKILL。
- **OS 级隔离**：landlock / seatbelt / windows-sandbox。轨迹见 [`sandbox-policy.md` §8](./sandbox-policy.md#8-演进口子os-级-sandbox)。届时 `ToolContext` 多一个 `sandbox: &dyn ToolSandbox`，本工具用 `sandbox.wrap_command(cmd, allows)`，主体逻辑不变。
- **命令解析 / 白名单**：codex 走 [`execpolicy`](../coding-reference/codex/codex-rs/execpolicy/)（Starlark + 规则集 + [`shell-command/parse_command.rs`](../coding-reference/codex/codex-rs/shell-command/src/parse_command.rs) 的 2500 行 shell 词法分析器）。我们直接用而不是自己写。届时 §2 的 `safety_hint` 改成调 execpolicy 决定 `ReadOnly` / `Mutating` / `Destructive`，每条 bash 不再无脑 Destructive。
- **argv 模式作为更安全形态**：codex 用 `Vec<String>` 把 program 与 args 分开 spawn，从协议层规避 shell 注入。我们 v0 用 `command: String` + `sh -c` 是为 LLM 友好（一行写完 pipeline / redirect），代价是注入风险归 LLM 负责。v1+ 可以并存两个工具：`bash`（任意 shell 行）走 ask_writes、`exec`（argv 形态）走更宽松 policy。
- **spill-to-disk 大输出**：v0 超 1 MiB 直接 drop。opencode `shell.ts:435-596` 双缓冲：内存只留 tail、超量 spill 到 `/tmp` 临时文件、metadata 暴露文件路径让 LLM 后续用 `tail` / `grep` 取。v1 实现时要把临时文件生命周期挂到 session（session 关闭一起清）。
- **流式增量输出**：v0 一次性 `Completed`（见 §4.2）。要做"边跑边在客户端滚"必须解决 wire 形态——要么走 ACP `terminal/create` 反向请求拿 `TerminalId` 后用 `ToolCallContent::Terminal` 引用（客户端读 terminal）、要么等 ACP 给 `content` 加 append 语义。前者是 ACP 标准答案，配合 `terminal` 工具一起做。
- **持久 shell session**：codex 的 zsh-fork backend 让多次 `bash` 调用共享 PWD / env。我们 v0 每条命令都是新 `sh -c`。需要持久态时由 LLM 自己 `cd ... && cmd` 串成一行。
- **交互式命令**：v0 `stdin=null` 截断。要交互式（PTY、子进程问 y/n）时引入新的 `terminal` 工具，对位 ACP [`terminal/create`] 反向请求；不挤进 `bash`。
- **后台/异步执行**：`bash` 调用必须在 turn 内同步完成。需要"后台跑构建"时引入 `background_task` 工具，对位 ACP 的长跑机制。
- **结构化输出捕获**：v0 `raw_output` 仅 `{exit_code, timed_out, truncated_bytes}`。LLM 想拿到 stdout / stderr 分流 / timing 信息时，要么解析 content 文本，要么等 `exec` 工具引入。

[`terminal/create`]: https://agentclientprotocol.com/protocol/terminals

## 9. 落地节奏

1. `crates/tools/Cargo.toml` 加依赖：`defect-agent`（`Tool` trait）、`agent-client-protocol-schema`、`tokio` (`process` / `io-util` / `time`)、`serde` / `serde_json`、`thiserror`、`futures`。
2. 新建 `crates/tools/src/bash/mod.rs`：`pub struct BashTool` 实现 `Tool`；私有子模块 `output.rs`（buffer + truncate）、`spawn.rs`（跨平台 `Command::new`）、`workdir.rs`（§5.1 路径校验）。
3. 单元测试 `bash/tests.rs` 跑 §7 #1–#9。#5 / #4 用 `tokio::time::pause()` 加速。
4. e2e：在 `crates/cli/examples/deepseek_e2e.rs` 注册 `BashTool` 与 `EchoTool` 同列；新增 prompt `"List the files in the current directory"` 验证 §7 #10。
5. 在 [`docs/internal/llm-trait.md`] / [`turn-loop.md`] 不需要改动——`Tool` trait 已经把 bash 装进去。
6. 更新 `crates/tools/src/lib.rs` 顶层 `pub mod bash; pub use bash::BashTool;`。

[`docs/internal/llm-trait.md`]: ./llm-trait.md
[`turn-loop.md`]: ./turn-loop.md
