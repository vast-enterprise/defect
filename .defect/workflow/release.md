# 发版流程（Release Workflow）

> 本文是给 **AI agent 与人类维护者共同遵循** 的发版操作手册。每一步都给出可直接执行的命令、校验方式与失败处理。**严格按顺序执行**，不要跳步。
>
> 真相源：版本号在 `Cargo.toml`，CI 在 `.github/workflows/{ci,publish-crates,release}.yml`。本文若与这些文件不符，以文件为准，并回头更新本文。

---

## 0. 心智模型：三条独立的 CI 管线

发一个版本牵动三个 workflow，**触发方式各不相同**，别搞混：

| Workflow | 文件 | 触发方式 | 作用 |
|---|---|---|---|
| `ci` | `.github/workflows/ci.yml` | 自动：push 到 `main` / 任何 PR | 质量门：fmt / clippy / test / doc / provider feature 矩阵 |
| `release` | `.github/workflows/release.yml` | 自动：push `v*` tag（也可手动 dispatch 仅验证构建） | 跨平台构建 `defect` 二进制 + 建 GitHub Release |
| `publish-crates` | `.github/workflows/publish-crates.yml` | **仅手动** `workflow_dispatch`，默认 `dry_run=true` | 按依赖拓扑发布各 crate 到 crates.io |

关键含义：
- **建 GitHub Release 靠推 tag**，不是手动建 release。推 `vX.Y.Z` tag → `release.yml` 自动构建产物并发 release。
- **发 crates.io 是独立的手动动作**，和 tag/release 不联动，需要单独去 Actions 页面触发。
- **版本号 bump 是前提**，两条发布管线都假设版本号已经改好并合入 `main`。

---

## 1. 前置检查（动手前）

```bash
# 必须在 main、且与 origin/main 同步、工作区干净
git checkout main && git pull --ff-only
git status --porcelain          # 应为空
```

- 确认 `CARGO_REGISTRY_TOKEN` 已配在仓库 Secrets（仅首次发 crates.io 需要；没配则 publish 真发布会失败）。
- 确认本次要发的版本号（遵循 semver；当前是 `0.1.0-alpha.N` 这种 prerelease 形态）。

---

## 2. Bump 版本号 ⚠️ 必须改 12 处

版本号在 `Cargo.toml` 里**写了 12 遍**，全部要改成同一个新版本，否则 workspace 内部依赖解析不一致、`cargo publish` 会失败。

> ⚠️ 注意：`Cargo.toml` 里有句注释说「一处改全仓库生效」——**那句话不准确**。`workspace.package.version` 只管 crate 自身版本；`[workspace.dependencies]` 里每条 `defect-*` 依赖都**硬写了一遍版本号**（path+version 双写：本地走 path，发 crates.io 时 path 被剥离只留 version），这些必须手动同步。

要改的位置（以 `0.1.0-alpha.4 → 0.1.0-alpha.5` 为例）：

1. `Cargo.toml` 第 29 行附近：`[workspace.package]` 的 `version = "..."`（crate 自身版本，真相源）
2. `Cargo.toml` 的 `[workspace.dependencies]` 段：11 条 `defect-* = { path = "...", version = "..." }`（agent / http / llm / tools / mcp / sandbox / storage / config / acp / obs，共 11 行）

一条命令全改（**改完务必人工 review diff**）：

```bash
OLD="0.1.0-alpha.4"
NEW="0.1.0-alpha.5"
# 只动 Cargo.toml 顶层 workspace 清单，不碰其它文件
sed -i "s/${OLD}/${NEW}/g" Cargo.toml
git diff Cargo.toml          # 应恰好 12 处改动，确认无误
```

各子 crate 的 `Cargo.toml` 用 `version.workspace = true`，**不需要**单独改。

然后刷新 lockfile（`--locked` 的 CI 会校验它）：

```bash
cargo update -p defect-cli --precise "$NEW"   # 或直接 cargo build 让 lock 自然更新
# 确认 Cargo.lock 里 defect-* 都变成新版本
grep -A1 'name = "defect-' Cargo.lock | grep version | sort -u
```

---

## 3. 本地全量校验（对齐 CI，省得推上去才发现挂）

CI 的五个 job 都能本地先跑一遍。**全绿才继续**：

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items
# provider feature 矩阵（逐家单独编，对齐 ci.yml 的 feature-matrix job）
for p in provider-anthropic provider-bedrock provider-openai provider-deepseek; do
  cargo build -p defect-cli --no-default-features --features "$p,yaml,repl,oneshot" --locked || break
done
```

---

## 4. 提交并合入 main，等 CI 绿

```bash
git add Cargo.toml Cargo.lock
git commit -m "release: bump version to ${NEW}"
git push origin main           # 或开 PR 合入；二者都会触发 ci.yml
```

去 GitHub Actions 看 `ci` workflow 在该 commit 上**全绿**再往下走。tag 一旦推出去就难收回，务必等 CI 通过。

---

## 5. crates.io 发布（手动触发）

```bash
gh workflow run publish-crates.yml -f dry_run=false
gh run watch
```

关于这条管线的内置行为（来自 `publish-crates.yml`，无需手动干预）：
- **发布顺序**按内部依赖拓扑：`agent → sandbox → config → http → storage → tools → llm → mcp → obs → acp → cli`（被依赖者先发）。
- **幂等**：真发布时已在 crates.io 上的 `crate@version` 会被跳过，失败后重跑不会在已发的包上撞 "already exists"。
- **包间等待**分两种情况：
  - **升版本（crate 名已存在）**：瓶颈只是索引传播，不是限流（crates.io 升版本 burst 30 / 每分钟补 1，11 个包远在 burst 内）。每发完一个包**轮询其索引出现该版本即继续**（常态 ~10s，上限 90s 兜底），整套约 2 分钟。
  - **首发新 crate 名**：crates.io 对新名限流是 burst 5 + 每 ~10 分钟补 1，会直接拒绝过快的新 crate 发布——无法靠轮询绕过，发布前等 ~11 分钟窗口。**只有第一次整套发布会撞到（11 个新 crate），属一次性成本**，别中途取消。

> 首次发布额外确认：各 crate 名在 crates.io 未被占用、`CARGO_REGISTRY_TOKEN` 已配置。

### ⚠️ 关于 `dry_run=true`：对本 workspace **整套会失败，属预期**

直觉上该先跑 `dry_run=true` 验证。但 **dry-run 不真正上传**（每个包都 "aborting upload due to dry run"），
而 `cargo publish` 校验依赖时查的是**真实 crates.io 索引**。于是从第一个有内部依赖的
crate（`defect-config` 依赖 `defect-agent`）起，dry-run 必然报：

```
failed to select a version for the requirement `defect-agent = "^0.1.0-alpha.4"`
candidate versions found which didn't match: <crates.io 上的旧版本>
```

——因为前面的 `defect-agent@<新版本>` 在 dry-run 里没真传上去，索引里查不到。

**这不是 bug，是 workspace 链式依赖用 dry-run 的固有死结。** 结论：
- dry-run **只能验证叶子 crate**（`defect-agent` / `defect-sandbox`，无内部依赖）能 package；
- 链式依赖的 crate 只能靠**真发布**逐个上传、索引传播后才解析得到（workflow 里 70 秒 sleep 就是给索引传播留的）；
- 因此**不要把整套 dry-run 全绿当作真发布的前置门**。真发布反而不受此限：被依赖者先真传，后续 crate 轮到时索引已更新。

---

## 6. 推 tag → 自动建 GitHub Release

tag 必须是 `vX.Y.Z` 形态（`release.yml` 监听 `v*`），且 **`v` + 第 2 步的版本号**：

```bash
git tag "v${NEW}"
git push origin "v${NEW}"
```

推上去后 `release.yml` 自动：
1. 在 6 个 target 上用 `--profile dist` 构建 `defect` 二进制
   （linux gnu x86_64/arm64、linux musl x86_64、macOS x86_64/arm64、windows x86_64）；
2. 打包（unix `.tar.gz` / windows `.zip`），命名 `defect-v<ver>-<target>.<ext>`；
3. 汇总 `SHASUMS256.txt`；
4. 用 `softprops/action-gh-release` 建 Release、`generate_release_notes: true` 自动生成 changelog、上传所有产物。

```bash
gh run watch                   # 看 release workflow
gh release view "v${NEW}"      # 确认 release 已建、6 个平台产物 + SHASUMS256.txt 齐全
```

> 想在不发 release 的情况下验证跨平台构建：去 Actions 手动 `workflow_dispatch` 跑 `release`——只产 artifact、不建 release（`if: startsWith(github.ref, 'refs/tags/v')` 守住了这一点）。

---

## 7. 发布后

- 确认 crates.io 上新版本可见、`gh release view` 产物齐全。
- 如果这是个里程碑版本，按需更新 README / 文档里的版本引用。
- 进入下一开发周期再 bump 到下一个 prerelease（可选）。

---

## 常见失败与处置

| 现象 | 原因 | 处置 |
|---|---|---|
| `cargo publish` 报版本/依赖解析错 | 第 2 步漏改了某条 `[workspace.dependencies]` 版本 | 回到第 2 步，确认 12 处全改、`git diff Cargo.toml` 恰好 12 处 |
| CI `test` job 挂在 `--locked` | `Cargo.lock` 没跟着 bump 更新 | 本地 `cargo build` 刷新 lock 后重新提交 |
| publish 中途失败 | crates.io 限流 / 网络 | 直接重跑同一 workflow——幂等逻辑会跳过已发布的包 |
| release 没建出来 | 推的不是 `v*` tag，或在 main 上手动 dispatch（被 `if` 守住只产 artifact） | 用 `git push origin vX.Y.Z` 推规范 tag |
| tag 推错了版本 | tag 名与 Cargo 版本不一致 | 删除错误 tag（`git push origin :vX.Y.Z`）、改对后重推；release 会随新 tag 重建 |
