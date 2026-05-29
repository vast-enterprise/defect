# defect 仓库常用操作
# 用法：`just <recipe>`，无参时打印 recipe 列表
default:
    @just --list

# 将 defect-cli 安装为 `defect` 二进制（dist profile：体积最小，编译慢）
install:
    cargo install --path crates/cli --bin defect --profile dist --locked

# 强制覆盖已有的 defect 安装
install-force:
    cargo install --path crates/cli --bin defect --profile dist --locked --force

# 卸载 `defect` 二进制
uninstall:
    cargo uninstall defect-cli

# 工作流程：clippy → fmt（参考 CLAUDE.MD 的修改后必做步骤）
check:
    cargo clippy --workspace --all-targets -- -D warnings
    cargo fmt --all -- --check

# 自动修复格式
fmt:
    cargo fmt --all

# 跑 workspace 全量测试
test:
    cargo test --workspace

# 调试构建
build:
    cargo build --workspace

# 优化构建（编译快、体积适中，日常用）
build-release:
    cargo build --workspace --release

# 发布构建（dist profile：fat LTO + 单 CGU，体积最小、编译慢）
build-dist:
    cargo build --bin defect --profile dist
