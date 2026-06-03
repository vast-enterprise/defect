//! 把 [`SandboxMode`] 翻成具体的 [`SandboxPolicy`] 实例。

use std::sync::Arc;

use defect_agent::policy::{
    AskWritesPolicy, DenyAllPolicy, ModeCatalog, OpenPolicy, PolicyMode, ReadOnlyPolicy,
    SandboxPolicy,
};
use defect_config::SandboxMode;

/// 按 `[sandbox].mode` 选择 policy 实现。
pub fn build_policy(mode: SandboxMode) -> Arc<dyn SandboxPolicy> {
    match mode {
        SandboxMode::ReadOnly => Arc::new(ReadOnlyPolicy),
        SandboxMode::AskWrites => Arc::new(AskWritesPolicy::new()),
        SandboxMode::Open => Arc::new(OpenPolicy),
        SandboxMode::DenyAll => Arc::new(DenyAllPolicy),
    }
}

/// 全部 sandbox 模式，按固定展示顺序（read-only → ask-writes → open →
/// deny-all）。`current` 标记当前选中项，映射到 ACP `SessionModeState`。
///
/// 暴露**全部** 4 个模式给客户端（`session/set_mode` 可在它们之间切换）。
/// mode id 用 [`SandboxMode::as_str`]——与配置文件里 `[sandbox].mode` 取值
/// 同一套字符串，单一真相源。
pub fn build_mode_catalog(current: SandboxMode) -> ModeCatalog {
    let modes = [
        (
            SandboxMode::ReadOnly,
            "Read-only",
            "只放行只读工具；写/执行/网络一律拒绝。",
        ),
        (
            SandboxMode::AskWrites,
            "Ask before writes",
            "只读直接放行；写/执行/网络逐次询问，可选择本次或永久允许。",
        ),
        (
            SandboxMode::Open,
            "Open",
            "一切放行，不询问。适合受信环境 / 全自动运行。",
        ),
        (
            SandboxMode::DenyAll,
            "Deny all",
            "一切拒绝。用于演练 / 只看不动。",
        ),
    ]
    .into_iter()
    .map(|(mode, name, desc)| PolicyMode {
        id: mode.as_str().to_string(),
        name: name.to_string(),
        description: Some(desc.to_string()),
        policy: build_policy(mode),
    })
    .collect::<Vec<_>>();

    // 不变量：`current` 必命中上面四条之一（SandboxMode 是封闭枚举），故
    // `ModeCatalog::new` 恒返回 `Some`——拿不到就是装配 bug，fail loud。
    ModeCatalog::new(modes, current.as_str())
        .expect("mode catalog must contain the current sandbox mode")
}
