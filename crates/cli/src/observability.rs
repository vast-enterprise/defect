//! 把 `defect-config` 的 typed Langfuse 配置翻译成 `defect-obs` 的上报观察器。
//!
//! 翻译 + 校验放 CLI 装配期：`defect-obs` 不依赖 `defect-config`（保持单向依赖），
//! 而“enabled 但缺 key”这类策略校验天然属于装配层。
//!
//! 设计详见 `docs/internal/observability-langfuse.md` §6。

use std::time::Duration;

use defect_config::LangfuseConfig;
use defect_obs::langfuse::{
    DEFAULT_FLUSH_INTERVAL, DEFAULT_HOST, DEFAULT_MAX_BATCH, LangfuseSetup, build_observer,
};
use defect_obs::LangfuseObserver;

/// 按 typed Langfuse 配置构造上报观察器。
///
/// 返回 `Ok(None)` 表示未启用（配置缺失 / `enabled = false`）——此时调用方
/// 不挂观察器。`enabled = true` 但缺 `public_key` / `secret_key` 时**告警并
/// 禁用**（返回 `Ok(None)`），不报错、不静默成功，符合显式校验约定。
///
/// # Errors
///
/// 当 HTTP 栈构造失败（TLS roots / 代理 URL 解析）时返回错误。
pub fn build_langfuse_observer(
    config: Option<&LangfuseConfig>,
    http_stack_config: defect_http::HttpStackConfig,
) -> anyhow::Result<Option<LangfuseObserver>> {
    let Some(cfg) = config.filter(|c| c.enabled) else {
        return Ok(None);
    };

    let (Some(public_key), Some(secret_key)) = (cfg.public_key.clone(), cfg.secret_key.clone())
    else {
        tracing::warn!(
            "tracing.langfuse.enabled = true but public_key / secret_key missing; \
             langfuse reporting disabled"
        );
        return Ok(None);
    };

    let http = defect_http::build_http_stack(http_stack_config)
        .map_err(|e| anyhow::anyhow!("langfuse http stack init failed: {e}"))?;

    let setup = LangfuseSetup {
        host: cfg.host.clone().unwrap_or_else(|| DEFAULT_HOST.to_string()),
        public_key,
        secret_key,
        flush_interval: cfg
            .flush_interval_ms
            .map_or(DEFAULT_FLUSH_INTERVAL, Duration::from_millis),
        max_batch: cfg.max_batch.unwrap_or(DEFAULT_MAX_BATCH),
    };

    Ok(Some(build_observer(setup, http)))
}
