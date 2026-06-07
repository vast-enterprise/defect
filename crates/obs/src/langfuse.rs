//! Langfuse 接入：把 `AgentEvent` 流上报为 Langfuse trace / generation / span。
//!
//! Langfuse integration — LLM tracing and usage analytics.
//!
//! 模块划分：
//! - [`model`]：ingestion API 的 wire 结构体。
//! - [`ingest`]：批量缓冲 + 后台上报（可丢弃降级）。
//! - [`projector`]：`AgentEvent` → ingestion 事件的翻译。
//! - [`observer`]：`SessionObserver` 实现，每 session 订阅事件流。

use std::time::Duration;

use defect_http::HttpStack;

pub mod ingest;
pub mod model;
pub mod observer;
pub mod projector;

pub use ingest::{IngestConfig, LangfuseIngest};
pub use observer::LangfuseObserver;
pub use projector::TraceProjector;

/// 缺省 Langfuse host。
pub const DEFAULT_HOST: &str = "https://cloud.langfuse.com";
/// 缺省周期冲刷间隔。
pub const DEFAULT_FLUSH_INTERVAL: Duration = Duration::from_secs(2);
/// 缺省单批最大事件数。
pub const DEFAULT_MAX_BATCH: usize = 100;
/// 入队 channel 容量（背压边界；满了丢弃，不反压主循环）。
pub const DEFAULT_QUEUE_CAPACITY: usize = 1024;

/// 解析好的 Langfuse 上报参数（凭据已校验非空）。
pub struct LangfuseSetup {
    pub host: String,
    pub public_key: String,
    pub secret_key: String,
    pub flush_interval: Duration,
    pub max_batch: usize,
}

/// 用一份 [`LangfuseSetup`] + 已建好的 [`HttpStack`] 启动上报器，返回观察器。
///
/// 上报器的后台 flush 任务在此启动；返回的 [`LangfuseObserver`] 交给
/// `AgentCore` 的 `observe_session`。
#[must_use]
pub fn build_observer(setup: LangfuseSetup, http: HttpStack) -> LangfuseObserver {
    let ingest = LangfuseIngest::spawn(IngestConfig {
        http,
        host: setup.host,
        public_key: setup.public_key,
        secret_key: setup.secret_key,
        max_batch: setup.max_batch,
        flush_interval: setup.flush_interval,
        queue_capacity: DEFAULT_QUEUE_CAPACITY,
    });
    LangfuseObserver::new(ingest)
}

// 测试里大量 `value["body"]["x"]` 链式索引 + `.unwrap()` 断言——比
// `.get().expect()` 可读得多，且 panic 即测试失败正是想要的，故在测试模块
// 豁免 indexing_slicing / unwrap_used。
#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::unwrap_used)]
mod tests;
