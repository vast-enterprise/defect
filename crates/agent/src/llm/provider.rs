//! [`LlmProvider`] trait — the main LLM vendor integration signature.

use std::pin::Pin;

use futures::{Stream, future::BoxFuture};
use tokio_util::sync::CancellationToken;

use super::capability::{Capabilities, HostedCapabilities};
use super::chunk::ProviderChunk;
use super::error::ProviderError;
use super::model::{ModelInfo, ProviderInfo};
use super::request::CompletionRequest;

/// provider 流式生成的事件流。类型擦除以便 `dyn LlmProvider` 直接可用。
pub type ProviderStream = Pin<Box<dyn Stream<Item = Result<ProviderChunk, ProviderError>> + Send>>;

/// LLM provider 抽象。
///
/// 取消语义：[`LlmProvider::complete`] 接收一个 [`CancellationToken`]，
/// 调用方可在任意时刻 `cancel()` 终止此次调用与下游流。drop 返回的
/// stream 同样视为取消。
pub trait LlmProvider: Send + Sync {
    /// 厂商元信息（厂商名、API 风格、tracing 标签等）。
    fn info(&self) -> ProviderInfo;

    /// 厂商级能力矩阵。模型级差异通过
    /// [`super::ModelCapabilityOverrides`] 表达，主循环按需合并。
    fn capabilities(&self) -> Capabilities;

    /// provider adapter 自报家门的 hosted capability 集合。
    ///
    /// 与 [`Self::capabilities`] 不同——前者是模型属性，这里是当前
    /// adapter 实现状态：能否把 hosted web_search / fetch 等通过 wire 暴露
    /// 给模型。session 启动期会读这个值与
    /// `capabilities.web_search.mode` 一起做 hosted web search 的能力来源裁决。
    ///
    /// 默认实现返回全 `false`，新 provider 不需要主动覆盖。
    /// 真支持 hosted 的 adapter（Anthropic / OpenAI Responses）应
    /// 显式 override 此方法。
    fn hosted_capabilities(&self) -> HostedCapabilities {
        HostedCapabilities::default()
    }

    /// 列出此 provider 当前可用的模型。
    ///
    /// 实现可能产生网络调用（如 OpenAI `/v1/models`），结果应在
    /// provider 内部缓存以供 [`Self::model_info`] 同步查询。
    ///
    /// # Errors
    ///
    /// 网络错误、鉴权错误、服务端错误等均映射为 [`ProviderError`]。
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>>;

    /// 同步查询某个模型的元信息。
    ///
    /// 用于主循环裁剪 context 时的快路径，**不应触发网络调用**。
    /// 若 provider 缓存里没有，返回 `None`；调用方可决定是先调
    /// [`Self::list_models`] 再重试，还是按未知模型处理。
    fn model_info(&self, model_id: &str) -> Option<ModelInfo>;

    /// 启动一次流式生成。
    ///
    /// # Errors
    ///
    /// 鉴权失败、参数非法、传输错误、服务端错误等均映射为
    /// [`ProviderError`]。流中产生的错误通过流上的 `Err` 项传递，
    /// 不通过此返回值。
    fn complete(
        &self,
        req: CompletionRequest,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>>;
}
