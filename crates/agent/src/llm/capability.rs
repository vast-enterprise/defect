//! provider 与模型的能力矩阵。
//!
//! 设计详见 `docs/internal/llm-trait.md` 第 5 节。

use serde::{Deserialize, Serialize};

/// provider 级能力矩阵。
///
/// 模型级差异由 [`ModelCapabilityOverrides`] 表达；主循环按需合并：
/// 模型级 `Some(_)` 覆盖 provider 级，`None` 沿用 provider 级。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    /// 工具调用（content_block 含 tool_use / tool_calls 字段）。
    pub tool_calls: FeatureSupport,
    /// 同一轮内并发多个 tool_use。
    pub parallel_tool_calls: FeatureSupport,
    /// 思考链。
    pub thinking: FeatureSupport,
    /// 多模态输入（图片）。
    pub vision: FeatureSupport,
    /// prompt cache。
    pub prompt_cache: FeatureSupport,
    /// thinking 内容回放策略。详见 [`ThinkingEcho`]。
    pub thinking_echo: ThinkingEcho,
}

/// 模型级覆写。`None` 表示沿用 provider 级 [`Capabilities`] 字段。
///
/// 字段集合按"现实中真的会按模型变化"的属性限定，不与 [`Capabilities`]
/// 机械一一对应。后续如出现新差异点再加。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCapabilityOverrides {
    pub thinking: Option<FeatureSupport>,
    pub vision: Option<FeatureSupport>,
    pub prompt_cache: Option<FeatureSupport>,
    pub parallel_tool_calls: Option<FeatureSupport>,
    pub thinking_echo: Option<ThinkingEcho>,
}

/// thinking 内容回放策略。
///
/// `Required` —— 上一轮 assistant 的 thinking 必须出现在下一轮请求里
/// （Anthropic extended thinking、DeepSeek-v4-pro）。`Forbidden` ——
/// 回放会被服务端拒（DeepSeek-R1、OpenAI o1 / o3 官方）。`Optional`
/// —— 服务端容忍两种行为。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingEcho {
    #[default]
    Forbidden,
    Required,
    Optional,
}

/// 三态特性支持声明。
///
/// 选用三态而非 `bool` 是为了表达 [`FeatureSupport::PassthroughAsTool`]
/// ——通过适配伪支持。即便 v0 没有产生此值的实现，从一开始定下三态
/// 也比未来从 `bool` 升枚举省事。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureSupport {
    Supported,
    Unsupported,
    /// 通过适配伪支持。
    ///
    /// 例如某 provider 没原生 `web_search`，但 agent 把它包装成一个
    /// 工具暴露给 LLM，借此"假装"支持。
    PassthroughAsTool,
}

/// provider 自报家门的 hosted capability 集合。
///
/// 与 [`Capabilities`] 区分：
/// - [`Capabilities`] 描述模型能力（thinking / vision / tool_calls 等）
/// - [`HostedCapabilities`] 描述 provider adapter 自身实现状态：当前
///   adapter 能不能在 wire 上声明 hosted search / fetch / code_execution
///
/// session 启动期通过 [`super::LlmProvider::hosted_capabilities`] 拿到
/// 此结构，与 `capabilities.search.mode` 一起决定本 session 的 search
/// 能力来源。
///
/// 设计详见 `docs/proposals/search-capability-and-fetch-tool.md` §10.1。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedCapabilities {
    /// provider adapter 是否支持 hosted web search。
    ///
    /// 当前 hosted tool 版本由 adapter 内部硬编码取最新（Anthropic
    /// `web_search_20260209`、OpenAI Responses API `web_search`）；
    /// agent 不感知具体版本字段。
    pub search: bool,
}

impl HostedCapabilities {
    /// 用单个字段构造。跨 crate 的测试或 adapter 实现需要这个入口，
    /// 因为本结构体 `#[non_exhaustive]` 后不能直接 struct literal。
    #[must_use]
    pub const fn with_search(search: bool) -> Self {
        Self { search }
    }
}
