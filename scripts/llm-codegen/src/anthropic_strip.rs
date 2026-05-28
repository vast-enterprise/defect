//! Anthropic codegen 后置 patch。
//!
//! toac 当前不识别 OAS `nullable: true`，把 `citations: Vec<TextCitation>`
//! 这种字段渲染成必字段——而 Anthropic 真实 wire（无论是官方 SSE 还是
//! Bedrock event-stream）在 `content_block_start` 阶段的 text block 经常
//! 不带 citations 字段，反序列化直接失败。
//!
//! 这里的 patch 只做一件事：给已知应当容忍 missing 的 `Vec<...>` 字段
//! 加 `#[serde(default)]`。OAS 里这些字段都标了 `nullable: true`，所以
//! 行为与 spec 一致——missing == empty。
//!
//! 做法：纯字符串替换。`old_string` 必须命中且唯一；命中失败说明 toac
//! 输出形态变了，patch 直接报错让上游同步时显式更新这条规则。

use anyhow::{Result, bail};

const QUIRKS: &[Quirk] = &[
    // TextBlock（响应版 text content block）。Anthropic SSE 在
    // content_block_start 那一帧只发 `{"type":"text","text":""}`，
    // 不会带 citations；OAS 标 nullable: true。
    Quirk {
        target: "TextBlock",
        before: "    pub struct TextBlock {\n        pub citations: Vec<TextCitation>,",
        after: "    pub struct TextBlock {\n        #[serde(default)]\n        pub citations: Vec<TextCitation>,",
    },
];

struct Quirk {
    /// 仅供调试：被 patch 的目标类型/字段简称，命中失败时打到 error 里。
    target: &'static str,
    before: &'static str,
    after: &'static str,
}

/// 跑全部 quirk patch；任何一条命中失败 → 整体报错。
pub fn patch_generated(body: &str) -> Result<String> {
    let mut out = body.to_string();
    for quirk in QUIRKS {
        let count = out.matches(quirk.before).count();
        if count == 0 {
            bail!(
                "anthropic_strip: quirk `{}` did not match any occurrence; \
                 toac output likely changed shape — update the patch",
                quirk.target,
            );
        }
        if count > 1 {
            bail!(
                "anthropic_strip: quirk `{}` matched {count} occurrences; \
                 narrow the `before` snippet so it matches exactly once",
                quirk.target,
            );
        }
        out = out.replace(quirk.before, quirk.after);
    }
    Ok(out)
}
