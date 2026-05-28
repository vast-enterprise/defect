# 多 Provider 设计参考报告：opencode 与 defect 的对照

## 1. 目的

本文整理 `opencode` 在多 provider 设计上的可借鉴点，并结合当前 `defect` 仓库的实现，给出一条适合本项目的演进路线。

目标不是照搬 `opencode`，而是回答一个更具体的问题：

- 当我们已经决定“已有大厂商值得单独拿出一个 provider”时，
- 又希望“自定义端点可以复用 openai 兼容 provider”时，
- 应该如何在 **provider 实现**、**provider profile**、**provider instance** 之间建立清晰边界。

## 2. 先说结论

`opencode` 值得借鉴的，不是它用了多少 provider，也不是它用了 plugin，而是它把“provider 是什么”从一个固定枚举，提升成了一种**可组合记录**。

对 `defect` 来说，最值得吸收的结论有三条：

1. **把 provider 实现 与 provider 配置实例分开。**
2. **把 endpoint/protocol 差异显式建模，而不是藏在 match 分支里。**
3. **让用户选择 instance，而不是直接选择 provider 枚举值。**

同时，`defect` 不应完全照搬 `opencode`：

1. `defect` 已经有明确的 `LlmProvider` trait 和“协议层 / 厂商层”两层架构，这部分是正确的。
2. `defect` 比 `opencode` 更强调 vendor-specific transport、auth、stream decode，这意味着“已有厂商独立 provider”这个方向应当保留。
3. 因此，`defect` 应该吸收的是 `opencode` 的**目录/实例化思路**，而不是放弃当前 trait 架构。

## 3. opencode 在做什么

### 3.1 provider 不是固定枚举分发，而是 catalog 里的记录

`opencode` 里最重要的不是单个 provider 文件，而是这一组结构：

- `packages/core/src/provider.ts`
- `packages/core/src/model.ts`
- `packages/core/src/catalog.ts`

其中 `ProviderV2.Info` 不是一个“执行器”，而是一条 **provider 元数据记录**，大致包含：

- `id`
- `name`
- `enabled`
- `env`
- `endpoint`
- `options`

`ModelV2.Info` 再引用 `providerID`，形成 “provider record + model record” 的 catalog。

这意味着 `opencode` 的核心抽象不是：

- `match provider_kind { ... }`

而是：

- 先有一份 provider/model catalog
- 再从 catalog 解析出某个 model 的有效 endpoint/options
- 最后交给具体底层 SDK 或 provider plugin 执行

这个方向的价值在于：**provider 的存在形式首先是数据，而不是代码分支。**

### 3.2 endpoint 被显式建模

`opencode/packages/core/src/provider.ts` 里专门定义了 `Endpoint` union：

- `openai/responses`
- `openai/completions`
- `anthropic/messages`
- `aisdk`
- `unknown`

这个建模很关键，因为它把几类本来容易混在一起的差异拆开了：

- 协议长什么样
- 请求打到哪里
- 额外 provider options 是什么

换句话说，`opencode` 没把“OpenRouter 是一个 provider”“OpenAI 是一个 provider”这种概念混成一个平面；它承认：

- 有些 provider 的真正差异是 endpoint 类型
- 有些 provider 的差异只是 options
- 有些 provider 的差异只是部署实例

这一点对 `defect` 很有启发。

### 3.3 provider options 与 model options 是可合成的

`opencode/packages/core/src/catalog.ts` 里的 `resolve()` 做了一件很值得注意的事：

- provider 级 options 先作为默认值
- model 级 options 再覆写进去
- endpoint 也允许 model 继承 provider 的 endpoint

这是一个非常典型的“目录配置合成”模型。

它的含义不是“模型一定要覆盖 provider”，而是：

- provider 表达共享默认值
- model 表达细粒度差异

这套机制在 `defect` 里可以平移成：

- provider profile 给静态兼容差异
- provider instance 给部署级默认值
- model 配置给模型级覆写

### 3.4 openai-compatible 在 opencode 里是一类共享能力，而不是一堆复制品

`opencode/packages/core/src/plugin/provider/openai-compatible.ts` 很薄，但它体现了一个方向：

- openai-compatible 是一种**可复用能力**
- 不是每接一家兼容厂商就复制一套完整 provider 栈

`defect` 当前的 `DeepSeekProvider` 已经部分体现了这个思路，但还停留在“包一层 wrapper”的阶段，没有把这套兼容机制提升成一个独立的、可实例化的抽象。

## 4. defect 当前现状

### 4.1 优点：运行时抽象已经是对的

当前 `defect` 的强项在于：

- [`docs/outbound/llm.md`](../outbound/llm.md) 已明确采用“协议层 + 厂商层”两层结构
- `defect-agent` 里有统一的 `LlmProvider` trait
- `defect-llm` 已经把 `Anthropic`、`OpenAI`、`DeepSeek` 分开实现

其中这条架构判断是成立的：

- Anthropic / Bedrock / Vertex 不能简单地被视为同一种 transport
- OpenAI / Azure / DeepSeek / Qwen 虽然更接近，但也不应被压成一个不可维护的大文件

也就是说，`defect` 现在的问题**不是抽象太多**，而是抽象的层次还缺了一层“实例化目录”。

### 4.2 缺点：provider 仍然被当作固定枚举

当前几个关键位置仍然是硬编码 provider：

- [`crates/config/src/types.rs`](../../crates/config/src/types.rs) 里的 `ProviderKind`
- [`crates/config/src/loader.rs`](../../crates/config/src/loader.rs) 里按 `ProviderKind` 决定默认模型与 provider 配置段
- [`crates/cli/src/main.rs`](../../crates/cli/src/main.rs) 里按 `match` 构造 provider

这会带来几个问题：

1. 每新增一家 provider，都要改 config schema、CLI 枚举、装配逻辑。
2. 用户无法定义多个同类 provider 实例。
3. “官方 OpenAI” 与 “企业内网 vLLM” 这种同协议不同实例，无法优雅共存。
4. `DeepSeekProvider` 这种兼容厂商，即使只差很少，也必须拥有完整的一套命名空间。

### 4.3 DeepSeek 已经暴露出“profile/instance 缺位”的问题

[`crates/llm/src/provider/deepseek.rs`](../../crates/llm/src/provider/deepseek.rs) 现在实际做的是：

- 复用 `OpenAiProvider`
- 覆写默认 capabilities
- 覆写 `/models` 解析
- 覆写 stream usage 解析

这本质上已经不是“完全独立的 provider”，而是：

- 一份 OpenAI-compatible 实现
- 套上一个 DeepSeek profile

只是当前代码里，这个 profile 还没有被显式命名出来。

## 5. defect 应该借鉴 opencode 的哪些点

## 5.1 引入第三层：instance

当前 `defect` 只有两层概念：

- provider 实现
- provider 配置

建议补上第三层概念：

- **provider implementation**
- **provider profile**
- **provider instance**

三者职责建议如下。

### provider implementation

表示一份 Rust 代码实现，负责：

- transport
- auth
- endpoint 调用
- stream decode
- `LlmProvider` trait 落地

例子：

- `AnthropicProvider`
- `OpenAiProvider`
- `DeepSeekProvider`
- 未来可能有 `OpenAiCompatProvider`

### provider profile

表示“同一实现下的一组静态兼容差异”。

适合放进 profile 的内容：

- vendor 标识与 display name
- 默认能力矩阵
- 默认 API key env 名
- `/models` 响应解析策略
- stream usage/thinking 解析策略
- 默认 headers/query patch
- 某些模型能力 hardcoded merge 策略

不适合放进 profile 的内容：

- 某个具体部署的 base URL
- 某个用户自己的 token
- 某个项目自己的默认模型

### provider instance

表示用户实际选择的命名配置项。

适合放进 instance 的内容：

- 绑定哪个 implementation
- 绑定哪个 profile
- `base_url`
- `api_key_env`
- `default_model`
- `allowed_models`
- `organization` / `project`
- 超时、代理、额外 header

这正是 `opencode` 的 provider/model catalog 思路里，最值得平移到 `defect` 的部分。

## 5.2 选择入口从 provider 改成 instance

`opencode` 的一个重要优点是：运行期真正使用的是某条 provider/model 记录，而不是“写死枚举后到处 match”。

`defect` 也应该改成：

- CLI 选择 `instance`
- 配置文件里默认值也是 `instance`
- 运行时通过 `instance` 找到 implementation + profile + overrides

而不是：

- CLI 选择 `provider=openai`
- 然后再去隐式拼各种别的配置

理由很简单：

- 一个用户完全可能同时有 `openai-official`
- 也有 `corp-vllm`
- 也有 `openrouter-prod`
- 它们都属于 OpenAI-compatible，但显然不是一个实例

## 5.3 endpoint/protocol 差异应继续显式建模

`opencode` 用 `Endpoint` union 把几类协议差异表达出来，这是好做法。

`defect` 当前已经有：

- `ProtocolId::OpenAiChat`
- `ProtocolId::AnthropicMessages`

建议继续沿这条线，而不是回退成隐式字符串判断。

更具体地说，`defect` 后续可以把 profile/instance 显式关联到一个 endpoint family：

- `anthropic_messages`
- `openai_chat`
- 未来若有 `openai_responses` 也可单列

这样 provider 目录、配置目录、文档目录会更一致。

## 6. defect 不该照搬 opencode 的哪些点

## 6.1 不建议把所有 provider 都降级成纯数据记录

`opencode` 的 catalog 模式很灵活，但 `defect` 的目标和约束不同：

- `defect` 需要更强的 Rust 类型约束
- `defect` 已经有 `LlmProvider` trait
- `defect` 对错误分类、stream 事件、能力矩阵都有更严格的语义要求

因此不建议把 `defect` 改造成：

- provider 只是 config 记录
- 真正执行全都走一个超级大兼容层

这样会削弱当前架构已经拥有的几个优势：

- vendor-specific tracing
- vendor-specific transport/auth
- vendor-specific error mapping
- vendor-specific stream state machine

## 6.2 不建议为了动态化而牺牲代码边界

`opencode` 通过 plugin 机制把 provider 扩展得很动态，这很适合它的运行时生态。

但对 `defect` 来说，目前没有必要为了“可动态加载 provider”而引入过重的插件式系统。当前更现实的问题是：

- 让内置 provider 的实例化能力更强
- 让 OpenAI-compatible 家族不再复制样板

所以 `defect` 需要的是：

- **静态实现 + 动态实例配置**

而不是：

- **动态 provider 代码加载**

## 7. 推荐的 defect 目标形态

## 7.1 概念模型

推荐把 `defect` 的多 provider 设计明确成下面三层：

1. `provider implementation`
2. `provider profile`
3. `provider instance`

其中关系如下：

- 一个 implementation 可以有多个 profile
- 一个 profile 可以被多个 instance 复用
- 用户实际选择的是 instance

## 7.2 配置形态

建议目标配置接近这样：

```toml
[default]
instance = "corp-vllm"
model = "qwen-plus"

[instances.corp-vllm]
provider = "openai_compat"
profile = "qwen"
base_url = "https://llm.internal.example.com/v1"
api_key_env = "CORP_LLM_API_KEY"
default_model = "qwen-plus"

[instances.openrouter-prod]
provider = "openai_compat"
profile = "openrouter"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
default_model = "openai/gpt-4o-mini"

[instances.openai-official]
provider = "openai"
default_model = "gpt-4o-mini"

[instances.deepseek-official]
provider = "deepseek"
default_model = "deepseek-chat"
```

在这个模型下：

- `openai`、`deepseek` 仍可保留独立 provider
- 新增一个企业自建兼容端点时，不需要新增 Rust provider
- 只要加一个 `openai_compat` instance 即可

## 7.3 Rust 侧装配形态

建议引入一层工厂，而不是继续在 CLI 里直接 `match`：

- `ProviderInstanceConfig`
- `ProviderFactory`
- `ProviderRegistry`

职责建议如下：

### ProviderRegistry

维护“名字 -> implementation builder”的映射，例如：

- `anthropic`
- `openai`
- `deepseek`
- `openai_compat`

### ProviderFactory

输入一条 `ProviderInstanceConfig`，输出：

- `Arc<dyn LlmProvider>`
- 以及与该 instance 绑定的 turn/model 默认配置

### ProviderInstanceConfig

保存某个实例的运行时参数：

- implementation 名
- profile 名
- endpoint/base_url
- auth env
- 默认模型
- 兼容能力覆写

## 8. 对 OpenAI-compatible 家族的具体建议

## 8.1 保留“已验证厂商的独立 provider”

用户提出“已有厂商值得单独拿出来一个 provider”，这个判断是合理的。

推荐保留：

- `AnthropicProvider`
- `OpenAiProvider`
- `DeepSeekProvider`

原因：

1. 它们已经存在。
2. 它们的错误语义、smoke、真实接口经验都在持续积累。
3. 对用户与文档来说，独立 provider 名字本身就是稳定 API。

## 8.2 新增一个 `OpenAiCompatProvider`

新增的不是替换现有实现，而是补一个更通用的“复用通道”：

- `OpenAiCompatProvider`

这份 provider 可以承接：

- 企业代理
- vLLM
- LM Studio
- OpenRouter
- Together 类兼容接口
- 其他“兼容到足够程度”的 OpenAI Chat Completions 端点

它和 `OpenAiProvider` 的关系不应是二选一，而应是：

- `OpenAiProvider`：官方 OpenAI 语义优先，保持强约束
- `OpenAiCompatProvider`：兼容端点优先，允许 profile patch

## 8.3 DeepSeek 可以逐步从“独立逻辑”演进为“独立品牌 + 共享 compat 内核”

对 `DeepSeekProvider` 来说，一个平衡的演进方式是：

- 对外名字继续叫 `deepseek`
- 对内实现尽量复用 `openai_compat` 内核

也就是：

- `deepseek` 仍是独立 provider brand
- 但其内部可由 `openai_compat + deepseek profile` 支撑

这比“完全删掉 `DeepSeekProvider`”更稳，也更符合当前项目的文档语义。

## 9. 推荐迁移路径

## 9.1 第一阶段：只引入 instance，不改 LLM 协议层

先做最有收益、侵入最小的部分：

1. 配置新增 `instances`
2. CLI 新增 `--instance`
3. 默认选择逻辑从 `provider` 改成 `instance`
4. 旧 `provider` 配置继续兼容

这一阶段不需要改：

- `LlmProvider` trait
- `protocol/`
- `ProviderChunk`

## 9.2 第二阶段：把 provider 构造从 CLI `match` 挪到工厂

把 [`crates/cli/src/main.rs`](../../crates/cli/src/main.rs) 里的 provider 直接装配收口到专门工厂中。

收益：

- CLI 不再关心每家 provider 的构造细节
- 后续新增 instance/profile 时不会把 CLI 文件继续做大

## 9.3 第三阶段：引入 `openai_compat`

这一阶段再做真正的 polyfill 抽象：

- 提炼 profile 结构
- 把 `/models` 解析、stream decode patch、capabilities override 收口成策略对象
- 用它支撑新接入的兼容端点

此时是否把 `DeepSeekProvider` 内核改到 compat 之上，可以单独评估，不必在第一步硬做。

## 9.4 第四阶段：按需新增 profile，而不是新增 provider

当框架成型后，新增多数兼容端点时应优先问：

- 这是新 provider 吗？
- 还是 `openai_compat` 下的新 profile / 新 instance？

推荐判断标准：

- 若 transport、auth、stream 语义明显不同，建独立 provider
- 若只是 OpenAI-compatible 的轻量差异，建 profile 或 instance

## 10. 最终建议

结合 `opencode` 与当前 `defect`，最终建议如下：

1. **保留当前“协议层 + 厂商层”的 Rust 架构。**
2. **不要再把 provider 选择绑定在固定枚举上。**
3. **引入 instance 作为用户真正选择的对象。**
4. **为 OpenAI-compatible 家族补一层 profile/instance 抽象。**
5. **保留 OpenAI、DeepSeek、Anthropic 等成熟厂商的独立 provider 品牌。**

如果把这件事压缩成一句话，就是：

> `defect` 不需要变成 `opencode`，但需要学会像 `opencode` 一样，把 provider 从“代码分支”提升成“可组合的目录对象”；同时保留自己已经做对的 trait 与 vendor-specific 实现边界。

## 11. 附：本报告参考位置

本报告主要参考了以下仓库内镜像资料：

- `docs/coding-reference/opencode/packages/core/src/provider.ts`
- `docs/coding-reference/opencode/packages/core/src/model.ts`
- `docs/coding-reference/opencode/packages/core/src/catalog.ts`
- `docs/coding-reference/opencode/packages/core/src/plugin/provider/openai-compatible.ts`
- `docs/outbound/llm.md`
- `crates/config/src/types.rs`
- `crates/config/src/loader.rs`
- `crates/cli/src/main.rs`
- `crates/llm/src/provider/deepseek.rs`
