//! OpenAI 上游 OAS 裁剪 + patch。
//!
//! 上游 spec 全量约 81k 行（assistants / audio / batches / files /
//! fine_tuning / images / responses / vector_stores / 各类 organization
//! 端点），用不到。这条命令只 keep `/v1/chat/completions` POST + `/v1/models`
//! GET 两条端点 + 它们 `$ref` 闭包覆盖到的全部 schema，并对上游已知问题
//! 做最小 patch。
//!
//! 详见 `docs/outbound/llm-openai.md` §1.1 / §1.2。
//!
//! 用法：
//! ```bash
//! cargo run -p defect-llm-codegen -- openai-strip \
//!     --upstream ../tower-openapi-client/examples/openai-openapi/openapi.yaml
//! ```
//!
//! 不每次跑——只在上游同步时人工触发。生成 `crates/llm/oas/openai.yaml`，
//! 之后再跑 `cargo run -p defect-llm-codegen -- openai` 出 wire。

use std::{
    collections::{BTreeSet, VecDeque},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde_yaml::{Mapping, Value};

use crate::workspace_root;

/// 仅保留这两条端点 + 各自 `$ref` 闭包。新增 LLM 接口前先在这里挂。
const KEEP_PATHS: &[(&str, &[&str])] = &[("/chat/completions", &["post"]), ("/models", &["get"])];

/// 上游 spec 把 seed 字段写成 `9223372036854776000`，超过 `i64::MAX` 193，
/// `oas3` parser 拒收。下行就把字面量替换成 `i64::MAX`。
const UPSTREAM_OUT_OF_RANGE_LIT: &str = "9223372036854776000";
const I64_MAX_LIT: &str = "9223372036854775807";

/// CLI 入口：解析 `--upstream <path>` flag，跑裁剪 + patch + 写文件。
///
/// `args` 是 `--openai-strip` 之后的剩余参数。返回 `Ok(())` 表示完成。
pub fn run(args: &[String]) -> Result<()> {
    let mut upstream: Option<PathBuf> = None;
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--upstream" => {
                let p = iter.next().context("--upstream requires a path argument")?;
                upstream = Some(PathBuf::from(p));
            }
            other => bail!("unknown argument {other:?} for openai-strip"),
        }
    }
    let upstream = upstream.context("missing --upstream <path-to-openai-openapi/openapi.yaml>")?;

    let workspace_root = workspace_root()?;
    let raw = fs::read_to_string(&upstream)
        .with_context(|| format!("read upstream spec {}", upstream.display()))?;

    let stripped =
        strip_and_patch(&raw).with_context(|| format!("strip + patch {}", upstream.display()))?;

    let out = workspace_root.join("crates/llm/oas/openai.yaml");
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create_dir_all {}", parent.display()))?;
    }
    fs::write(&out, stripped).with_context(|| format!("write {}", out.display()))?;
    println!("[strip] {} → {}", upstream.display(), out.display(),);
    Ok(())
}

/// 主 pipeline：text-level patches → YAML parse → keep 端点闭包 →
/// 重写 paths / components → 序列化回 YAML。返回新 spec 文本。
pub fn strip_and_patch(raw: &str) -> Result<String> {
    // 1. 文本级 patches，必须在 YAML parse 之前——`9223372036854776000`
    //    parse 成 number 后超过 i64 范围，serde_yaml 自己也会拒。
    let raw = raw.replace(UPSTREAM_OUT_OF_RANGE_LIT, I64_MAX_LIT);
    // OAS 3.1 / JSON Schema 2020-12 期望 `exclusiveMinimum` /
    // `exclusiveMaximum` 是数字而非 bool；上游混着写。bool 形式 codegen
    // 用不到，行级删除即可。
    let raw = strip_lines(&raw, |line| {
        let t = line.trim();
        t == "exclusiveMinimum: true" || t == "exclusiveMaximum: true"
    });

    let mut spec: Value = serde_yaml::from_str(&raw).context("parse upstream YAML")?;
    let map = spec.as_mapping_mut().context("spec is not a mapping")?;

    // 2. 裁 paths 到我们要的 KEEP_PATHS 子集。
    let kept_paths = filter_paths(map)?;

    // 3. 计算 schemas / parameters / responses / requestBodies /
    //    securitySchemes 闭包：从 KEEP_PATHS 出发扫所有 `$ref`。
    let closure = compute_ref_closure(map, &kept_paths)?;

    // 4. 用闭包重写 components.schemas / parameters / 等等。
    filter_components(map, &closure)?;

    // 5. 顶层只留必要字段：openapi / info / servers / security / tags(裁) /
    //    paths / components。
    prune_top_level(map);

    // 6. 注入错误 schema + 4xx/5xx response 分支。上游 spec 这两条端点
    //    只声明 200，4xx/5xx 全靠 toac::DecodeError::UnexpectedStatus
    //    冒上来——provider 拿不到错误体。我们这里额外挂上 OpenAiErrorResponse
    //    （结构跟 OpenAI 实际错误返回一致，文档 §6 的 error.message /
    //    error.type / error.code / error.param 全保留）。
    inject_error_schema(map)?;
    inject_error_responses(map)?;

    // 7. 把流式 `finish_reason` 从 required 移除。OAS 把它列为 required
    //    但实际 wire：所有非终结 chunk 的 `finish_reason` 都是 null。
    //    codegen 投成 non-Option 的 enum 后，每条非终结 chunk 都会反序失败。
    //    去 required 之后 codegen 会投成 `Option<…FinishReason>`，符合实际。
    relax_stream_finish_reason(map)?;

    // 8. 删除若干已知会触发 toac codegen bug 的 `discriminator` 节点：
    //    toac 对 `oneOf + discriminator` 响应不一致——content-part 类投成
    //    `#[serde(untagged)]`（OK），而消息壳/数组 item 类投成
    //    `#[serde(tag = "<field>")]`，结果序列化时把 Rust variant **类型名**
    //    （如 `ChatCompletionRequestUserMessage`）注入 `<field>` 字段，
    //    覆盖每个 variant struct 内部自带的真实 `<field>` 值，上游一律拒收。
    //    删 discriminator 后 codegen 落到 untagged 路径，每个 variant 自带
    //    的 `role` / `type` 字段保留下来，形态符合上游期望。
    //
    //    位置一：顶层 schema `ChatCompletionRequestMessage`
    //    位置二：`ChatCompletionMessageToolCalls.items` 这个 inline oneOf
    drop_discriminator_at(
        map,
        &[&["components", "schemas", "ChatCompletionRequestMessage"]],
    )?;
    drop_discriminator_at(
        map,
        &[&[
            "components",
            "schemas",
            "ChatCompletionMessageToolCalls",
            "items",
        ]],
    )?;

    // 9. 给 `ChatCompletionRequestAssistantMessage` 挂一个非标 `reasoning_content`
    //    字段。OpenAI 官方 wire schema 没有这个字段，但 DeepSeek-v4-pro 等
    //    兼容厂商在 thinking 模式下要求把上一轮 reasoning 文本回放回去，
    //    否则 400 "reasoning_content must be passed back to the API"。我们
    //    把它作为可选 nullable string 注入 schema，protocol 层根据
    //    `Capabilities.thinking_echo` 决定是否填值。OpenAI 官方收到额外
    //    字段会忽略，不会破坏现有路径。
    inject_assistant_reasoning_content(map)?;

    // 10. 序列化回 YAML。
    let out = serde_yaml::to_string(&spec).context("serialize stripped spec")?;
    Ok(out)
}

/// 在 `components.schemas` 下挂一个 `OpenAiErrorResponse`。结构对应 OpenAI
/// 实际返回（platform 文档 + 实测）：
///
/// ```json
/// { "error": { "message": "...", "type": "...", "code": "...", "param": "..." } }
/// ```
fn inject_error_schema(spec: &mut Mapping) -> Result<()> {
    let components = spec
        .get_mut(Value::String("components".into()))
        .context("spec.components missing")?
        .as_mapping_mut()
        .context("spec.components is not a mapping")?;
    let schemas = components
        .entry(Value::String("schemas".into()))
        .or_insert_with(|| Value::Mapping(Mapping::new()))
        .as_mapping_mut()
        .context("components.schemas is not a mapping")?;

    // 先建 `error` 子对象。
    let mut error_props = Mapping::new();
    error_props.insert(
        Value::String("message".into()),
        yaml_string_schema("人类可读的错误描述。"),
    );
    error_props.insert(
        Value::String("type".into()),
        yaml_string_schema(
            "OpenAI 错误大类：invalid_request_error / authentication_error / \
             permission_error / not_found_error / rate_limit_error / \
             insufficient_quota / server_error / overloaded 等。",
        ),
    );
    error_props.insert(
        Value::String("code".into()),
        yaml_nullable_string_schema(
            "细化错误代码（如 model_not_found / context_length_exceeded / \
             insufficient_quota）。",
        ),
    );
    error_props.insert(
        Value::String("param".into()),
        yaml_nullable_string_schema("引发错误的请求参数名（如有）。"),
    );

    let mut error_obj = Mapping::new();
    error_obj.insert(Value::String("type".into()), Value::String("object".into()));
    error_obj.insert(
        Value::String("required".into()),
        Value::Sequence(vec![
            Value::String("message".into()),
            Value::String("type".into()),
        ]),
    );
    error_obj.insert(
        Value::String("properties".into()),
        Value::Mapping(error_props),
    );

    let mut top_props = Mapping::new();
    top_props.insert(Value::String("error".into()), Value::Mapping(error_obj));

    let mut top = Mapping::new();
    top.insert(Value::String("type".into()), Value::String("object".into()));
    top.insert(
        Value::String("required".into()),
        Value::Sequence(vec![Value::String("error".into())]),
    );
    top.insert(
        Value::String("properties".into()),
        Value::Mapping(top_props),
    );
    top.insert(
        Value::String("description".into()),
        Value::String(
            "OpenAI 4xx/5xx 错误响应。`error.code` / `param` 在不同错误类型上可能缺省。".into(),
        ),
    );

    schemas.insert(
        Value::String("OpenAiErrorResponse".into()),
        Value::Mapping(top),
    );
    Ok(())
}

/// 把 4xx / 5xx response 分支挂到所有保留 paths 的所有方法上。每条都
/// 引用 `OpenAiErrorResponse`。covering: 400 / 401 / 403 / 404 / 413 /
/// 429 / 500 / 502 / 503——足够覆盖文档 §6 列出的所有 OpenAI 实际返回。
fn inject_error_responses(spec: &mut Mapping) -> Result<()> {
    const STATUSES: &[&str] = &[
        "400", "401", "403", "404", "413", "429", "500", "502", "503",
    ];
    let descriptions: &[(&str, &str)] = &[
        ("400", "Bad Request — 请求体格式或字段无效。"),
        ("401", "Unauthorized — API key 缺失 / 无效。"),
        ("403", "Forbidden — 资源访问被拒，可能是配额耗尽。"),
        ("404", "Not Found — 模型 id 不存在或被弃用。"),
        ("413", "Payload Too Large — 请求体超出上游限制。"),
        ("429", "Too Many Requests — 触发 RPM / TPM 限速。"),
        ("500", "Internal Server Error — 上游异常。"),
        ("502", "Bad Gateway — 上游网关异常。"),
        ("503", "Service Unavailable — 上游过载或维护中。"),
    ];
    let desc_lookup: std::collections::HashMap<&str, &str> = descriptions.iter().copied().collect();

    let paths = spec
        .get_mut(Value::String("paths".into()))
        .context("spec.paths missing")?
        .as_mapping_mut()
        .context("spec.paths is not a mapping")?;

    for (path, methods) in KEEP_PATHS {
        let Some(item) = paths.get_mut(Value::String((*path).into())) else {
            continue;
        };
        let item_map = item
            .as_mapping_mut()
            .with_context(|| format!("paths[{path}] is not a mapping"))?;
        for m in *methods {
            let Some(op) = item_map.get_mut(Value::String((*m).into())) else {
                continue;
            };
            let op_map = op
                .as_mapping_mut()
                .with_context(|| format!("paths[{path}].{m} is not a mapping"))?;
            let responses = op_map
                .entry(Value::String("responses".into()))
                .or_insert_with(|| Value::Mapping(Mapping::new()))
                .as_mapping_mut()
                .with_context(|| format!("paths[{path}].{m}.responses is not a mapping"))?;
            for status in STATUSES {
                let key = Value::String((*status).to_string());
                if responses.contains_key(&key) {
                    continue;
                }
                let desc = desc_lookup.get(status).copied().unwrap_or("Error");
                responses.insert(key, error_response_node(desc));
            }
        }
    }
    Ok(())
}

/// 把 `CreateChatCompletionStreamResponse.choices.items.required` 数组里
/// 的 `finish_reason` 移除。流式响应的中间 chunk 实际带的是
/// `finish_reason: null`，留着 required 会让 codegen 投成 non-Option
/// 然后每条中间 chunk 反序失败。
fn relax_stream_finish_reason(spec: &mut Mapping) -> Result<()> {
    let Some(schemas) = spec
        .get_mut(Value::String("components".into()))
        .and_then(Value::as_mapping_mut)
        .and_then(|m| m.get_mut(Value::String("schemas".into())))
        .and_then(Value::as_mapping_mut)
    else {
        return Ok(());
    };
    let Some(stream_resp) = schemas
        .get_mut(Value::String("CreateChatCompletionStreamResponse".into()))
        .and_then(Value::as_mapping_mut)
    else {
        return Ok(());
    };
    let Some(choices) = stream_resp
        .get_mut(Value::String("properties".into()))
        .and_then(Value::as_mapping_mut)
        .and_then(|p| p.get_mut(Value::String("choices".into())))
        .and_then(Value::as_mapping_mut)
    else {
        return Ok(());
    };
    let Some(items) = choices
        .get_mut(Value::String("items".into()))
        .and_then(Value::as_mapping_mut)
    else {
        return Ok(());
    };
    let Some(required) = items
        .get_mut(Value::String("required".into()))
        .and_then(Value::as_sequence_mut)
    else {
        return Ok(());
    };
    required.retain(|v| v.as_str() != Some("finish_reason"));
    Ok(())
}

/// 在 `ChatCompletionRequestAssistantMessage.properties` 上挂一个
/// `reasoning_content`（nullable string，optional）。详见 §1.2 / §6.1。
fn inject_assistant_reasoning_content(spec: &mut Mapping) -> Result<()> {
    let Some(props) = spec
        .get_mut(Value::String("components".into()))
        .and_then(Value::as_mapping_mut)
        .and_then(|m| m.get_mut(Value::String("schemas".into())))
        .and_then(Value::as_mapping_mut)
        .and_then(|m| {
            m.get_mut(Value::String(
                "ChatCompletionRequestAssistantMessage".into(),
            ))
        })
        .and_then(Value::as_mapping_mut)
        .and_then(|m| m.get_mut(Value::String("properties".into())))
        .and_then(Value::as_mapping_mut)
    else {
        return Ok(());
    };
    if props.contains_key(Value::String("reasoning_content".into())) {
        return Ok(());
    }
    props.insert(
        Value::String("reasoning_content".into()),
        yaml_nullable_string_schema(
            "兼容厂商扩展（DeepSeek-v4-pro 等）：上一轮 assistant 的思考链文本，\
             thinking 模式下必须回放给服务端。OpenAI 官方忽略此字段。",
        ),
    );
    Ok(())
}

/// 在 spec 内某条路径的 mapping 节点上删 `discriminator` 字段。
///
/// 路径写法：`&["components", "schemas", "Foo", "items"]`，逐级 mapping 解引。
/// 任意一级不是 mapping 或缺失，则按 no-op 跳过——上游 schema 形态变化时
/// patch 自动失效，不会让 codegen 失败。
fn drop_discriminator_at(spec: &mut Mapping, paths: &[&[&str]]) -> Result<()> {
    for path in paths {
        let mut cursor: Option<&mut Mapping> = Some(spec);
        for segment in *path {
            let Some(map) = cursor else { break };
            cursor = map
                .get_mut(Value::String((*segment).into()))
                .and_then(Value::as_mapping_mut);
        }
        if let Some(map) = cursor {
            map.remove(Value::String("discriminator".into()));
        }
    }
    Ok(())
}

fn error_response_node(description: &str) -> Value {
    let mut schema_ref = Mapping::new();
    schema_ref.insert(
        Value::String("$ref".into()),
        Value::String("#/components/schemas/OpenAiErrorResponse".into()),
    );
    let mut json_branch = Mapping::new();
    json_branch.insert(Value::String("schema".into()), Value::Mapping(schema_ref));
    let mut content = Mapping::new();
    content.insert(
        Value::String("application/json".into()),
        Value::Mapping(json_branch),
    );
    let mut node = Mapping::new();
    node.insert(
        Value::String("description".into()),
        Value::String(description.to_owned()),
    );
    node.insert(Value::String("content".into()), Value::Mapping(content));
    Value::Mapping(node)
}

fn yaml_string_schema(description: &str) -> Value {
    let mut m = Mapping::new();
    m.insert(Value::String("type".into()), Value::String("string".into()));
    m.insert(
        Value::String("description".into()),
        Value::String(description.to_owned()),
    );
    Value::Mapping(m)
}

fn yaml_nullable_string_schema(description: &str) -> Value {
    let mut m = Mapping::new();
    m.insert(Value::String("type".into()), Value::String("string".into()));
    m.insert(Value::String("nullable".into()), Value::Bool(true));
    m.insert(
        Value::String("description".into()),
        Value::String(description.to_owned()),
    );
    Value::Mapping(m)
}

/// 删 paths 里非 KEEP_PATHS 的条目，并裁剪每个保留 path 下的方法。
/// 返回保留下来的 path → method 列表，给后续闭包扫描用。
fn filter_paths(spec: &mut Mapping) -> Result<Vec<(String, Vec<String>)>> {
    let paths = spec
        .get_mut(Value::String("paths".into()))
        .context("spec.paths missing")?
        .as_mapping_mut()
        .context("spec.paths is not a mapping")?;

    let keep: Vec<_> = paths
        .iter()
        .filter_map(|(k, _)| k.as_str().map(str::to_owned))
        .filter(|k| KEEP_PATHS.iter().any(|(p, _)| *p == k))
        .collect();
    let drop: Vec<_> = paths
        .iter()
        .filter_map(|(k, _)| k.as_str().map(str::to_owned))
        .filter(|k| !keep.contains(k))
        .collect();
    for k in drop {
        paths.remove(Value::String(k));
    }

    let mut kept = Vec::new();
    for (path, methods) in KEEP_PATHS {
        let Some(item) = paths.get_mut(Value::String((*path).into())) else {
            continue;
        };
        let item_map = item
            .as_mapping_mut()
            .with_context(|| format!("paths[{path}] is not a mapping"))?;
        // 只保留 KEEP_PATHS 指定的方法 + 共享字段（parameters / summary 等）。
        let allowed: BTreeSet<&str> = (*methods).iter().copied().collect();
        let common: BTreeSet<&str> = ["parameters", "summary", "description"]
            .into_iter()
            .collect();
        let drop: Vec<_> = item_map
            .iter()
            .filter_map(|(k, _)| k.as_str().map(str::to_owned))
            .filter(|k| !allowed.contains(k.as_str()) && !common.contains(k.as_str()))
            .collect();
        for k in drop {
            item_map.remove(Value::String(k));
        }
        // 进一步：每个保留方法里删掉 codegen 用不到的 x-* 扩展（x-oaiMeta
        // 这些含大段 example 字符串，留着只是噪音）。
        for m in *methods {
            if let Some(op) = item_map.get_mut(Value::String((*m).into()))
                && let Some(op_map) = op.as_mapping_mut()
            {
                let drop: Vec<_> = op_map
                    .iter()
                    .filter_map(|(k, _)| k.as_str().map(str::to_owned))
                    .filter(|k| k.starts_with("x-"))
                    .collect();
                for k in drop {
                    op_map.remove(Value::String(k));
                }
            }
        }
        let kept_methods: Vec<String> = methods
            .iter()
            .filter(|m| item_map.contains_key(Value::String((**m).into())))
            .map(|m| (*m).to_owned())
            .collect();
        kept.push(((*path).to_owned(), kept_methods));
    }
    Ok(kept)
}

/// 从 KEEP_PATHS 出发 BFS 扫所有 `$ref`，返回闭包内所有 component 的
/// "类型/名字" 元组（如 `("schemas", "CreateChatCompletionRequest")`）。
fn compute_ref_closure(
    spec: &Mapping,
    kept: &[(String, Vec<String>)],
) -> Result<BTreeSet<(String, String)>> {
    let paths = spec
        .get(Value::String("paths".into()))
        .context("spec.paths missing")?;
    let components = spec
        .get(Value::String("components".into()))
        .context("spec.components missing")?;

    let mut closure: BTreeSet<(String, String)> = BTreeSet::new();
    let mut queue: VecDeque<Value> = VecDeque::new();

    // seed：所有保留 method 的子树。
    for (path, methods) in kept {
        let Some(item) = paths.get(Value::String(path.into())) else {
            continue;
        };
        for m in methods {
            if let Some(op) = item.get(Value::String(m.into())) {
                queue.push_back(op.clone());
            }
        }
    }

    while let Some(node) = queue.pop_front() {
        for r in collect_refs(&node) {
            let Some((kind, name)) = parse_components_ref(&r) else {
                continue;
            };
            if !closure.insert((kind.clone(), name.clone())) {
                continue;
            }
            // 把该 component 的子树压回队列，继续扫。
            if let Some(child) = components
                .get(Value::String(kind.clone()))
                .and_then(|m| m.get(Value::String(name.clone())))
            {
                queue.push_back(child.clone());
            }
        }
    }

    Ok(closure)
}

/// 递归收集一个 YAML 子树里所有 `$ref` 的字符串值。
fn collect_refs(node: &Value) -> Vec<String> {
    let mut out = Vec::new();
    walk(node, &mut out);
    out
}

fn walk(node: &Value, out: &mut Vec<String>) {
    match node {
        Value::Mapping(m) => {
            for (k, v) in m {
                if k.as_str() == Some("$ref")
                    && let Some(s) = v.as_str()
                {
                    out.push(s.to_owned());
                    continue;
                }
                walk(v, out);
            }
        }
        Value::Sequence(seq) => {
            for v in seq {
                walk(v, out);
            }
        }
        _ => {}
    }
}

/// 解析 `#/components/<kind>/<name>` 形式 ref。其它形式（外部 spec、
/// fragments）当前不支持，按 None 跳过。
fn parse_components_ref(r: &str) -> Option<(String, String)> {
    let suffix = r.strip_prefix("#/components/")?;
    let mut parts = suffix.splitn(2, '/');
    let kind = parts.next()?.to_owned();
    let name = parts.next()?.to_owned();
    Some((kind, name))
}

/// 重写 components.<kind>，只保留闭包里出现过的项。
fn filter_components(spec: &mut Mapping, closure: &BTreeSet<(String, String)>) -> Result<()> {
    let components = spec
        .get_mut(Value::String("components".into()))
        .context("spec.components missing")?
        .as_mapping_mut()
        .context("spec.components is not a mapping")?;

    // securitySchemes 不通过 ref 引用，只通过顶层 security: [{ ApiKeyAuth: [] }]
    // 触发——保留全表（一般也很小）。
    let preserve_kinds: BTreeSet<&str> = ["securitySchemes"].into_iter().collect();

    let kinds: Vec<String> = components
        .iter()
        .filter_map(|(k, _)| k.as_str().map(str::to_owned))
        .collect();
    for kind in kinds {
        if preserve_kinds.contains(kind.as_str()) {
            continue;
        }
        let Some(submap_value) = components.get_mut(Value::String(kind.clone())) else {
            continue;
        };
        let Some(submap) = submap_value.as_mapping_mut() else {
            continue;
        };
        let drop: Vec<_> = submap
            .iter()
            .filter_map(|(k, _)| k.as_str().map(str::to_owned))
            .filter(|name| !closure.contains(&(kind.clone(), name.clone())))
            .collect();
        for name in drop {
            submap.remove(Value::String(name));
        }
        // 整个 kind 都空了：直接删 kind 节点。
        if submap.is_empty() {
            components.remove(Value::String(kind));
        }
    }
    Ok(())
}

/// 顶层只保留 codegen 关心的 key。tags 顺手裁到只剩用到的 `Chat` /
/// `Models`，省得无关 tag 把 doc string 搞乱。
fn prune_top_level(spec: &mut Mapping) {
    let keep_top: BTreeSet<&str> = [
        "openapi",
        "info",
        "servers",
        "security",
        "tags",
        "paths",
        "components",
    ]
    .into_iter()
    .collect();
    let drop: Vec<_> = spec
        .iter()
        .filter_map(|(k, _)| k.as_str().map(str::to_owned))
        .filter(|k| !keep_top.contains(k.as_str()))
        .collect();
    for k in drop {
        spec.remove(Value::String(k));
    }

    // tags 裁到只剩 Chat / Models（KEEP_PATHS 用到的 tag）。
    if let Some(tags) = spec.get_mut(Value::String("tags".into()))
        && let Some(seq) = tags.as_sequence_mut()
    {
        let allowed: BTreeSet<&str> = ["Chat", "Models"].into_iter().collect();
        seq.retain(|tag| {
            tag.get(Value::String("name".into()))
                .and_then(Value::as_str)
                .map(|n| allowed.contains(n))
                .unwrap_or(false)
        });
    }
}

fn strip_lines(input: &str, drop: impl Fn(&str) -> bool) -> String {
    let mut out = String::with_capacity(input.len());
    for line in input.split_inclusive('\n') {
        if !drop(line.trim_end_matches(['\n', '\r'])) {
            out.push_str(line);
        }
    }
    out
}

#[allow(dead_code)]
fn _ensure_path(_p: &Path) {}
