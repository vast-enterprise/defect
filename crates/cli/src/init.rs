//! `defect init` — scan the environment for provider API keys, fetch each detected
//! provider's real model list over its API, and write a global
//! `~/.config/defect/config.toml`.
//!
//! Interaction model follows `rustup-init`: detect the environment, show what will be
//! written, confirm. When exactly one provider key is present and the user passes
//! `--yes`, it is fully non-interactive. With multiple keys, `--yes` requires an explicit
//! `--default-provider` — defect never guesses which provider should be the default from
//! ambient state (see the project's explicit-provider rule).
//!
//! Model IDs are NEVER hardcoded: with an API key in hand, init calls the provider's
//! `list_models()` (the same path the agent uses) to obtain the live, authoritative set.
//! A provider whose model list cannot be fetched is a hard error, not a fall back to a
//! guessed model id.
//!
//! The interactive prompts require the `init` build feature (on by default); a binary
//! built with `--no-default-features` can still run `defect init --yes`.

use std::fs;
use std::sync::Arc;

use defect_agent::llm::{LlmProvider, ModelInfo};

use crate::args::InitArgs;

/// A provider defect can write into the generated config.
struct ProviderSpec {
    /// The `[default] provider` / `[providers.<id>]` key.
    id: &'static str,
    /// Human label for prompts.
    display: &'static str,
    /// Environment variable that holds the API key.
    api_key_env: &'static str,
}

/// Providers eligible for auto-detection. Bedrock is intentionally excluded: it
/// authenticates through the AWS credential chain (profiles, IMDS, SSO), not a single
/// env var, so there is no reliable "key present" signal to scan for.
const PROVIDERS: &[ProviderSpec] = &[
    ProviderSpec {
        id: "anthropic",
        display: "Anthropic (Claude)",
        api_key_env: "ANTHROPIC_API_KEY",
    },
    ProviderSpec {
        id: "openai",
        display: "OpenAI",
        api_key_env: "OPENAI_API_KEY",
    },
    ProviderSpec {
        id: "deepseek",
        display: "DeepSeek",
        api_key_env: "DEEPSEEK_API_KEY",
    },
];

fn provider_by_id(id: &str) -> Option<&'static ProviderSpec> {
    PROVIDERS.iter().find(|p| p.id == id)
}

/// Providers whose API-key env var is set (non-empty), in [`PROVIDERS`] order.
fn detect_present() -> Vec<&'static ProviderSpec> {
    PROVIDERS
        .iter()
        .filter(|p| std::env::var(p.api_key_env).is_ok_and(|v| !v.trim().is_empty()))
        .collect()
}

/// Construct the LLM provider for `id` from the environment API key and return its live
/// model list. Errors (auth, transport, server, or a provider not compiled into this
/// build) are surfaced verbatim — init never substitutes a guessed model id.
async fn fetch_models(id: &str) -> anyhow::Result<Vec<ModelInfo>> {
    let provider = build_provider(id)?;
    let models = provider
        .list_models()
        .await
        .map_err(|e| anyhow::anyhow!("failed to list models for `{id}`: {e}"))?;
    if models.is_empty() {
        anyhow::bail!("provider `{id}` returned an empty model list");
    }
    Ok(models)
}

/// Build a provider instance for `id` using only environment credentials and default
/// endpoints. Mirrors the per-provider construction in `crate::providers`, but standalone
/// so it does not require a `LoadedConfig` (init runs before any config exists).
#[cfg_attr(
    not(any(
        feature = "provider-anthropic",
        feature = "provider-openai",
        feature = "provider-deepseek"
    )),
    allow(unused_variables)
)]
fn build_provider(id: &str) -> anyhow::Result<Arc<dyn LlmProvider>> {
    match id {
        #[cfg(feature = "provider-anthropic")]
        "anthropic" => {
            use defect_llm::provider::anthropic::{AnthropicConfig, AnthropicProvider};
            let provider = AnthropicProvider::new(AnthropicConfig::from_env())
                .map_err(|e| anyhow::anyhow!("anthropic provider init failed: {e}"))?;
            Ok(Arc::new(provider) as Arc<dyn LlmProvider>)
        }
        #[cfg(feature = "provider-openai")]
        "openai" => {
            use defect_llm::provider::openai::{OpenAiConfig, OpenAiProvider};
            let provider = OpenAiProvider::new(OpenAiConfig::from_env())
                .map_err(|e| anyhow::anyhow!("openai provider init failed: {e}"))?;
            Ok(Arc::new(provider) as Arc<dyn LlmProvider>)
        }
        #[cfg(feature = "provider-deepseek")]
        "deepseek" => {
            use defect_llm::provider::deepseek::{DeepSeekConfig, DeepSeekProvider};
            let provider = DeepSeekProvider::new(DeepSeekConfig::from_env())
                .map_err(|e| anyhow::anyhow!("deepseek provider init failed: {e}"))?;
            Ok(Arc::new(provider) as Arc<dyn LlmProvider>)
        }
        #[cfg(not(feature = "provider-anthropic"))]
        "anthropic" => Err(provider_not_compiled("anthropic")),
        #[cfg(not(feature = "provider-openai"))]
        "openai" => Err(provider_not_compiled("openai")),
        #[cfg(not(feature = "provider-deepseek"))]
        "deepseek" => Err(provider_not_compiled("deepseek")),
        other => Err(anyhow::anyhow!("unknown provider `{other}`")),
    }
}

#[cfg(not(all(
    feature = "provider-anthropic",
    feature = "provider-openai",
    feature = "provider-deepseek"
)))]
fn provider_not_compiled(feature_suffix: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "provider `{feature_suffix}` was not compiled into this build; \
         rebuild with `--features provider-{feature_suffix}` (or the default feature set)"
    )
}

/// Entry point for `defect init`.
pub async fn run(args: InitArgs) -> anyhow::Result<()> {
    let path = defect_config::user_config_path().ok_or_else(|| {
        anyhow::anyhow!(
            "cannot determine global config location: neither XDG_CONFIG_HOME nor HOME is set"
        )
    })?;

    if path.exists() && !args.force {
        anyhow::bail!(
            "global config already exists at {}\n\
             re-run with `--force` to overwrite it",
            path.display()
        );
    }

    let detected = detect_present();
    if detected.is_empty() {
        eprintln!(
            "No provider API keys found in the environment.\n\
             Set one of {} and re-run, or edit {} by hand.",
            PROVIDERS
                .iter()
                .map(|p| p.api_key_env)
                .collect::<Vec<_>>()
                .join(", "),
            path.display()
        );
        anyhow::bail!("nothing to configure: no provider API keys detected");
    }

    // Decide which providers to configure and which is the default — without touching the
    // network yet.
    let selection = if args.yes {
        select_non_interactive(&detected, args.default_provider.as_deref())?
    } else {
        select_interactive(&detected, args.default_provider.as_deref())?
    };
    if selection.providers.is_empty() {
        anyhow::bail!("no providers selected; aborting");
    }

    // Fetch the live model list for each selected provider. This is where the user's API
    // key is actually used; failures are hard errors.
    let mut configured: Vec<ConfiguredProvider> = Vec::new();
    for id in &selection.providers {
        eprintln!("Fetching models for {id}…");
        let models = fetch_models(id).await?;
        let model_ids: Vec<String> = models.into_iter().map(|m| m.id).collect();
        configured.push(ConfiguredProvider {
            id,
            models: model_ids,
        });
    }

    // Resolve the default model from the default provider's live list.
    let default_entry = configured
        .iter()
        .find(|c| c.id == selection.default_provider)
        .ok_or_else(|| anyhow::anyhow!("internal: default provider not among configured"))?;
    let default_model = resolve_default_model(
        default_entry,
        args.default_model.as_deref(),
        args.yes,
    )?;

    let plan = Plan {
        providers: configured,
        default_provider: selection.default_provider,
        default_model,
    };

    let body = render_config(&plan);

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| anyhow::anyhow!("failed to create {}: {e}", parent.display()))?;
    }
    fs::write(&path, body)
        .map_err(|e| anyhow::anyhow!("failed to write {}: {e}", path.display()))?;

    println!("Wrote global config to {}", path.display());
    println!(
        "Default: provider `{}`, model `{}`",
        plan.default_provider, plan.default_model
    );
    Ok(())
}

/// Which providers to configure and which is the default — decided before any network I/O.
#[derive(Debug)]
struct Selection {
    providers: Vec<&'static str>,
    default_provider: &'static str,
}

/// A provider plus its fetched (live) model ids.
#[derive(Debug)]
struct ConfiguredProvider {
    id: &'static str,
    models: Vec<String>,
}

/// The fully resolved decision to write.
struct Plan {
    providers: Vec<ConfiguredProvider>,
    default_provider: &'static str,
    default_model: String,
}

/// Non-interactive (`--yes`): configure every detected provider. The default is the sole
/// detected provider, or `--default-provider` when more than one is present.
fn select_non_interactive(
    detected: &[&'static ProviderSpec],
    default_provider: Option<&str>,
) -> anyhow::Result<Selection> {
    let providers: Vec<&'static str> = detected.iter().map(|p| p.id).collect();

    let default_provider = match default_provider {
        Some(id) => {
            let spec = provider_by_id(id)
                .ok_or_else(|| anyhow::anyhow!("unknown --default-provider `{id}`"))?;
            if !providers.contains(&spec.id) {
                anyhow::bail!(
                    "--default-provider `{}` has no API key in the environment ({} is unset)",
                    spec.id,
                    spec.api_key_env
                );
            }
            spec.id
        }
        None => match providers.as_slice() {
            [only] => only,
            // Explicit-provider rule: with multiple keys, defect will not pick for the
            // user under --yes.
            _ => anyhow::bail!(
                "multiple provider keys detected ({}); pass --default-provider <{}> to \
                 choose the default explicitly",
                providers.join(", "),
                providers.join("|")
            ),
        },
    };

    Ok(Selection {
        providers,
        default_provider,
    })
}

#[cfg(feature = "init")]
fn select_interactive(
    detected: &[&'static ProviderSpec],
    default_provider: Option<&str>,
) -> anyhow::Result<Selection> {
    use inquire::{MultiSelect, Select};

    let options: Vec<&'static str> = PROVIDERS.iter().map(|p| p.display).collect();
    let default_idx: Vec<usize> = PROVIDERS
        .iter()
        .enumerate()
        .filter(|(_, p)| detected.iter().any(|d| d.id == p.id))
        .map(|(i, _)| i)
        .collect();

    let chosen_displays = MultiSelect::new("Which providers do you want to configure?", options)
        .with_default(&default_idx)
        .with_help_message("space to toggle, enter to confirm; detected keys are pre-selected")
        .prompt()?;

    let providers: Vec<&'static str> = PROVIDERS
        .iter()
        .filter(|p| chosen_displays.contains(&p.display))
        .map(|p| p.id)
        .collect();

    if providers.is_empty() {
        return Ok(Selection {
            providers,
            default_provider: "",
        });
    }

    let default_provider = if let Some(id) = default_provider {
        let spec =
            provider_by_id(id).ok_or_else(|| anyhow::anyhow!("unknown --default-provider `{id}`"))?;
        if !providers.contains(&spec.id) {
            anyhow::bail!("--default-provider `{}` is not among the chosen providers", spec.id);
        }
        spec.id
    } else {
        match providers.as_slice() {
            [only] => only,
            _ => {
                let labels: Vec<&'static str> = providers
                    .iter()
                    .filter_map(|id| provider_by_id(id).map(|p| p.display))
                    .collect();
                let picked =
                    Select::new("Which provider should be the default?", labels).prompt()?;
                PROVIDERS
                    .iter()
                    .find(|p| p.display == picked)
                    .map(|p| p.id)
                    .or_else(|| providers.first().copied())
                    .unwrap_or("")
            }
        }
    };

    Ok(Selection {
        providers,
        default_provider,
    })
}

#[cfg(not(feature = "init"))]
fn select_interactive(
    _detected: &[&'static ProviderSpec],
    _default_provider: Option<&str>,
) -> anyhow::Result<Selection> {
    anyhow::bail!(
        "this binary was built without the `init` feature; \
         run `defect init --yes` for non-interactive setup, or rebuild with `--features init`"
    )
}

/// Pick the default model for the default provider from its live model list.
/// `--default-model` is validated against the list; under `--yes` with no flag the first
/// listed model is used; interactively the user selects one.
fn resolve_default_model(
    entry: &ConfiguredProvider,
    requested: Option<&str>,
    non_interactive: bool,
) -> anyhow::Result<String> {
    if let Some(model) = requested {
        if !entry.models.iter().any(|m| m == model) {
            anyhow::bail!(
                "--default-model `{model}` is not offered by `{}`; available: {}",
                entry.id,
                entry.models.join(", ")
            );
        }
        return Ok(model.to_string());
    }

    if non_interactive {
        // The first model the provider lists. Hard error already guards against empty.
        return entry
            .models
            .first()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("provider `{}` returned no models", entry.id));
    }

    pick_default_model_interactive(entry)
}

#[cfg(feature = "init")]
fn pick_default_model_interactive(entry: &ConfiguredProvider) -> anyhow::Result<String> {
    use inquire::Select;
    let choice = Select::new(
        &format!("Default model for `{}`?", entry.id),
        entry.models.clone(),
    )
    .prompt()?;
    Ok(choice)
}

#[cfg(not(feature = "init"))]
fn pick_default_model_interactive(_entry: &ConfiguredProvider) -> anyhow::Result<String> {
    // Unreachable in practice: non-interactive callers pass `non_interactive = true`,
    // and the `init` feature is required to reach the interactive selection path at all.
    anyhow::bail!("interactive model selection requires the `init` feature")
}

/// Render a commented `config.toml`. Hand-written (not serde) to preserve comments and
/// field ordering, matching the repo's TOML style.
fn render_config(plan: &Plan) -> String {
    let mut out = String::new();
    out.push_str("# defect global configuration — generated by `defect init`.\n");
    out.push_str("# Model lists were fetched live from each provider's API.\n");
    out.push_str("# Edit freely; unknown keys hard-fail with this file's path.\n\n");

    out.push_str("[default]\n");
    out.push_str("# Provider/model used when --provider / DEFECT_PROVIDER is not given.\n");
    out.push_str(&format!("provider = \"{}\"\n", plan.default_provider));
    out.push_str(&format!("model = \"{}\"\n\n", plan.default_model));

    for entry in &plan.providers {
        let display = provider_by_id(entry.id).map(|p| p.display).unwrap_or(entry.id);
        let api_key_env = provider_by_id(entry.id)
            .map(|p| p.api_key_env)
            .unwrap_or("");
        out.push_str(&format!("# {display} — key read from ${api_key_env}\n"));
        out.push_str(&format!("[providers.{}]\n", entry.id));
        out.push_str("models = [\n");
        for model in &entry.models {
            out.push_str(&format!("    \"{}\",\n", model.replace('"', "")));
        }
        out.push_str("]\n\n");
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(id: &str) -> &'static ProviderSpec {
        provider_by_id(id).expect("known provider")
    }

    fn configured(id: &'static str, models: &[&str]) -> ConfiguredProvider {
        ConfiguredProvider {
            id,
            models: models.iter().map(|m| m.to_string()).collect(),
        }
    }

    #[test]
    fn renders_with_fetched_models() {
        let plan = Plan {
            providers: vec![configured("deepseek", &["deepseek-v4-flash", "deepseek-v4-pro"])],
            default_provider: "deepseek",
            default_model: "deepseek-v4-pro".to_string(),
        };
        let body = render_config(&plan);
        assert!(body.contains("provider = \"deepseek\""));
        assert!(body.contains("model = \"deepseek-v4-pro\""));
        assert!(body.contains("\"deepseek-v4-flash\""));
        assert!(body.contains("\"deepseek-v4-pro\""));
        // No hardcoded guesses leaked in.
        assert!(!body.contains("deepseek-chat"));
        let _: toml::Value = body.parse().expect("valid toml");
    }

    #[test]
    fn select_single_picks_default() {
        let sel = select_non_interactive(&[spec("anthropic")], None).expect("select");
        assert_eq!(sel.default_provider, "anthropic");
        assert_eq!(sel.providers, vec!["anthropic"]);
    }

    #[test]
    fn select_multiple_requires_explicit_default() {
        let err = select_non_interactive(&[spec("anthropic"), spec("openai")], None)
            .expect_err("should require --default-provider");
        assert!(err.to_string().contains("--default-provider"), "{err}");
    }

    #[test]
    fn select_multiple_honors_explicit_default() {
        let sel = select_non_interactive(&[spec("anthropic"), spec("openai")], Some("openai"))
            .expect("select");
        assert_eq!(sel.default_provider, "openai");
        assert_eq!(sel.providers, vec!["anthropic", "openai"]);
    }

    #[test]
    fn select_rejects_undetected_default() {
        let err = select_non_interactive(&[spec("anthropic")], Some("deepseek"))
            .expect_err("deepseek not detected");
        assert!(err.to_string().contains("no API key"), "{err}");
    }

    #[test]
    fn select_rejects_unknown_provider() {
        let err =
            select_non_interactive(&[spec("anthropic")], Some("bogus")).expect_err("unknown");
        assert!(err.to_string().contains("unknown --default-provider"), "{err}");
    }

    #[test]
    fn default_model_yes_takes_first_listed() {
        let entry = configured("deepseek", &["deepseek-v4-flash", "deepseek-v4-pro"]);
        let model = resolve_default_model(&entry, None, true).expect("model");
        assert_eq!(model, "deepseek-v4-flash");
    }

    #[test]
    fn default_model_validates_against_live_list() {
        let entry = configured("deepseek", &["deepseek-v4-flash", "deepseek-v4-pro"]);
        let err = resolve_default_model(&entry, Some("deepseek-chat"), true)
            .expect_err("not offered");
        assert!(err.to_string().contains("not offered"), "{err}");
    }

    #[test]
    fn default_model_accepts_listed_model() {
        let entry = configured("deepseek", &["deepseek-v4-flash", "deepseek-v4-pro"]);
        let model = resolve_default_model(&entry, Some("deepseek-v4-pro"), true).expect("model");
        assert_eq!(model, "deepseek-v4-pro");
    }
}
