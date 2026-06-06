use std::path::PathBuf;

use toml::Value as TomlValue;

use crate::types::{CliOverrides, ConfigError, ConfigLayerEntry, ConfigSource};

/// 解析单个 `KEY=VALUE` 形式的 CLI 覆盖项。
///
/// # Errors
///
/// 当输入不包含 `=`，或 override path 为空时返回 [`ConfigError::Invalid`]。
pub fn parse_cli_override(spec: &str) -> Result<(String, TomlValue), ConfigError> {
    let Some((path, raw_value)) = spec.split_once('=') else {
        return Err(ConfigError::Invalid {
            path: PathBuf::from("<cli>"),
            message: format!("expected KEY=VALUE, got {spec:?}"),
        });
    };
    let path = path.trim();
    if path.is_empty() {
        return Err(ConfigError::Invalid {
            path: PathBuf::from("<cli>"),
            message: format!("override path must not be empty in {spec:?}"),
        });
    }

    let toml_snippet = format!("value = {}", raw_value.trim());
    let value = match toml_snippet.parse::<TomlValue>() {
        Ok(TomlValue::Table(mut table)) => table
            .remove("value")
            .unwrap_or_else(|| TomlValue::String(raw_value.trim().to_string())),
        Ok(_) => unreachable!("wrapper snippet always parses to a table"),
        Err(_) => TomlValue::String(raw_value.trim().to_string()),
    };
    Ok((path.to_string(), value))
}

pub(crate) fn build_cli_layer(cli: &CliOverrides) -> Result<Option<ConfigLayerEntry>, ConfigError> {
    let mut root = TomlValue::Table(Default::default());
    let mut has_values = false;

    for (path, value) in &cli.config_overrides {
        apply_toml_override(&mut root, path, value.clone());
        has_values = true;
    }
    if let Some(provider) = &cli.provider {
        apply_toml_override(
            &mut root,
            "default.provider",
            TomlValue::String(provider.to_string()),
        );
        has_values = true;
    }
    if let Some(model) = &cli.model {
        apply_toml_override(&mut root, "default.model", TomlValue::String(model.clone()));
        has_values = true;
    }
    if let Some(sandbox) = &cli.sandbox {
        apply_toml_override(
            &mut root,
            "sandbox.mode",
            TomlValue::String(sandbox.as_str().to_string()),
        );
        has_values = true;
    }

    if !has_values {
        return Ok(None);
    }

    Ok(Some(ConfigLayerEntry {
        source: ConfigSource::Cli,
        path: None,
        raw_toml: None,
        value: root,
    }))
}

pub(crate) fn merge_toml_values(base: &mut TomlValue, overlay: &TomlValue) {
    if let TomlValue::Table(base_table) = base
        && let TomlValue::Table(overlay_table) = overlay
    {
        for (key, value) in overlay_table {
            if let Some(existing) = base_table.get_mut(key) {
                merge_toml_values(existing, value);
            } else {
                base_table.insert(key.clone(), value.clone());
            }
        }
    } else {
        *base = overlay.clone();
    }
}

pub(crate) fn apply_toml_override(root: &mut TomlValue, path: &str, value: TomlValue) {
    use toml::map::Map;

    let mut current = root;
    let mut segments = path.split('.').peekable();
    while let Some(segment) = segments.next() {
        let is_last = segments.peek().is_none();
        if is_last {
            match current {
                TomlValue::Table(table) => {
                    table.insert(segment.to_string(), value);
                }
                _ => {
                    let mut table = Map::new();
                    table.insert(segment.to_string(), value);
                    *current = TomlValue::Table(table);
                }
            }
            return;
        }

        match current {
            TomlValue::Table(table) => {
                current = table
                    .entry(segment.to_string())
                    .or_insert_with(|| TomlValue::Table(Map::new()));
            }
            _ => {
                *current = TomlValue::Table(Map::new());
                let TomlValue::Table(table) = current else {
                    unreachable!();
                };
                current = table
                    .entry(segment.to_string())
                    .or_insert_with(|| TomlValue::Table(Map::new()));
            }
        }
    }
}
