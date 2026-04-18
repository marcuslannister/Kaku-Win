use crate::assistant_config;
use crate::utils::{is_jsonc_path, open_path_in_editor, parse_json_or_jsonc, write_atomic};
use anyhow::Context;
use chrono::{DateTime, Utc};
use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEventKind, KeyModifiers,
    MouseEventKind,
};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::ffi::OsStr;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

mod ui;

#[derive(Clone, Copy, PartialEq)]
enum Tool {
    KakuAssistant,
    ClaudeCode,
    Codex,
    Kimi,
    Antigravity,
    Gemini,
    Copilot,
    FactoryDroid,
    OpenClaw,
}

impl Tool {
    fn label(&self) -> &'static str {
        match self {
            Tool::KakuAssistant => "Kaku Assistant",
            Tool::ClaudeCode => "Claude Code",
            Tool::Codex => "Codex",
            Tool::Kimi => "Kimi Code",
            Tool::Antigravity => "Antigravity",
            Tool::Gemini => "Gemini CLI",
            Tool::Copilot => "Copilot CLI",
            Tool::FactoryDroid => "Factory Droid",
            Tool::OpenClaw => "OpenClaw",
        }
    }

    fn config_path(&self) -> PathBuf {
        let home = config::HOME_DIR.clone();
        match self {
            Tool::KakuAssistant => assistant_config::assistant_toml_path()
                .unwrap_or_else(|_| config::KAKU_CONFIG_DIR.join("assistant.toml")),
            Tool::ClaudeCode => home.join(".claude").join("settings.json"),
            Tool::Codex => home.join(".codex").join("config.toml"),
            Tool::Kimi => home.join(".kimi").join("config.toml"),
            Tool::Antigravity => home
                .join("Library")
                .join("Application Support")
                .join("Antigravity")
                .join("User")
                .join("settings.json"),
            Tool::Gemini => home.join(".gemini").join("settings.json"),
            Tool::Copilot => home.join(".copilot").join("config.json"),
            Tool::FactoryDroid => home.join(".factory").join("settings.json"),
            Tool::OpenClaw => {
                let new_path = home.join(".openclaw").join("openclaw.json");
                if new_path.exists() {
                    return new_path;
                }
                let legacy = home.join(".clawdbot").join("clawdbot.json");
                if legacy.exists() {
                    return legacy;
                }
                new_path
            }
        }
    }
}

const ALL_TOOLS: [Tool; 9] = [
    Tool::KakuAssistant,
    Tool::ClaudeCode,
    Tool::Codex,
    Tool::Kimi,
    Tool::Antigravity,
    Tool::Gemini,
    Tool::Copilot,
    Tool::FactoryDroid,
    Tool::OpenClaw,
];

const FACTORY_DROID_DEFAULT_MODEL: &str = "opus";
const FACTORY_DROID_DEFAULT_REASONING: &str = "off";
const FACTORY_DROID_DEFAULT_AUTONOMY: &str = "normal";
const FACTORY_DROID_REASONING_OPTIONS: [&str; 5] = ["off", "none", "low", "medium", "high"];
const FACTORY_DROID_AUTONOMY_OPTIONS: [&str; 5] =
    ["normal", "spec", "auto-low", "auto-medium", "auto-high"];
const USAGE_CACHE_TTL: Duration = Duration::from_secs(120);
const ASSISTANT_MODELS_CACHE_TTL: Duration = Duration::from_secs(1800);
const EVENT_POLL_INTERVAL: Duration = Duration::from_millis(100);
const UI_STATUS_TTL: Duration = Duration::from_secs(3);
const UI_ERROR_TTL: Duration = Duration::from_secs(5);
const CLAUDE_OAUTH_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const CLAUDE_OAUTH_TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
const KIMI_OAUTH_CLIENT_ID: &str = "17e5f671-d194-4dfb-9706-5516cb48c098";
const KIMI_OAUTH_TOKEN_URL: &str = "https://auth.kimi.com/api/oauth/token";
const KIMI_DEFAULT_BASE_URL: &str = "https://api.kimi.com/coding/v1";
const ANTIGRAVITY_LSP_PROCESS_NAME: &str = "language_server_macos";
const ANTIGRAVITY_CSRF_HEADER: &str = "x-codeium-csrf-token";
const ANTIGRAVITY_GET_USER_STATUS_PATH: &str =
    "/exa.language_server_pb.LanguageServerService/GetUserStatus";
const ANTIGRAVITY_GET_COMMAND_MODEL_CONFIGS_PATH: &str =
    "/exa.language_server_pb.LanguageServerService/GetCommandModelConfigs";
const ANTIGRAVITY_GET_UNLEASH_DATA_PATH: &str =
    "/exa.language_server_pb.LanguageServerService/GetUnleashData";
const ANTIGRAVITY_CONNECT_PROTOCOL_VERSION: &str = "1";

static UI_ERRORS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

fn ui_errors() -> &'static Mutex<Vec<String>> {
    UI_ERRORS.get_or_init(|| Mutex::new(Vec::new()))
}

fn push_ui_error(message: impl Into<String>) {
    let message = message.into();
    let mut guard = match ui_errors().lock() {
        Ok(guard) => guard,
        Err(_) => return,
    };
    if guard.last().is_some_and(|last| last == &message) {
        return;
    }
    if guard.len() >= 8 {
        guard.remove(0);
    }
    guard.push(message);
}

fn pop_ui_error() -> Option<String> {
    let mut guard = ui_errors().lock().ok()?;
    if guard.is_empty() {
        None
    } else {
        Some(guard.remove(0))
    }
}

#[derive(Clone)]
struct FieldEntry {
    key: String,
    value: String,
    options: Vec<String>,
    editable: bool,
}

impl Default for FieldEntry {
    fn default() -> Self {
        Self {
            key: String::new(),
            value: String::new(),
            options: Vec::new(),
            editable: true,
        }
    }
}

struct ToolState {
    tool: Tool,
    installed: bool,
    fields: Vec<FieldEntry>,
    summary: Option<String>,
}

impl ToolState {
    fn load_without_remote_usage(tool: Tool) -> Self {
        Self::load_with_usage(tool, false)
    }

    fn load_with_usage(tool: Tool, eager_remote_usage: bool) -> Self {
        let path = if tool == Tool::KakuAssistant {
            match assistant_config::ensure_assistant_toml_exists() {
                Ok(path) => path,
                Err(err) => {
                    return ToolState {
                        tool,
                        installed: true,
                        fields: vec![FieldEntry {
                            key: "error".into(),
                            value: err.to_string(),
                            options: vec![],
                            editable: false,
                        }],
                        summary: Some("Setup failed".into()),
                    };
                }
            }
        } else {
            tool.config_path()
        };

        let extra_exists = match tool {
            Tool::Codex => config::HOME_DIR.join(".codex").join("auth.json").exists(),
            Tool::Kimi => config::HOME_DIR
                .join(".kimi")
                .join("credentials")
                .join("kimi-code.json")
                .exists(),
            _ => false,
        };

        if tool == Tool::Antigravity && !antigravity_app_bundle_path().exists() {
            return ToolState {
                tool,
                installed: false,
                fields: Vec::new(),
                summary: None,
            };
        }

        if tool != Tool::KakuAssistant && !path.exists() && !extra_exists {
            return ToolState {
                tool,
                installed: false,
                fields: Vec::new(),
                summary: None,
            };
        }

        let raw = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) if e.kind() == io::ErrorKind::NotFound && extra_exists => String::new(),
            Err(e) => {
                log::warn!("failed to read config for {}: {}", tool.label(), e);
                return ToolState {
                    tool,
                    installed: false,
                    fields: vec![FieldEntry {
                        key: "error".into(),
                        value: format!("failed to read config: {}", e),
                        options: vec![],
                        ..Default::default()
                    }],
                    summary: Some("Config unreadable".into()),
                };
            }
        };

        let (fields, usage_summary) = match tool {
            Tool::KakuAssistant => (extract_kaku_assistant_fields(&raw), None),
            Tool::ClaudeCode => {
                let parsed = parse_json_or_jsonc_with_debug(&raw, tool.label());
                (
                    extract_claude_code_fields(&parsed),
                    eager_remote_usage
                        .then(|| load_claude_usage_snapshot().and_then(|snapshot| snapshot.summary))
                        .flatten(),
                )
            }
            Tool::Codex => (
                extract_codex_fields(&raw),
                eager_remote_usage
                    .then(|| load_codex_usage_snapshot().and_then(|snapshot| snapshot.summary))
                    .flatten(),
            ),
            Tool::Kimi => (
                extract_kimi_fields(&raw),
                eager_remote_usage
                    .then(|| load_kimi_usage_snapshot().and_then(|snapshot| snapshot.summary))
                    .flatten(),
            ),
            Tool::Antigravity => {
                let snapshot = if eager_remote_usage {
                    load_antigravity_usage_snapshot()
                } else {
                    Some(load_cached_antigravity_usage_snapshot())
                };
                (
                    extract_antigravity_fields(snapshot.as_ref()),
                    snapshot.and_then(|snapshot| snapshot.summary),
                )
            }
            Tool::Gemini => {
                let parsed = parse_json_with_debug(&raw, tool.label());
                (
                    extract_gemini_fields(&parsed),
                    gemini_quota_summary(&parsed),
                )
            }
            Tool::Copilot => {
                let parsed = parse_json_with_debug(&raw, tool.label());
                (
                    extract_copilot_fields(&parsed),
                    eager_remote_usage
                        .then(|| {
                            load_copilot_usage_snapshot().and_then(|snapshot| snapshot.summary)
                        })
                        .flatten(),
                )
            }
            Tool::FactoryDroid => {
                let parsed = parse_json_with_debug(&raw, tool.label());
                (extract_factory_droid_fields(&parsed), None)
            }
            Tool::OpenClaw => {
                let parsed = parse_json_or_jsonc_with_debug(&raw, tool.label());
                (extract_openclaw_fields(&parsed), None)
            }
        };

        let summary = if !eager_remote_usage && supports_remote_usage(tool) {
            Some("Loading usage...".into())
        } else {
            summarize_tool_fields(tool, true, &fields, usage_summary.as_deref())
        };

        ToolState {
            tool,
            installed: true,
            fields,
            summary,
        }
    }
}

fn field_value<'a>(fields: &'a [FieldEntry], key: &str) -> Option<&'a str> {
    fields
        .iter()
        .find(|field| field.key == key)
        .map(|field| field.value.as_str())
}

fn compact_summary_value(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty() || value == "—" {
        return None;
    }

    let value = value
        .strip_prefix("✓ ")
        .or_else(|| value.strip_prefix("✗ "))
        .unwrap_or(value)
        .trim();
    let value = value.strip_suffix(" (default)").unwrap_or(value).trim();

    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

fn summarize_tool_fields(
    tool: Tool,
    installed: bool,
    fields: &[FieldEntry],
    usage_summary: Option<&str>,
) -> Option<String> {
    if !installed {
        return None;
    }

    if tool == Tool::KakuAssistant {
        let model = field_value(fields, "Model").and_then(compact_summary_value)?;
        let has_api_key = field_value(fields, "API Key").is_some_and(|value| value != "—");
        if has_api_key {
            return Some(format!("Ready · {model}"));
        }
        return Some(format!("Setup required · {model}"));
    }

    if matches!(
        tool,
        Tool::Codex | Tool::ClaudeCode | Tool::Kimi | Tool::Copilot
    ) {
        return usage_summary
            .map(str::to_string)
            .or_else(|| Some("Quota unavailable".into()));
    }

    if tool == Tool::Antigravity {
        return usage_summary.map(str::to_string).or_else(|| {
            field_value(fields, "Model")
                .map(|_| "Open Antigravity to sync quota".to_string())
                .or_else(|| Some("Open Antigravity to load quota".into()))
        });
    }

    if tool == Tool::Gemini {
        if let Some(summary) = usage_summary {
            return Some(summary.to_string());
        }
    }

    let mut parts = Vec::new();
    if let Some(account) = field_value(fields, "Auth").and_then(compact_summary_value) {
        parts.push(account);
    }

    let model = field_value(fields, "Model")
        .or_else(|| field_value(fields, "Primary Model"))
        .and_then(compact_summary_value);
    if let Some(model) = model {
        parts.push(model);
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" · "))
    }
}

struct CodexUsageSnapshot {
    summary: Option<String>,
}

struct ClaudeUsageSnapshot {
    summary: Option<String>,
}

struct CopilotUsageSnapshot {
    summary: Option<String>,
}

struct KimiUsageSnapshot {
    summary: Option<String>,
}

struct AntigravityUsageSnapshot {
    summary: Option<String>,
    selected_model_label: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
struct AntigravityQuotaWindow {
    model_id: Option<String>,
    label: String,
    remaining_fraction: f64,
    reset_at: Option<String>,
}

struct AntigravityProcessInfo {
    pid: u32,
    csrf_token: String,
    extension_server_port: Option<u16>,
}

#[derive(Clone)]
struct UsageSummaryUpdate {
    tool: Tool,
    summary: Option<String>,
    fields: Option<Vec<FieldEntry>>,
}

fn codex_usage_cache_path() -> PathBuf {
    config::HOME_DIR
        .join(".cache")
        .join("kaku")
        .join("codex_usage.json")
}

fn claude_usage_cache_path() -> PathBuf {
    config::HOME_DIR
        .join(".cache")
        .join("kaku")
        .join("claude_usage.json")
}

fn copilot_usage_cache_path() -> PathBuf {
    config::HOME_DIR
        .join(".cache")
        .join("kaku")
        .join("copilot_usage.json")
}

fn kimi_usage_cache_path() -> PathBuf {
    config::HOME_DIR
        .join(".cache")
        .join("kaku")
        .join("kimi_usage.json")
}

fn antigravity_usage_cache_path() -> PathBuf {
    config::HOME_DIR
        .join(".cache")
        .join("kaku")
        .join("antigravity_usage.json")
}

fn assistant_models_cache_path() -> PathBuf {
    config::HOME_DIR
        .join(".cache")
        .join("kaku")
        .join("assistant_models.json")
}

fn antigravity_app_bundle_path() -> PathBuf {
    PathBuf::from("/Applications/Antigravity.app")
}

fn antigravity_state_db_path() -> PathBuf {
    config::HOME_DIR
        .join("Library")
        .join("Application Support")
        .join("Antigravity")
        .join("User")
        .join("globalStorage")
        .join("state.vscdb")
}

fn read_codex_auth_info() -> Option<(String, String)> {
    let auth_path = config::HOME_DIR.join(".codex").join("auth.json");
    let auth_json = read_json_file_with_debug(&auth_path, "codex auth status")?;

    let access_token = auth_json
        .get("tokens")
        .and_then(|tokens| tokens.get("access_token"))
        .and_then(|value| value.as_str())
        .or_else(|| {
            auth_json
                .get("access_token")
                .and_then(|value| value.as_str())
        })?
        .to_string();

    let account_id = auth_json
        .get("tokens")
        .and_then(|tokens| tokens.get("account_id"))
        .and_then(|value| value.as_str())
        .or_else(|| auth_json.get("account_id").and_then(|value| value.as_str()))
        .map(|value| value.to_string())
        .or_else(|| {
            decode_jwt_payload_with_debug(&access_token, "codex auth status").and_then(|payload| {
                payload
                    .get("chatgpt_account_id")
                    .and_then(|value| value.as_str())
                    .map(|value| value.to_string())
            })
        })?;

    Some((access_token, account_id))
}

fn read_sqlite_value_with_debug(path: &Path, query: &str, context: &str) -> Option<String> {
    use rusqlite::types::ValueRef;
    use rusqlite::{Connection, OpenFlags};

    if !path.exists() {
        log::debug!("{context}: sqlite db missing at {}", path.display());
        return None;
    }

    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|err| log::debug!("{context}: sqlite open failed: {}", err))
        .ok()?;
    let mut stmt = conn
        .prepare(query)
        .map_err(|err| log::debug!("{context}: sqlite prepare failed: {}", err))
        .ok()?;
    let mut rows = stmt
        .query([])
        .map_err(|err| log::debug!("{context}: sqlite query failed: {}", err))
        .ok()?;
    let row = rows
        .next()
        .map_err(|err| log::debug!("{context}: sqlite row fetch failed: {}", err))
        .ok()??;
    let value = row
        .get_ref(0)
        .map_err(|err| log::debug!("{context}: sqlite value read failed: {}", err))
        .ok()?;

    let text = match value {
        ValueRef::Null => return None,
        ValueRef::Text(bytes) => std::str::from_utf8(bytes)
            .map_err(|err| log::debug!("{context}: sqlite text value is not utf-8: {}", err))
            .ok()?
            .to_string(),
        ValueRef::Blob(bytes) => String::from_utf8(bytes.to_vec())
            .map_err(|err| log::debug!("{context}: sqlite blob value is not utf-8: {}", err))
            .ok()?,
        ValueRef::Integer(value) => value.to_string(),
        ValueRef::Real(value) => value.to_string(),
    };

    let value = text.trim();
    if value.is_empty() {
        return None;
    }
    Some(value.to_string())
}

fn decode_base64_standard_with_debug(raw: &str, context: &str) -> Option<Vec<u8>> {
    use base64::Engine;

    base64::engine::general_purpose::STANDARD
        .decode(raw)
        .map_err(|err| log::debug!("{context}: base64 decode failed: {}", err))
        .ok()
}

fn read_protobuf_varint(bytes: &[u8], idx: &mut usize) -> Option<u64> {
    let mut shift = 0;
    let mut value = 0u64;
    while *idx < bytes.len() {
        let byte = bytes[*idx];
        *idx += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Some(value);
        }
        shift += 7;
        if shift > 63 {
            return None;
        }
    }
    None
}

fn read_protobuf_bytes<'a>(bytes: &'a [u8], idx: &mut usize) -> Option<&'a [u8]> {
    let len = usize::try_from(read_protobuf_varint(bytes, idx)?).ok()?;
    let start = *idx;
    let end = start.checked_add(len)?;
    let slice = bytes.get(start..end)?;
    *idx = end;
    Some(slice)
}

fn skip_protobuf_field(bytes: &[u8], idx: &mut usize, wire_type: u64) -> Option<()> {
    match wire_type {
        0 => {
            let _ = read_protobuf_varint(bytes, idx)?;
        }
        1 => {
            *idx = idx.checked_add(8)?;
        }
        2 => {
            let _ = read_protobuf_bytes(bytes, idx)?;
        }
        5 => {
            *idx = idx.checked_add(4)?;
        }
        _ => return None,
    }
    Some(())
}

fn parse_antigravity_state_value_container(bytes: &[u8]) -> Option<String> {
    let mut idx = 0;
    while idx < bytes.len() {
        let tag = read_protobuf_varint(bytes, &mut idx)?;
        let field_number = tag >> 3;
        let wire_type = tag & 0x7;
        if field_number == 1 && wire_type == 2 {
            let value = read_protobuf_bytes(bytes, &mut idx)?;
            return String::from_utf8(value.to_vec()).ok();
        }
        skip_protobuf_field(bytes, &mut idx, wire_type)?;
    }
    None
}

fn parse_antigravity_state_entry(bytes: &[u8]) -> Option<(String, String)> {
    let mut idx = 0;
    let mut key = None;
    let mut value = None;
    while idx < bytes.len() {
        let tag = read_protobuf_varint(bytes, &mut idx)?;
        let field_number = tag >> 3;
        let wire_type = tag & 0x7;
        match (field_number, wire_type) {
            (1, 2) => {
                let raw_key = read_protobuf_bytes(bytes, &mut idx)?;
                key = String::from_utf8(raw_key.to_vec()).ok();
            }
            (2, 2) => {
                let nested = read_protobuf_bytes(bytes, &mut idx)?;
                value = parse_antigravity_state_value_container(nested);
            }
            _ => skip_protobuf_field(bytes, &mut idx, wire_type)?,
        }
    }
    Some((key?, value?))
}

fn parse_antigravity_unified_state(raw: &str) -> Option<Vec<(String, String)>> {
    let decoded = decode_base64_standard_with_debug(raw, "antigravity unified state")?;
    let mut idx = 0;
    let mut entries = Vec::new();
    while idx < decoded.len() {
        let tag = read_protobuf_varint(&decoded, &mut idx)?;
        let field_number = tag >> 3;
        let wire_type = tag & 0x7;
        match (field_number, wire_type) {
            (1, 2) => {
                let entry = read_protobuf_bytes(&decoded, &mut idx)?;
                if let Some(parsed) = parse_antigravity_state_entry(entry) {
                    entries.push(parsed);
                }
            }
            _ => skip_protobuf_field(&decoded, &mut idx, wire_type)?,
        }
    }
    Some(entries)
}

fn decode_antigravity_int32_value(raw: &str) -> Option<i32> {
    let decoded = decode_base64_standard_with_debug(raw, "antigravity int32 state")?;
    let mut idx = 0;
    while idx < decoded.len() {
        let tag = read_protobuf_varint(&decoded, &mut idx)?;
        let field_number = tag >> 3;
        let wire_type = tag & 0x7;
        match (field_number, wire_type) {
            (2, 0) => {
                let value = read_protobuf_varint(&decoded, &mut idx)?;
                return i32::try_from(value).ok();
            }
            _ => skip_protobuf_field(&decoded, &mut idx, wire_type)?,
        }
    }
    None
}

fn read_antigravity_storage_value(key: &str, context: &str) -> Option<String> {
    let escaped_key = key.replace('\'', "''");
    let query = format!("select value from ItemTable where key='{escaped_key}';");
    read_sqlite_value_with_debug(&antigravity_state_db_path(), &query, context)
}

fn read_antigravity_auth_status() -> Option<serde_json::Value> {
    let raw = read_antigravity_storage_value("antigravityAuthStatus", "antigravity auth status")?;
    serde_json::from_str(&raw)
        .map_err(|err| log::debug!("antigravity auth status JSON parse failed: {}", err))
        .ok()
}

#[cfg(test)]
fn extract_antigravity_printable_strings(bytes: &[u8]) -> Vec<String> {
    let mut current = Vec::new();
    let mut strings = Vec::new();
    for byte in bytes {
        if byte.is_ascii_graphic() || *byte == b' ' {
            current.push(*byte);
        } else if current.len() >= 6 {
            strings.push(String::from_utf8_lossy(&current).trim().to_string());
            current.clear();
        } else {
            current.clear();
        }
    }
    if current.len() >= 6 {
        strings.push(String::from_utf8_lossy(&current).trim().to_string());
    }
    strings
}

#[cfg(test)]
fn extract_antigravity_plan_name(raw: &str) -> Option<String> {
    let decoded = decode_base64_standard_with_debug(raw, "antigravity user status")?;
    extract_antigravity_printable_strings(&decoded)
        .into_iter()
        .find(|candidate| candidate.starts_with("Google AI "))
}

fn antigravity_arg_value<'a>(args: &'a [&'a str], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|arg| *arg == key)
        .and_then(|idx| args.get(idx + 1).copied())
}

fn parse_antigravity_process_info_line(line: &str) -> Option<AntigravityProcessInfo> {
    let mut parts = line.split_whitespace();
    let pid = parts.next()?.parse::<u32>().ok()?;
    let args = parts.collect::<Vec<_>>();

    if !args
        .iter()
        .any(|arg| arg.contains(ANTIGRAVITY_LSP_PROCESS_NAME))
    {
        return None;
    }

    // Restrict to the desktop app process to avoid collisions with unrelated servers.
    let is_antigravity_app = args
        .windows(2)
        .any(|pair| pair[0] == "--app_data_dir" && pair[1] == "antigravity");
    if !is_antigravity_app {
        return None;
    }

    let csrf_token = antigravity_arg_value(&args, "--csrf_token")?.to_string();
    let extension_server_port = antigravity_arg_value(&args, "--extension_server_port")
        .and_then(|value| value.parse::<u16>().ok());

    Some(AntigravityProcessInfo {
        pid,
        csrf_token,
        extension_server_port,
    })
}

fn find_antigravity_process_info() -> Option<AntigravityProcessInfo> {
    let output = std::process::Command::new("/bin/ps")
        .args(["-ax", "-o", "pid=,command="])
        .output()
        .ok()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log::debug!(
            "antigravity process probe failed with status {}: {}",
            output.status,
            stderr.trim()
        );
        return None;
    }

    let stdout = String::from_utf8(output.stdout)
        .map_err(|err| log::debug!("antigravity process probe returned non-utf8: {}", err))
        .ok()?;
    stdout
        .lines()
        .filter_map(parse_antigravity_process_info_line)
        .max_by_key(|info| info.pid)
}

fn parse_antigravity_listen_port(line: &str) -> Option<u16> {
    // Example line:
    // language_ 34643 tang ... TCP 127.0.0.1:56503 (LISTEN)
    let token = line.split_whitespace().find(|token| token.contains(':'))?;
    let (_, port) = token.rsplit_once(':')?;
    port.parse::<u16>().ok()
}

fn read_antigravity_listen_ports(pid: u32) -> Vec<u16> {
    let output = match std::process::Command::new("/usr/sbin/lsof")
        .args(["-nP", "-iTCP", "-sTCP:LISTEN", "-a", "-p", &pid.to_string()])
        .output()
    {
        Ok(output) => output,
        Err(err) => {
            log::debug!("antigravity lsof probe failed to launch: {}", err);
            return Vec::new();
        }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log::debug!(
            "antigravity lsof probe failed with status {}: {}",
            output.status,
            stderr.trim()
        );
        return Vec::new();
    }

    let stdout = match String::from_utf8(output.stdout) {
        Ok(stdout) => stdout,
        Err(err) => {
            log::debug!("antigravity lsof probe returned non-utf8: {}", err);
            return Vec::new();
        }
    };

    stdout
        .lines()
        .skip(1)
        .filter_map(parse_antigravity_listen_port)
        .collect()
}

fn post_antigravity_lsp_json(
    https_port: u16,
    csrf_token: &str,
    path: &str,
    payload: &serde_json::Value,
) -> Option<serde_json::Value> {
    let payload = serde_json::to_string(payload)
        .map_err(|err| log::debug!("antigravity payload serialize failed: {}", err))
        .ok()?;
    let url = format!("https://127.0.0.1:{https_port}{path}");

    // Token is sent through argv for portability across environments.
    let output = std::process::Command::new("/usr/bin/curl")
        .args([
            "-k",
            "-sS",
            "--max-time",
            "3",
            "-X",
            "POST",
            &url,
            "-H",
            "Content-Type: application/json",
            "-H",
            &format!(
                "Connect-Protocol-Version: {}",
                ANTIGRAVITY_CONNECT_PROTOCOL_VERSION
            ),
            "-H",
            &format!("{ANTIGRAVITY_CSRF_HEADER}: {csrf_token}"),
            "--data",
            &payload,
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log::debug!(
            "antigravity lsp request failed for {} with status {}: {}",
            path,
            output.status,
            stderr.trim()
        );
        return None;
    }

    let raw = String::from_utf8(output.stdout)
        .map_err(|err| log::debug!("antigravity lsp response non-utf8 for {}: {}", path, err))
        .ok()?;
    let parsed = serde_json::from_str::<serde_json::Value>(&raw)
        .map_err(|err| {
            log::debug!(
                "antigravity lsp response JSON parse failed for {}: {}",
                path,
                err
            )
        })
        .ok()?;

    // Responses shaped like {"code":"unauthenticated",...} indicate auth mismatch.
    if parsed.get("code").is_some() && parsed.get("message").is_some() {
        let code = parsed
            .get("code")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        if matches!(code, "unauthenticated" | "permission_denied") {
            return None;
        }
    }

    Some(parsed)
}

fn antigravity_unleash_probe(https_port: u16, csrf_token: &str) -> bool {
    post_antigravity_lsp_json(
        https_port,
        csrf_token,
        ANTIGRAVITY_GET_UNLEASH_DATA_PATH,
        &serde_json::json!({}),
    )
    .is_some()
}

fn discover_antigravity_https_port(process: &AntigravityProcessInfo) -> Option<u16> {
    let mut candidates = Vec::new();
    if let Some(extension_port) = process.extension_server_port {
        if extension_port < u16::MAX {
            candidates.push(extension_port + 1);
        }
        candidates.push(extension_port);
    }
    candidates.extend(read_antigravity_listen_ports(process.pid));

    let mut seen = HashSet::new();
    candidates.retain(|port| seen.insert(*port));
    candidates
        .into_iter()
        .find(|port| antigravity_unleash_probe(*port, &process.csrf_token))
}

fn fetch_antigravity_usage_json() -> Option<serde_json::Value> {
    let process = find_antigravity_process_info()?;
    let https_port = discover_antigravity_https_port(&process)?;

    let user_status = post_antigravity_lsp_json(
        https_port,
        &process.csrf_token,
        ANTIGRAVITY_GET_USER_STATUS_PATH,
        &serde_json::json!({}),
    )?;
    let command_model_configs = post_antigravity_lsp_json(
        https_port,
        &process.csrf_token,
        ANTIGRAVITY_GET_COMMAND_MODEL_CONFIGS_PATH,
        &serde_json::json!({}),
    )
    .unwrap_or(serde_json::Value::Null);

    Some(serde_json::json!({
        "fetched_at": Utc::now().to_rfc3339(),
        "user_status": user_status,
        "command_model_configs": command_model_configs,
    }))
}

fn antigravity_value_as_f64(value: &serde_json::Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|value| value as f64))
        .or_else(|| value.as_u64().map(|value| value as f64))
        .or_else(|| value.as_str()?.parse::<f64>().ok())
}

fn collect_antigravity_model_name_map(
    value: &serde_json::Value,
    model_name_map: &mut HashMap<String, String>,
) {
    match value {
        serde_json::Value::Object(map) => {
            let model_id = map
                .get("modelId")
                .or_else(|| map.get("id"))
                .and_then(|value| value.as_str())
                .filter(|value| !value.is_empty());
            let display_name = map
                .get("modelDisplayName")
                .or_else(|| map.get("displayName"))
                .or_else(|| map.get("name"))
                .and_then(|value| value.as_str())
                .filter(|value| !value.is_empty());
            if let (Some(model_id), Some(display_name)) = (model_id, display_name) {
                model_name_map
                    .entry(model_id.to_string())
                    .or_insert_with(|| display_name.to_string());
            }
            for child in map.values() {
                collect_antigravity_model_name_map(child, model_name_map);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_antigravity_model_name_map(item, model_name_map);
            }
        }
        _ => {}
    }
}

fn antigravity_model_name_map(
    command_model_configs: &serde_json::Value,
) -> HashMap<String, String> {
    let mut model_name_map = HashMap::new();
    collect_antigravity_model_name_map(command_model_configs, &mut model_name_map);
    model_name_map
}

fn antigravity_strip_parenthetical_suffix(label: &str) -> String {
    let trimmed = label.trim();
    if !trimmed.ends_with(')') {
        return trimmed.to_string();
    }
    let Some(idx) = trimmed.rfind(" (") else {
        return trimmed.to_string();
    };
    let candidate = trimmed[..idx].trim();
    if candidate.is_empty() {
        trimmed.to_string()
    } else {
        candidate.to_string()
    }
}

fn antigravity_window_label(
    object: &serde_json::Map<String, serde_json::Value>,
    model_name_map: &HashMap<String, String>,
) -> Option<(Option<String>, String)> {
    let model_id = object
        .get("modelId")
        .or_else(|| object.get("id"))
        .or_else(|| {
            object
                .get("modelOrAlias")
                .and_then(|value| value.get("model"))
        })
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .map(|value| value.to_string());

    let raw_label = object
        .get("modelDisplayName")
        .or_else(|| object.get("displayName"))
        .or_else(|| object.get("label"))
        .or_else(|| object.get("modelName"))
        .or_else(|| object.get("name"))
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .map(|value| value.to_string())
        .or_else(|| model_id.clone())?;

    let resolved_from_id = model_id
        .as_ref()
        .and_then(|model_id| model_name_map.get(model_id))
        .cloned();
    let label = resolved_from_id
        .or_else(|| model_name_map.get(&raw_label).cloned())
        .unwrap_or_else(|| raw_label.clone());
    let label = antigravity_strip_parenthetical_suffix(&label);
    Some((model_id, label))
}

fn antigravity_quota_window_from_object(
    object: &serde_json::Map<String, serde_json::Value>,
    model_name_map: &HashMap<String, String>,
) -> Option<AntigravityQuotaWindow> {
    let quota_info = object.get("quotaInfo").and_then(|value| value.as_object());

    let mut remaining_fraction = object
        .get("remainingFraction")
        .or_else(|| object.get("remaining_fraction"))
        .or_else(|| object.get("remaining"))
        .or_else(|| quota_info.and_then(|quota| quota.get("remainingFraction")))
        .or_else(|| quota_info.and_then(|quota| quota.get("remaining_fraction")))
        .or_else(|| quota_info.and_then(|quota| quota.get("remaining")))
        .and_then(antigravity_value_as_f64)?;
    if remaining_fraction > 1.0 && remaining_fraction <= 100.0 {
        remaining_fraction /= 100.0;
    }
    if !(0.0..=1.0).contains(&remaining_fraction) {
        remaining_fraction = remaining_fraction.clamp(0.0, 1.0);
    }

    let reset_at = object
        .get("resetAt")
        .or_else(|| object.get("resetsAt"))
        .or_else(|| object.get("resetTime"))
        .or_else(|| object.get("reset_at"))
        .or_else(|| quota_info.and_then(|quota| quota.get("resetAt")))
        .or_else(|| quota_info.and_then(|quota| quota.get("resetsAt")))
        .or_else(|| quota_info.and_then(|quota| quota.get("resetTime")))
        .or_else(|| quota_info.and_then(|quota| quota.get("reset_at")))
        .and_then(|value| {
            value
                .as_str()
                .map(|value| value.to_string())
                .or_else(|| value.as_i64().map(|value| value.to_string()))
                .or_else(|| value.as_u64().map(|value| value.to_string()))
        });

    let (model_id, label) = antigravity_window_label(object, model_name_map)?;
    Some(AntigravityQuotaWindow {
        model_id,
        label,
        remaining_fraction,
        reset_at,
    })
}

fn collect_antigravity_quota_windows_from_value(
    value: &serde_json::Value,
    model_name_map: &HashMap<String, String>,
    windows: &mut Vec<AntigravityQuotaWindow>,
) {
    match value {
        serde_json::Value::Array(items) => {
            for item in items {
                if let Some(map) = item.as_object() {
                    if let Some(window) = antigravity_quota_window_from_object(map, model_name_map)
                    {
                        windows.push(window);
                    }
                }
            }
        }
        serde_json::Value::Object(map) => {
            if let Some(window) = antigravity_quota_window_from_object(map, model_name_map) {
                windows.push(window);
            }
        }
        _ => {}
    }
}

fn collect_antigravity_quota_windows_from_paths(
    root: &serde_json::Value,
    paths: &[&str],
    model_name_map: &HashMap<String, String>,
    windows: &mut Vec<AntigravityQuotaWindow>,
) {
    for path in paths {
        if let Some(value) = root.pointer(path) {
            collect_antigravity_quota_windows_from_value(value, model_name_map, windows);
        }
    }
}

fn antigravity_command_model_ids(command_model_configs: &serde_json::Value) -> Vec<String> {
    const ARRAY_PATHS: [&str; 2] = ["/clientModelConfigs", "/configs"];
    let mut model_ids = Vec::new();

    for path in ARRAY_PATHS {
        let Some(items) = command_model_configs
            .pointer(path)
            .and_then(|value| value.as_array())
        else {
            continue;
        };
        for item in items {
            let Some(map) = item.as_object() else {
                continue;
            };
            let model_id = map
                .get("modelId")
                .or_else(|| map.get("id"))
                .or_else(|| map.get("modelOrAlias").and_then(|value| value.get("model")))
                .and_then(|value| value.as_str())
                .filter(|value| !value.is_empty());
            if let Some(model_id) = model_id {
                model_ids.push(model_id.to_string());
            }
        }
    }

    model_ids.sort();
    model_ids.dedup();
    model_ids
}

fn antigravity_format_reset_time(raw: &str) -> Option<String> {
    raw.parse::<i64>()
        .ok()
        .and_then(format_reset_time_from_epoch)
        .or_else(|| format_reset_time_from_iso(raw))
}

fn antigravity_format_quota_value(window: &AntigravityQuotaWindow) -> String {
    let mut value = format!(
        "remain {}",
        format_percent_value((window.remaining_fraction * 100.0).clamp(0.0, 100.0))
    );
    if let Some(reset_in) = window
        .reset_at
        .as_deref()
        .and_then(antigravity_format_reset_time)
    {
        value.push_str(" · reset ");
        value.push_str(&reset_in);
    }
    value
}

fn antigravity_selected_model_sentinel() -> Option<i32> {
    #[cfg(test)]
    {
        return None;
    }

    #[cfg(not(test))]
    let raw = read_antigravity_storage_value(
        "antigravityUnifiedStateSync.modelPreferences",
        "antigravity model preferences",
    )?;
    #[cfg(not(test))]
    let entries = parse_antigravity_unified_state(&raw)?;
    #[cfg(not(test))]
    entries
        .into_iter()
        .find(|(key, _)| key == "last_selected_agent_model_sentinel_key")
        .and_then(|(_, value)| decode_antigravity_int32_value(&value))
}

fn antigravity_model_id_from_sentinel(sentinel: i32, model_ids: &[String]) -> Option<String> {
    if sentinel <= 0 {
        return None;
    }

    let mut candidates = Vec::new();
    candidates.push(sentinel);
    if sentinel >= 1000 {
        candidates.push(sentinel - 1000);
    }
    if sentinel >= 100 {
        candidates.push(sentinel % 1000);
    }
    candidates.retain(|candidate| *candidate > 0);
    candidates.sort_unstable();
    candidates.dedup();

    for candidate in candidates {
        let exact = format!("MODEL_PLACEHOLDER_M{candidate}");
        if model_ids.iter().any(|model_id| model_id == &exact) {
            return Some(exact);
        }

        let suffix = format!("_M{candidate}");
        if let Some(found) = model_ids
            .iter()
            .find(|model_id| model_id.ends_with(&suffix))
            .cloned()
        {
            return Some(found);
        }
    }

    None
}

fn antigravity_model_label_from_sentinel_value(value: i32) -> Option<&'static str> {
    // Last-resort fallback when live LSP fetch, cached usage data, and
    // model-id resolution all fail. These sentinel values come from
    // Antigravity's internal model enum and may need refreshing as the app
    // updates its bundled model list.
    match value {
        18 => Some("Gemini 3 Flash"),
        26 => Some("Claude Opus 4.6"),
        35 => Some("Claude Sonnet 4.6"),
        36 | 37 => Some("Gemini 3.1 Pro"),
        _ => None,
    }
}

fn antigravity_fallback_selected_model_label() -> Option<String> {
    let sentinel = antigravity_selected_model_sentinel()?;
    let mut candidates = Vec::new();
    candidates.push(sentinel);
    if sentinel >= 1000 {
        candidates.push(sentinel - 1000);
    }
    if sentinel >= 100 {
        candidates.push(sentinel % 1000);
    }
    candidates.sort_unstable();
    candidates.dedup();

    candidates
        .into_iter()
        .find_map(|value| antigravity_model_label_from_sentinel_value(value).map(str::to_string))
}

fn antigravity_selected_model_id(
    user_status: &serde_json::Value,
    command_model_configs: &serde_json::Value,
    model_ids: &[String],
) -> Option<String> {
    const COMMAND_MODEL_PATHS: [&str; 2] = [
        "/selectedModelConfig/modelOrAlias/model",
        "/commandModelConfig/modelOrAlias/model",
    ];
    if let Some(model_id) = COMMAND_MODEL_PATHS.iter().find_map(|path| {
        command_model_configs
            .pointer(path)
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .map(|value| value.to_string())
    }) {
        return Some(model_id);
    }

    // defaultOverrideModelConfig tracks the current Antigravity model in live status.
    // A single entry in GetCommandModelConfigs can simply mean a command default,
    // so we only use that endpoint as a last-resort fallback.
    const PATHS: [&str; 4] = [
        "/userStatus/cascadeModelConfigData/defaultOverrideModelConfig/modelOrAlias/model",
        "/cascadeModelConfigData/defaultOverrideModelConfig/modelOrAlias/model",
        "/userStatus/defaultOverrideModelConfig/modelOrAlias/model",
        "/defaultOverrideModelConfig/modelOrAlias/model",
    ];
    if let Some(model_id) = PATHS.iter().find_map(|path| {
        user_status
            .pointer(path)
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .map(|value| value.to_string())
    }) {
        return Some(model_id);
    }

    if let Some(sentinel) = antigravity_selected_model_sentinel() {
        if let Some(model_id) = antigravity_model_id_from_sentinel(sentinel, model_ids) {
            return Some(model_id);
        }
    }

    let command_model_ids = antigravity_command_model_ids(command_model_configs);
    if let [model_id] = command_model_ids.as_slice() {
        return Some(model_id.clone());
    }

    None
}

fn parse_antigravity_usage_snapshot(data: &serde_json::Value) -> Option<AntigravityUsageSnapshot> {
    let user_status = data
        .get("user_status")
        .or_else(|| data.get("userStatus"))
        .unwrap_or(data);
    let command_model_configs = data
        .get("command_model_configs")
        .or_else(|| data.get("commandModelConfigs"))
        .unwrap_or(&serde_json::Value::Null);

    let model_name_map = antigravity_model_name_map(command_model_configs);
    let mut windows = Vec::new();
    collect_antigravity_quota_windows_from_paths(
        user_status,
        &[
            "/userStatus/cascadeModelConfigData/clientModelConfigs",
            "/cascadeModelConfigData/clientModelConfigs",
            "/clientModelConfigs",
        ],
        &model_name_map,
        &mut windows,
    );
    collect_antigravity_quota_windows_from_paths(
        command_model_configs,
        &["/clientModelConfigs", "/configs"],
        &model_name_map,
        &mut windows,
    );

    let mut deduped = HashMap::<String, AntigravityQuotaWindow>::new();
    let mut label_order = Vec::<String>::new();
    for window in windows {
        if !deduped.contains_key(&window.label) {
            label_order.push(window.label.clone());
        }
        deduped
            .entry(window.label.clone())
            .and_modify(|existing| {
                if window.remaining_fraction < existing.remaining_fraction {
                    *existing = window.clone();
                }
            })
            .or_insert(window);
    }

    let windows = label_order
        .into_iter()
        .filter_map(|label| deduped.remove(&label))
        .collect::<Vec<_>>();

    let mut model_ids = windows
        .iter()
        .filter_map(|window| window.model_id.clone())
        .collect::<Vec<_>>();
    model_ids.extend(model_name_map.keys().cloned());
    model_ids.sort();
    model_ids.dedup();

    let mut selected_model_label = None;
    let windows = if let Some(selected_model_id) =
        antigravity_selected_model_id(user_status, command_model_configs, &model_ids)
    {
        let selected_model_label_hint = model_name_map
            .get(&selected_model_id)
            .map(|label| antigravity_strip_parenthetical_suffix(label));
        let selected_windows = windows
            .iter()
            .filter(|window| {
                window.model_id.as_deref() == Some(selected_model_id.as_str())
                    || selected_model_label_hint
                        .as_deref()
                        .is_some_and(|label| window.label == label)
            })
            .cloned()
            .collect::<Vec<_>>();

        if !selected_windows.is_empty() {
            selected_model_label = selected_windows.first().map(|window| window.label.clone());
            selected_windows
        } else {
            windows
        }
    } else {
        windows
    };

    let selected_window = if selected_model_label.is_none() {
        match windows.first().cloned() {
            Some(window) => {
                selected_model_label = Some(window.label.clone());
                Some(window)
            }
            None => None,
        }
    } else {
        windows.first().cloned()
    };

    let summary = selected_window.as_ref().map(antigravity_format_quota_value);

    Some(AntigravityUsageSnapshot {
        summary,
        selected_model_label,
    })
}

fn load_antigravity_usage_snapshot() -> Option<AntigravityUsageSnapshot> {
    let cache_path = antigravity_usage_cache_path();
    // Antigravity model selection can change out of band while Kaku is open,
    // so prefer a live local fetch and only fall back to cache when it fails.
    if let Some(live) = fetch_antigravity_usage_json() {
        write_json_cache(&cache_path, &live);
        if let Some(snapshot) = parse_antigravity_usage_snapshot(&live) {
            return Some(snapshot);
        }
    }

    if cache_path.exists() && usage_cache_is_fresh(&cache_path) {
        if let Some(cached) = load_usage_json_from_cache(&cache_path, "antigravity usage cache")
            .and_then(|value| parse_antigravity_usage_snapshot(&value))
        {
            return Some(cached);
        }
    }

    Some(antigravity_fallback_usage_snapshot())
}

fn antigravity_fallback_usage_snapshot() -> AntigravityUsageSnapshot {
    AntigravityUsageSnapshot {
        summary: None,
        selected_model_label: antigravity_fallback_selected_model_label(),
    }
}

fn load_cached_antigravity_usage_snapshot() -> AntigravityUsageSnapshot {
    let cache_path = antigravity_usage_cache_path();
    if cache_path.exists() && usage_cache_is_fresh(&cache_path) {
        if let Some(cached) = load_usage_json_from_cache(&cache_path, "antigravity usage cache")
            .and_then(|value| parse_antigravity_usage_snapshot(&value))
        {
            return cached;
        }
    }

    antigravity_fallback_usage_snapshot()
}

fn format_duration_short(total_seconds: i64) -> Option<String> {
    if total_seconds <= 0 {
        return None;
    }

    let days = total_seconds / 86_400;
    let hours = (total_seconds % 86_400) / 3_600;
    let minutes = (total_seconds % 3_600) / 60;

    if days > 0 {
        Some(format!("{days}d{hours}h"))
    } else if hours > 0 {
        Some(format!("{hours}h{minutes}m"))
    } else {
        Some(format!("{minutes}m"))
    }
}

fn format_reset_time_from_epoch(reset_at: i64) -> Option<String> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs() as i64;
    format_duration_short(reset_at - now)
}

fn format_reset_time_from_iso(reset_at: &str) -> Option<String> {
    let reset_at = DateTime::parse_from_rfc3339(reset_at).ok()?;
    let reset_at = reset_at.with_timezone(&Utc);
    format_duration_short((reset_at - Utc::now()).num_seconds())
}

fn supports_remote_usage(tool: Tool) -> bool {
    matches!(
        tool,
        Tool::ClaudeCode | Tool::Codex | Tool::Kimi | Tool::Antigravity | Tool::Copilot
    )
}

fn load_usage_update(tool: Tool) -> UsageSummaryUpdate {
    match tool {
        Tool::Antigravity => {
            let snapshot = load_antigravity_usage_snapshot();
            UsageSummaryUpdate {
                tool,
                summary: snapshot
                    .as_ref()
                    .and_then(|snapshot| snapshot.summary.clone()),
                fields: Some(extract_antigravity_fields(snapshot.as_ref())),
            }
        }
        Tool::ClaudeCode => UsageSummaryUpdate {
            tool,
            summary: load_claude_usage_snapshot().and_then(|snapshot| snapshot.summary),
            fields: None,
        },
        Tool::Codex => UsageSummaryUpdate {
            tool,
            summary: load_codex_usage_snapshot().and_then(|snapshot| snapshot.summary),
            fields: None,
        },
        Tool::Kimi => UsageSummaryUpdate {
            tool,
            summary: load_kimi_usage_snapshot().and_then(|snapshot| snapshot.summary),
            fields: None,
        },
        Tool::Copilot => UsageSummaryUpdate {
            tool,
            summary: load_copilot_usage_snapshot().and_then(|snapshot| snapshot.summary),
            fields: None,
        },
        _ => UsageSummaryUpdate {
            tool,
            summary: None,
            fields: None,
        },
    }
}

fn format_percent_value(percent: f64) -> String {
    if (percent.fract()).abs() < 0.05 {
        format!("{percent:.0}%")
    } else {
        format!("{percent:.1}%")
    }
}

fn format_remaining_percent_value(used_percent: f64) -> String {
    format_percent_value((100.0 - used_percent).clamp(0.0, 100.0))
}

fn format_remaining_window_value(
    label: &str,
    used_percent: f64,
    reset_in: Option<String>,
) -> String {
    let mut value = format!(
        "{label} remain {}",
        format_remaining_percent_value(used_percent)
    );
    if let Some(reset_in) = reset_in {
        value.push_str(" · reset ");
        value.push_str(&reset_in);
    }
    value
}

fn format_remaining_count_value(value: f64) -> String {
    if (value.fract()).abs() < 0.05 {
        format!("{value:.0}")
    } else {
        format!("{value:.1}")
    }
}

fn format_codex_usage_value(label: &str, window: &serde_json::Value) -> Option<String> {
    let used_percent = window.get("used_percent")?.as_f64()?;
    let reset_in = window
        .get("reset_at")
        .and_then(|value| value.as_i64())
        .and_then(format_reset_time_from_epoch);
    Some(format_remaining_window_value(label, used_percent, reset_in))
}

fn parse_codex_usage_snapshot(data: &serde_json::Value) -> Option<CodexUsageSnapshot> {
    let rate_limit = data.get("rate_limit")?;
    let current_value = rate_limit
        .get("primary_window")
        .and_then(|window| format_codex_usage_value("5h", window));
    let weekly_value = rate_limit
        .get("secondary_window")
        .and_then(|window| format_codex_usage_value("7d", window));

    let summary = match (current_value, weekly_value) {
        (Some(current), Some(weekly)) => Some(format!("{current}  |  {weekly}")),
        (Some(current), None) => Some(current),
        (None, Some(weekly)) => Some(weekly),
        (None, None) => None,
    };

    summary.as_ref()?;
    Some(CodexUsageSnapshot { summary })
}

fn usage_cache_is_fresh(path: &Path) -> bool {
    path.metadata()
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|modified| modified.elapsed().ok())
        .is_some_and(|elapsed| elapsed < USAGE_CACHE_TTL)
}

fn write_json_cache(path: &Path, value: &serde_json::Value) {
    if let Some(parent) = path.parent() {
        if let Err(err) = config::create_user_owned_dirs(parent) {
            log::debug!("failed to create cache dir {}: {}", parent.display(), err);
            return;
        }
    }

    match serde_json::to_vec(value) {
        Ok(bytes) => {
            if let Err(err) = write_atomic(path, &bytes) {
                log::debug!("failed to write {}: {}", path.display(), err);
            }
        }
        Err(err) => log::debug!("failed to serialize {}: {}", path.display(), err),
    }
}

fn run_curl(args: &[&str]) -> Option<serde_json::Value> {
    // Request headers are passed via argv for portability, which means short-lived
    // tokens may be visible to local process inspectors such as `ps` while curl runs.
    let output = std::process::Command::new(OsStr::new("/usr/bin/curl"))
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log::debug!(
            "curl failed with status {}: {}",
            output.status,
            stderr.trim()
        );
        return None;
    }

    let raw = String::from_utf8(output.stdout)
        .map_err(|err| log::debug!("curl returned non-utf8 stdout: {}", err))
        .ok()?;
    serde_json::from_str(&raw)
        .map_err(|err| log::debug!("failed to parse curl json: {}", err))
        .ok()
}

fn fetch_codex_usage_json() -> Option<serde_json::Value> {
    let cache_path = codex_usage_cache_path();
    if cache_path.exists() && usage_cache_is_fresh(&cache_path) {
        return load_usage_json_from_cache(&cache_path, "codex usage cache");
    }

    let (access_token, account_id) = read_codex_auth_info()?;
    let live = run_curl(&[
        "-sS",
        "--max-time",
        "3",
        "-H",
        &format!("Authorization: Bearer {access_token}"),
        "-H",
        &format!("ChatGPT-Account-Id: {account_id}"),
        "-H",
        "Accept: application/json",
        "https://chatgpt.com/backend-api/wham/usage",
    ]);

    if let Some(value) = live {
        write_json_cache(&cache_path, &value);
        return Some(value);
    }

    load_usage_json_from_cache(&cache_path, "codex usage cache")
}

fn load_codex_usage_snapshot() -> Option<CodexUsageSnapshot> {
    let data = fetch_codex_usage_json()?;
    parse_codex_usage_snapshot(&data)
}

fn load_usage_json_from_cache(path: &Path, context: &str) -> Option<serde_json::Value> {
    read_json_file_with_debug(path, context)
}

fn fetch_usage_json_with_cache<F>(
    path: PathBuf,
    context: &str,
    fetcher: F,
) -> Option<serde_json::Value>
where
    F: FnOnce() -> Option<serde_json::Value>,
{
    if path.exists() && usage_cache_is_fresh(&path) {
        return load_usage_json_from_cache(&path, context);
    }

    if let Some(value) = fetcher() {
        write_json_cache(&path, &value);
        return Some(value);
    }

    load_usage_json_from_cache(&path, context)
}

fn read_claude_oauth_credentials() -> Option<serde_json::Value> {
    let output = std::process::Command::new("/usr/bin/security")
        .args([
            "find-generic-password",
            "-s",
            "Claude Code-credentials",
            "-w",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log::debug!(
            "claude keychain probe failed with status {}: {}",
            output.status,
            stderr.trim()
        );
        return None;
    }

    let raw = String::from_utf8(output.stdout)
        .map_err(|err| log::debug!("claude keychain probe returned non-utf8 stdout: {}", err))
        .ok()?;
    serde_json::from_str::<serde_json::Value>(raw.trim())
        .map_err(|err| log::debug!("failed to parse claude keychain json: {}", err))
        .ok()
}

fn read_claude_oauth_access_token() -> Option<String> {
    let parsed = read_claude_oauth_credentials()?;

    parsed
        .get("claudeAiOauth")
        .and_then(|value| value.get("accessToken"))
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())
}

fn read_claude_oauth_refresh_token() -> Option<String> {
    let parsed = read_claude_oauth_credentials()?;
    parsed
        .get("claudeAiOauth")
        .and_then(|value| value.get("refreshToken"))
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())
}

fn parse_claude_keychain_account(raw: &str) -> Option<String> {
    let marker = "\"acct\"<blob>=\"";
    raw.lines().find_map(|line| {
        let line = line.trim();
        let start = line.find(marker)? + marker.len();
        let end = line[start..].find('"')?;
        Some(line[start..start + end].to_string())
    })
}

fn read_claude_keychain_account() -> Option<String> {
    let output = std::process::Command::new("/usr/bin/security")
        .args(["find-generic-password", "-s", "Claude Code-credentials"])
        .output()
        .ok()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log::debug!(
            "claude keychain account probe failed with status {}: {}",
            output.status,
            stderr.trim()
        );
        return None;
    }

    let raw = String::from_utf8(output.stdout)
        .map_err(|err| log::debug!("claude keychain account probe returned non-utf8: {}", err))
        .ok()?;
    parse_claude_keychain_account(&raw)
}

fn write_claude_oauth_credentials(credentials: &serde_json::Value) -> Option<()> {
    let account = read_claude_keychain_account()?;
    let secret = serde_json::to_string(credentials)
        .map_err(|err| log::debug!("failed to serialize claude keychain json: {}", err))
        .ok()?;

    let output = std::process::Command::new("/usr/bin/security")
        .args([
            "add-generic-password",
            "-U",
            "-a",
            &account,
            "-s",
            "Claude Code-credentials",
            "-w",
            &secret,
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log::warn!(
            "failed to update claude keychain credentials: status {}: {}",
            output.status,
            stderr.trim()
        );
        return None;
    }

    Some(())
}

fn refresh_claude_oauth_access_token() -> Option<String> {
    let current_credentials = read_claude_oauth_credentials()?;
    let refresh_token = read_claude_oauth_refresh_token()?;
    let refreshed = run_curl(&[
        "-sS",
        "--max-time",
        "5",
        "-X",
        "POST",
        CLAUDE_OAUTH_TOKEN_URL,
        "-H",
        "Content-Type: application/x-www-form-urlencoded",
        "--data-urlencode",
        "grant_type=refresh_token",
        "--data-urlencode",
        &format!("refresh_token={refresh_token}"),
        "--data-urlencode",
        &format!("client_id={CLAUDE_OAUTH_CLIENT_ID}"),
    ])?;

    let access_token = refreshed
        .get("access_token")
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())?;
    let rotated_refresh_token = refreshed
        .get("refresh_token")
        .and_then(|value| value.as_str())
        .unwrap_or(&refresh_token)
        .to_string();

    let mut updated_credentials = current_credentials;
    let oauth = updated_credentials
        .get_mut("claudeAiOauth")
        .and_then(|value| value.as_object_mut())?;
    oauth.insert(
        "accessToken".into(),
        serde_json::Value::String(access_token.clone()),
    );
    oauth.insert(
        "refreshToken".into(),
        serde_json::Value::String(rotated_refresh_token),
    );

    if let Some(expires_in) = refreshed.get("expires_in").and_then(|value| value.as_i64()) {
        let expires_at_ms = Utc::now().timestamp_millis() + expires_in * 1000;
        oauth.insert(
            "expiresAt".into(),
            serde_json::Value::Number(expires_at_ms.into()),
        );
    }

    if let Some(scope) = refreshed.get("scope").and_then(|value| value.as_str()) {
        let scopes = scope
            .split_whitespace()
            .map(|item| serde_json::Value::String(item.to_string()))
            .collect::<Vec<_>>();
        oauth.insert("scopes".into(), serde_json::Value::Array(scopes));
    }

    let _ = write_claude_oauth_credentials(&updated_credentials);
    Some(access_token)
}

fn fetch_claude_usage_with_access_token(access_token: &str) -> Option<serde_json::Value> {
    run_curl(&[
        "-sS",
        "--max-time",
        "3",
        "-H",
        &format!("Authorization: Bearer {access_token}"),
        "-H",
        "anthropic-beta: oauth-2025-04-20",
        "-H",
        "Accept: application/json",
        "-H",
        "Content-Type: application/json",
        "-H",
        "User-Agent: claude-code/2.0.27",
        "https://api.anthropic.com/api/oauth/usage",
    ])
}

fn fetch_claude_usage_json() -> Option<serde_json::Value> {
    let cache_path = claude_usage_cache_path();
    if cache_path.exists() && usage_cache_is_fresh(&cache_path) {
        if let Some(cached) = load_usage_json_from_cache(&cache_path, "claude usage cache") {
            if parse_claude_usage_error(&cached).is_none() {
                return Some(cached);
            }
        }
    }

    let access_token = read_claude_oauth_access_token()?;
    let live = fetch_claude_usage_with_access_token(&access_token)
        .filter(|value| parse_claude_usage_error(value).is_none())
        .or_else(|| {
            let refreshed = refresh_claude_oauth_access_token()?;
            fetch_claude_usage_with_access_token(&refreshed)
        });

    if let Some(value) = live {
        if parse_claude_usage_error(&value).is_none() {
            write_json_cache(&cache_path, &value);
        }
        return Some(value);
    }

    load_usage_json_from_cache(&cache_path, "claude usage cache")
}

fn parse_claude_usage_error(data: &serde_json::Value) -> Option<String> {
    let error = data.get("error")?;
    let error_type = error.get("type").and_then(|value| value.as_str());
    let error_code = error
        .get("details")
        .and_then(|value| value.get("error_code"))
        .and_then(|value| value.as_str());

    if matches!(error_type, Some("authentication_error"))
        || matches!(error_code, Some("token_expired" | "invalid_token"))
    {
        return Some("Re-auth required".into());
    }

    None
}

fn parse_claude_usage_snapshot(data: &serde_json::Value) -> Option<ClaudeUsageSnapshot> {
    if let Some(summary) = parse_claude_usage_error(data) {
        return Some(ClaudeUsageSnapshot {
            summary: Some(summary),
        });
    }

    let current_value = data.get("five_hour").and_then(|window| {
        let used_percent = window.get("utilization")?.as_f64()?;
        let reset_in = window
            .get("resets_at")
            .and_then(|value| value.as_str())
            .and_then(format_reset_time_from_iso);
        Some(format_remaining_window_value("5h", used_percent, reset_in))
    });
    let weekly_value = data.get("seven_day").and_then(|window| {
        let used_percent = window.get("utilization")?.as_f64()?;
        let reset_in = window
            .get("resets_at")
            .and_then(|value| value.as_str())
            .and_then(format_reset_time_from_iso);
        Some(format_remaining_window_value("7d", used_percent, reset_in))
    });

    let summary = match (current_value, weekly_value) {
        (Some(current), Some(weekly)) => Some(format!("{current}  |  {weekly}")),
        (Some(current), None) => Some(current),
        (None, Some(weekly)) => Some(weekly),
        (None, None) => None,
    };

    summary.as_ref()?;
    Some(ClaudeUsageSnapshot { summary })
}

fn load_claude_usage_snapshot() -> Option<ClaudeUsageSnapshot> {
    let data = fetch_claude_usage_json()?;
    parse_claude_usage_snapshot(&data)
}

fn read_kimi_oauth_credentials() -> Option<serde_json::Value> {
    read_json_file_with_debug(&kimi_credentials_path(), "kimi credentials")
}

fn write_kimi_oauth_credentials(credentials: &serde_json::Value) -> Option<()> {
    let path = kimi_credentials_path();
    let bytes = serde_json::to_vec(credentials)
        .map_err(|err| log::debug!("failed to serialize kimi credentials: {}", err))
        .ok()?;
    write_atomic(&path, &bytes)
        .map_err(|err| log::debug!("failed to write kimi credentials: {}", err))
        .ok()?;
    Some(())
}

fn read_kimi_access_token() -> Option<String> {
    read_kimi_oauth_credentials()?
        .get("access_token")
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())
}

fn read_kimi_refresh_token() -> Option<String> {
    read_kimi_oauth_credentials()?
        .get("refresh_token")
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())
}

fn kimi_access_token_needs_refresh() -> bool {
    let Some(credentials) = read_kimi_oauth_credentials() else {
        return false;
    };
    let expires_at = credentials
        .get("expires_at")
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0);
    expires_at <= now + 300.0
}

fn refresh_kimi_access_token() -> Option<String> {
    let current_credentials = read_kimi_oauth_credentials()?;
    let refresh_token = read_kimi_refresh_token()?;
    let refreshed = run_curl(&[
        "-sS",
        "--max-time",
        "5",
        "-X",
        "POST",
        KIMI_OAUTH_TOKEN_URL,
        "--data-urlencode",
        &format!("client_id={KIMI_OAUTH_CLIENT_ID}"),
        "--data-urlencode",
        "grant_type=refresh_token",
        "--data-urlencode",
        &format!("refresh_token={refresh_token}"),
    ])?;

    let access_token = refreshed
        .get("access_token")
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())?;
    let rotated_refresh_token = refreshed
        .get("refresh_token")
        .and_then(|value| value.as_str())
        .unwrap_or(&refresh_token)
        .to_string();

    let mut updated_credentials = current_credentials;
    let object = updated_credentials.as_object_mut()?;
    object.insert(
        "access_token".into(),
        serde_json::Value::String(access_token.clone()),
    );
    object.insert(
        "refresh_token".into(),
        serde_json::Value::String(rotated_refresh_token),
    );
    if let Some(expires_in) = refreshed.get("expires_in").and_then(|value| value.as_f64()) {
        let expires_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .map(|duration| duration.as_secs_f64() + expires_in)?;
        object.insert("expires_at".into(), serde_json::json!(expires_at));
    }
    if let Some(scope) = refreshed.get("scope").and_then(|value| value.as_str()) {
        object.insert("scope".into(), serde_json::Value::String(scope.to_string()));
    }
    if let Some(token_type) = refreshed.get("token_type").and_then(|value| value.as_str()) {
        object.insert(
            "token_type".into(),
            serde_json::Value::String(token_type.to_string()),
        );
    }

    let _ = write_kimi_oauth_credentials(&updated_credentials);
    Some(access_token)
}

fn read_kimi_base_url() -> String {
    let raw = std::fs::read_to_string(Tool::Kimi.config_path()).ok();
    raw.and_then(|raw| raw.parse::<toml::Value>().ok())
        .and_then(|parsed| {
            parsed
                .get("providers")
                .and_then(|value| value.get("managed:kimi-code"))
                .and_then(|value| value.get("base_url"))
                .and_then(|value| value.as_str())
                .map(|value| value.to_string())
        })
        .unwrap_or_else(|| KIMI_DEFAULT_BASE_URL.to_string())
}

fn fetch_kimi_usage_with_access_token(access_token: &str) -> Option<serde_json::Value> {
    let usage_url = format!("{}/usages", read_kimi_base_url().trim_end_matches('/'));
    run_curl(&[
        "-sS",
        "--max-time",
        "3",
        "-H",
        &format!("Authorization: Bearer {access_token}"),
        usage_url.as_str(),
    ])
}

fn fetch_kimi_usage_json() -> Option<serde_json::Value> {
    let cache_path = kimi_usage_cache_path();
    if cache_path.exists() && usage_cache_is_fresh(&cache_path) {
        if let Some(cached) = load_usage_json_from_cache(&cache_path, "kimi usage cache") {
            if cached.get("usage").is_some() {
                return Some(cached);
            }
        }
    }

    let mut access_token = read_kimi_access_token()?;
    let mut refreshed = false;
    if kimi_access_token_needs_refresh() {
        access_token = refresh_kimi_access_token()?;
        refreshed = true;
    }

    let live = if let Some(value) = fetch_kimi_usage_with_access_token(&access_token)
        .and_then(|value| value.get("usage").is_some().then_some(value))
    {
        Some(value)
    } else if refreshed {
        None
    } else {
        let retried = {
            let refreshed = refresh_kimi_access_token()?;
            fetch_kimi_usage_with_access_token(&refreshed)
        };
        retried.and_then(|value| value.get("usage").is_some().then_some(value))
    };

    if let Some(value) = live {
        write_json_cache(&cache_path, &value);
        return Some(value);
    }

    load_usage_json_from_cache(&cache_path, "kimi usage cache")
}

fn kimi_limit_label(item: &serde_json::Value, idx: usize) -> String {
    let duration = item
        .get("window")
        .and_then(|value| value.get("duration"))
        .and_then(|value| value.as_i64())
        .or_else(|| item.get("duration").and_then(|value| value.as_i64()));
    let time_unit = item
        .get("window")
        .and_then(|value| value.get("timeUnit"))
        .and_then(|value| value.as_str())
        .or_else(|| item.get("timeUnit").and_then(|value| value.as_str()))
        .unwrap_or("");

    match (duration, time_unit) {
        (Some(duration), unit) if unit.contains("MINUTE") => {
            if duration >= 60 && duration % 60 == 0 {
                format!("{}h", duration / 60)
            } else {
                format!("{duration}m")
            }
        }
        (Some(duration), unit) if unit.contains("HOUR") => format!("{duration}h"),
        (Some(duration), unit) if unit.contains("DAY") => format!("{duration}d"),
        _ => format!("Limit #{}", idx + 1),
    }
}

fn format_kimi_usage_value(label: &str, detail: &serde_json::Value) -> Option<String> {
    let limit = detail
        .get("limit")
        .and_then(|value| value.as_str().and_then(|value| value.parse::<f64>().ok()))
        .or_else(|| detail.get("limit").and_then(|value| value.as_f64()))?;
    let used = detail
        .get("used")
        .and_then(|value| value.as_str().and_then(|value| value.parse::<f64>().ok()))
        .or_else(|| detail.get("used").and_then(|value| value.as_f64()))
        .or_else(|| {
            let remaining = detail
                .get("remaining")
                .and_then(|value| value.as_str().and_then(|value| value.parse::<f64>().ok()))
                .or_else(|| detail.get("remaining").and_then(|value| value.as_f64()))?;
            Some(limit - remaining)
        })?;
    let reset_in = detail
        .get("resetTime")
        .or_else(|| detail.get("reset_at"))
        .or_else(|| detail.get("resetAt"))
        .and_then(|value| value.as_str())
        .and_then(format_reset_time_from_iso);
    let used_percent = if limit > 0.0 {
        (used / limit) * 100.0
    } else {
        100.0
    };
    Some(format_remaining_window_value(label, used_percent, reset_in))
}

fn parse_kimi_usage_snapshot(data: &serde_json::Value) -> Option<KimiUsageSnapshot> {
    let weekly_value = data
        .get("usage")
        .and_then(|usage| format_kimi_usage_value("7d", usage));
    let current_value = data
        .get("limits")
        .and_then(|value| value.as_array())
        .and_then(|limits| {
            limits.iter().enumerate().find_map(|(idx, item)| {
                let label = kimi_limit_label(item, idx);
                let detail = item.get("detail").unwrap_or(item);
                if label == "5h" || label == "300m" {
                    format_kimi_usage_value("5h", detail)
                } else {
                    format_kimi_usage_value(&label, detail)
                }
            })
        });

    let summary = match (current_value, weekly_value) {
        (Some(current), Some(weekly)) => Some(format!("{current}  |  {weekly}")),
        (Some(current), None) => Some(current),
        (None, Some(weekly)) => Some(weekly),
        (None, None) => None,
    };

    summary.as_ref()?;
    Some(KimiUsageSnapshot { summary })
}

fn load_kimi_usage_snapshot() -> Option<KimiUsageSnapshot> {
    let data = fetch_kimi_usage_json()?;
    parse_kimi_usage_snapshot(&data)
}

fn read_gh_auth_token() -> Option<String> {
    let output = std::process::Command::new("gh")
        .args(["auth", "token"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .ok()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log::debug!(
            "gh auth token probe failed with status {}: {}",
            output.status,
            stderr.trim()
        );
        return None;
    }

    String::from_utf8(output.stdout)
        .map_err(|err| log::debug!("gh auth token probe returned non-utf8 stdout: {}", err))
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn fetch_copilot_usage_json() -> Option<serde_json::Value> {
    let cache_path = copilot_usage_cache_path();
    fetch_usage_json_with_cache(cache_path, "copilot usage cache", || {
        let token = read_gh_auth_token()?;
        run_curl(&[
            "-sS",
            "--max-time",
            "3",
            "-H",
            &format!("Authorization: token {token}"),
            "-H",
            "Accept: application/json",
            "-H",
            "Editor-Version: vscode/1.96.2",
            "-H",
            "Editor-Plugin-Version: copilot-chat/0.26.7",
            "-H",
            "User-Agent: GitHubCopilotChat/0.26.7",
            "-H",
            "X-Github-Api-Version: 2025-04-01",
            "https://api.github.com/copilot_internal/user",
        ])
    })
}

fn parse_copilot_usage_snapshot(data: &serde_json::Value) -> Option<CopilotUsageSnapshot> {
    let premium = data
        .get("quota_snapshots")
        .and_then(|value| value.get("premium_interactions"))?;
    let remaining = premium
        .get("remaining")
        .and_then(|value| value.as_f64())
        .or_else(|| {
            premium
                .get("remaining")
                .and_then(|value| value.as_str()?.parse::<f64>().ok())
        })
        .or_else(|| {
            premium
                .get("quota_remaining")
                .and_then(|value| value.as_f64())
        })?;
    let reset_in = data
        .get("quota_reset_date_utc")
        .and_then(|value| value.as_str())
        .and_then(format_reset_time_from_iso);

    let mut summary = format!(
        "{} left this month",
        format_remaining_count_value(remaining)
    );
    if let Some(reset_in) = reset_in {
        summary.push_str(" · reset ");
        summary.push_str(&reset_in);
    }

    Some(CopilotUsageSnapshot {
        summary: Some(summary),
    })
}

fn load_copilot_usage_snapshot() -> Option<CopilotUsageSnapshot> {
    let data = fetch_copilot_usage_json()?;
    parse_copilot_usage_snapshot(&data)
}

fn gemini_quota_summary(data: &serde_json::Value) -> Option<String> {
    let auth_type = data
        .get("security")
        .and_then(|security| security.get("auth"))
        .and_then(|auth| auth.get("selectedType"))
        .and_then(|value| value.as_str())?;

    // Gemini CLI doesn't expose a stable local "remaining quota" endpoint yet,
    // so these are approximate plan limits used for quick at-a-glance guidance.
    match auth_type {
        "oauth-personal" => Some("Quota 1000/day · 60/min".into()),
        "gemini-api-key" | "api-key" => Some("Quota 250/day · 10/min".into()),
        "workspace-standard" => Some("Quota 1500/day · 120/min".into()),
        "workspace-enterprise" => Some("Quota 2000/day · 120/min".into()),
        _ if auth_type.contains("workspace") => Some("Quota via workspace plan".into()),
        _ if auth_type.contains("vertex") => Some("Quota via Vertex AI".into()),
        _ => None,
    }
}

fn parse_json_with_debug(raw: &str, tool_label: &str) -> serde_json::Value {
    serde_json::from_str(raw).unwrap_or_else(|e| {
        log::warn!("failed to parse {} config json: {}", tool_label, e);
        push_ui_error(format!("{tool_label} config is invalid JSON"));
        serde_json::Value::Null
    })
}

fn parse_json_or_jsonc_with_debug(raw: &str, tool_label: &str) -> serde_json::Value {
    parse_json_or_jsonc(raw).unwrap_or_else(|e| {
        log::warn!("failed to parse {} config json/jsonc: {}", tool_label, e);
        push_ui_error(format!("{tool_label} config is malformed"));
        serde_json::Value::Null
    })
}

fn read_text_with_debug(path: &Path, context: &str) -> Option<String> {
    match std::fs::read_to_string(path) {
        Ok(raw) => Some(raw),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            log::debug!("{context}: {} not found", path.display());
            None
        }
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::PermissionDenied | io::ErrorKind::InvalidData
            ) =>
        {
            log::warn!("{context}: failed to read {}: {}", path.display(), e);
            push_ui_error(format!(
                "{context}: cannot read {}. Check file permission or encoding.",
                path.display()
            ));
            None
        }
        Err(e) => {
            log::debug!("{context}: failed to read {}: {}", path.display(), e);
            None
        }
    }
}

fn parse_json_value_with_debug(raw: &str, context: &str) -> Option<serde_json::Value> {
    serde_json::from_str(raw)
        .map_err(|e| {
            log::warn!("{context}: failed to parse json: {}", e);
            push_ui_error(format!("{context}: invalid JSON format"));
        })
        .ok()
}

fn read_json_file_with_debug(path: &Path, context: &str) -> Option<serde_json::Value> {
    let raw = read_text_with_debug(path, context)?;
    parse_json_value_with_debug(&raw, context)
}

fn decode_jwt_payload_with_debug(token: &str, context: &str) -> Option<serde_json::Value> {
    // JWT format: header.payload.signature
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() < 2 {
        log::debug!("{context}: invalid jwt format");
        push_ui_error(format!("{context}: invalid token format"));
        return None;
    }

    let mut payload = parts[1].to_string();
    while payload.len() % 4 != 0 {
        payload.push('=');
    }

    use base64::Engine;
    let decoded = base64::engine::general_purpose::URL_SAFE
        .decode(&payload)
        .map_err(|e| {
            log::debug!("{context}: failed to decode jwt payload: {}", e);
            push_ui_error(format!("{context}: token payload decode failed"));
        })
        .ok()?;
    serde_json::from_slice(&decoded)
        .map_err(|e| {
            log::debug!("{context}: failed to parse jwt payload json: {}", e);
            push_ui_error(format!("{context}: token payload JSON is invalid"));
        })
        .ok()
}

fn json_str(val: &serde_json::Value, key: &str) -> String {
    val.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn json_nested_or_top_str(
    val: &serde_json::Value,
    parent_key: &str,
    nested_key: &str,
    legacy_top_key: &str,
) -> String {
    val.get(parent_key)
        .and_then(|v| v.as_object())
        .and_then(|obj| obj.get(nested_key))
        .and_then(|v| v.as_str())
        .or_else(|| val.get(legacy_top_key).and_then(|v| v.as_str()))
        .unwrap_or("")
        .to_string()
}

fn string_options(values: &[&str]) -> Vec<String> {
    values.iter().map(|v| (*v).to_string()).collect()
}

fn mask_key(val: &str) -> String {
    if val.is_empty() {
        return "—".into();
    }
    if val.len() <= 12 {
        return "****".into();
    }
    // Show first 12 chars and last 4 chars
    format!("{}...{}", &val[..12], &val[val.len() - 4..])
}

fn normalize_custom_header(value: &str) -> Option<String> {
    let raw = value.trim();
    if raw.is_empty() {
        return None;
    }
    let (name, header_value) = raw.split_once(':')?;
    let name = name.trim();
    let header_value = header_value.trim();
    if name.is_empty() || header_value.is_empty() {
        return None;
    }
    if name.eq_ignore_ascii_case("authorization") || name.eq_ignore_ascii_case("content-type") {
        return None;
    }
    Some(format!("{name}: {header_value}"))
}

fn normalize_custom_headers(values: Vec<String>) -> Vec<String> {
    let mut dedup = HashSet::new();
    values
        .into_iter()
        .filter_map(|item| normalize_custom_header(&item))
        .filter(|header| {
            let key = header
                .split_once(':')
                .map(|(name, _)| name)
                .unwrap_or("")
                .trim()
                .to_ascii_lowercase();
            dedup.insert(key)
        })
        .collect()
}

fn parse_custom_headers_toml(value: Option<&toml::Value>) -> Vec<String> {
    match value {
        Some(toml::Value::Array(items)) => normalize_custom_headers(
            items
                .iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect(),
        ),
        Some(toml::Value::String(raw)) => normalize_custom_headers(
            raw.split(',')
                .map(|part| part.trim().to_string())
                .filter(|part| !part.is_empty())
                .collect(),
        ),
        _ => vec![],
    }
}

/// Configuration for the Kaku built-in AI assistant.
///
/// This struct holds the configuration for Kaku's AI-powered command analysis
/// feature. It ensures that model and base_url always have valid values
/// by falling back to defaults when empty strings are provided.
#[derive(Debug, Clone)]
struct KakuAssistantConfig {
    /// Whether the AI assistant is enabled
    enabled: bool,
    /// API key for the AI service (may be empty if not configured)
    api_key: String,
    /// Model identifier (never empty, falls back to default)
    model: String,
    /// Base URL for the API endpoint (never empty, falls back to default)
    base_url: String,
    /// Provider preset name (e.g. "OpenAI", "Custom")
    provider: String,
    /// Optional extra request headers as `Name: Value`
    custom_headers: Vec<String>,
    /// Web search backend: "none" (disabled), "brave", "pipellm", or "tavily".
    web_search_provider: String,
    /// API key for the web search provider (empty = not configured).
    web_search_api_key: String,
}

impl KakuAssistantConfig {
    /// Creates a new configuration with the given values.
    ///
    /// # Arguments
    /// * `enabled` - Whether the assistant is enabled
    /// * `api_key` - API key (empty string if not set)
    /// * `model` - Model identifier (empty strings will be replaced with default)
    /// * `base_url` - Base URL (empty strings will be replaced with default)
    fn new(
        enabled: bool,
        api_key: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        let model = model.into();
        let base_url = base_url.into();
        let resolved_base_url = if base_url.trim().is_empty() {
            assistant_config::DEFAULT_BASE_URL.to_string()
        } else {
            base_url
        };
        let provider = assistant_config::detect_provider(&resolved_base_url).to_string();
        Self {
            enabled,
            api_key: api_key.into(),
            model: if model.trim().is_empty() {
                assistant_config::DEFAULT_MODEL.to_string()
            } else {
                model
            },
            base_url: resolved_base_url,
            provider,
            custom_headers: vec![],
            web_search_provider: "none".to_string(),
            web_search_api_key: String::new(),
        }
    }

    fn with_custom_headers(mut self, custom_headers: Vec<String>) -> Self {
        self.custom_headers = normalize_custom_headers(custom_headers);
        self
    }

    /// Returns whether the assistant is enabled.
    fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Returns the API key (may be empty).
    fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Returns the model identifier (never empty).
    fn model(&self) -> &str {
        &self.model
    }

    /// Returns the base URL (never empty).
    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn provider(&self) -> &str {
        &self.provider
    }

    fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    fn web_search_provider(&self) -> &str {
        &self.web_search_provider
    }

    fn web_search_api_key(&self) -> &str {
        &self.web_search_api_key
    }

    fn with_web_search(mut self, provider: impl Into<String>, api_key: impl Into<String>) -> Self {
        self.web_search_provider = provider.into();
        self.web_search_api_key = api_key.into();
        self
    }

    fn custom_headers(&self) -> &[String] {
        &self.custom_headers
    }
}

impl Default for KakuAssistantConfig {
    fn default() -> Self {
        Self::new(true, String::new(), String::new(), String::new())
    }
}

/// Parses a KakuAssistantConfig from TOML content.
///
/// This function gracefully handles malformed TOML by using default values
/// for any missing or invalid fields.
fn parse_kaku_assistant_config(raw: &str) -> KakuAssistantConfig {
    let parsed = raw.parse::<toml::Value>().unwrap_or_else(|e| {
        log::warn!("failed to parse assistant.toml: {}", e);
        push_ui_error("Kaku Assistant config TOML is malformed");
        toml::Value::Table(Default::default())
    });

    let enabled = parsed
        .get("enabled")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let api_key = parsed.get("api_key").and_then(|v| v.as_str()).unwrap_or("");
    let model = parsed.get("model").and_then(|v| v.as_str()).unwrap_or("");
    let base_url = parsed
        .get("base_url")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let custom_headers = parse_custom_headers_toml(parsed.get("custom_headers"));

    let web_search_provider = parsed
        .get("web_search_provider")
        .and_then(|v| v.as_str())
        .filter(|s| matches!(*s, "brave" | "pipellm" | "tavily"))
        .unwrap_or("none")
        .to_string();

    let web_search_api_key = parsed
        .get("web_search_api_key")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    KakuAssistantConfig::new(enabled, api_key, model, base_url)
        .with_custom_headers(custom_headers)
        .with_web_search(web_search_provider, web_search_api_key)
}

fn get_kaku_assistant_api_key() -> Option<String> {
    let path = assistant_config::ensure_assistant_toml_exists()
        .map_err(|e| log::debug!("assistant config not available: {}", e))
        .ok()?;
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| log::debug!("failed to read assistant config: {}", e))
        .ok()?;
    let cfg = parse_kaku_assistant_config(&raw);
    if cfg.api_key().trim().is_empty() {
        log::debug!("assistant config has no api_key set");
        None
    } else {
        Some(cfg.api_key().to_string())
    }
}

// Delegated to kaku-ai-utils crate to avoid cross-binary drift.

fn parse_assistant_models_cache(path: &Path, base_url: &str) -> Option<Vec<String>> {
    let raw = std::fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    if v.get("base_url").and_then(|s| s.as_str())? != base_url {
        return None;
    }
    let models: Vec<String> = v
        .get("models")
        .and_then(|a| a.as_array())?
        .iter()
        .filter_map(|s| s.as_str().map(String::from))
        .collect();
    if models.is_empty() {
        None
    } else {
        Some(models)
    }
}

fn load_assistant_models_cache(path: &Path, base_url: &str) -> Option<Vec<String>> {
    let elapsed = path
        .metadata()
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.elapsed().ok())?;
    if elapsed >= ASSISTANT_MODELS_CACHE_TTL {
        return None;
    }
    parse_assistant_models_cache(path, base_url)
}

fn load_assistant_models_stale_cache(path: &Path, base_url: &str) -> Vec<String> {
    parse_assistant_models_cache(path, base_url).unwrap_or_default()
}

fn save_assistant_models_cache(path: &Path, base_url: &str, models: &[String]) {
    let value = serde_json::json!({ "base_url": base_url, "models": models });
    write_json_cache(path, &value);
}

/// Fetch available chat models from the assistant API's `/models` endpoint.
///
/// Uses a 3-second curl timeout. Results are cached to disk for 30 minutes
/// per base URL. Returns an empty vec on any failure so the field gracefully
/// falls back to free-text entry.
///
/// The API key is passed via stdin (curl --config -) to avoid exposing it in
/// process arguments visible to ps/audit logs.
fn fetch_kaku_assistant_models(api_key: &str, base_url: &str) -> Vec<String> {
    let api_key = api_key.trim();
    let base_url = base_url.trim().trim_end_matches('/');
    if api_key.is_empty() || base_url.is_empty() {
        return Vec::new();
    }

    let cache_path = assistant_models_cache_path();

    if let Some(cached) = load_assistant_models_cache(&cache_path, base_url) {
        return cached;
    }

    let url = format!("{}/models", base_url);
    // Pass Authorization header via --config - (stdin) to avoid argv exposure.
    let curl_config = format!("header = \"Authorization: Bearer {}\"", api_key);
    let mut child = match std::process::Command::new("/usr/bin/curl")
        .args(["-sS", "--fail", "--max-time", "3", "--config", "-", &url])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            log::debug!("assistant models fetch: curl spawn failed: {}", e);
            return load_assistant_models_stale_cache(&cache_path, base_url);
        }
    };

    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        let _ = stdin.write_all(curl_config.as_bytes());
    }

    let output = match child.wait_with_output() {
        Ok(o) => o,
        Err(e) => {
            log::debug!("assistant models fetch: curl wait failed: {}", e);
            return load_assistant_models_stale_cache(&cache_path, base_url);
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log::debug!(
            "assistant models fetch: curl failed ({}): {}",
            output.status,
            stderr.trim()
        );
        return load_assistant_models_stale_cache(&cache_path, base_url);
    }

    let raw = match String::from_utf8(output.stdout) {
        Ok(s) => s,
        Err(e) => {
            log::debug!("assistant models fetch: non-utf8 response: {}", e);
            return load_assistant_models_stale_cache(&cache_path, base_url);
        }
    };

    let v: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            log::debug!("assistant models fetch: JSON parse failed: {}", e);
            return load_assistant_models_stale_cache(&cache_path, base_url);
        }
    };

    let mut models: Vec<String> = v
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|s| s.as_str()).map(String::from))
                .filter(|id| kaku_ai_utils::is_chat_model_id(id))
                .collect()
        })
        .unwrap_or_default();

    models.sort();
    models.dedup();
    models.truncate(50);

    if !models.is_empty() {
        save_assistant_models_cache(&cache_path, base_url, &models);
    }
    models
}

fn extract_kaku_assistant_fields(raw: &str) -> Vec<FieldEntry> {
    let cfg = parse_kaku_assistant_config(raw);

    // Fetch live model list from the API; falls back to empty (free-text) on failure.
    let model_options = fetch_kaku_assistant_models(cfg.api_key(), cfg.base_url());

    let mut fields = vec![
        FieldEntry {
            key: "Enabled".into(),
            value: if cfg.is_enabled() { "On" } else { "Off" }.into(),
            options: vec!["On".into(), "Off".into()],
            editable: true,
        },
        FieldEntry {
            key: "Provider".into(),
            value: cfg.provider().to_string(),
            options: assistant_config::provider_names(),
            editable: true,
        },
        FieldEntry {
            key: "Model".into(),
            value: cfg.model().to_string(),
            options: model_options,
            editable: true,
        },
        FieldEntry {
            key: "Base URL".into(),
            value: cfg.base_url().to_string(),
            options: vec![],
            editable: true,
        },
        FieldEntry {
            key: "API Key".into(),
            value: mask_key(cfg.api_key()),
            options: vec![],
            editable: true,
        },
        FieldEntry {
            key: "Web Search".into(),
            value: cfg.web_search_provider().to_string(),
            options: vec![
                "none".into(),
                "brave".into(),
                "pipellm".into(),
                "tavily".into(),
            ],
            editable: true,
        },
    ];

    // Show Search Key entry only when a provider is selected.
    if cfg.web_search_provider() != "none" {
        fields.push(FieldEntry {
            key: "Search Key".into(),
            value: mask_key(cfg.web_search_api_key()),
            options: vec![],
            editable: true,
        });
    }

    fields
}

fn render_toml_string(value: &str) -> String {
    toml::Value::String(value.to_string()).to_string()
}

fn write_kaku_assistant_config(path: &Path, cfg: &KakuAssistantConfig) -> anyhow::Result<()> {
    let mut out = String::new();
    out.push_str("# Kaku Assistant configuration\n");
    out.push_str(
        "# enabled: true enables command analysis suggestions; false disables requests.\n",
    );
    out.push_str("# api_key: provider API key, example: \"sk-xxxx\".\n");
    out.push_str("# model: model id, example: \"gpt-5.4-mini\" or \"gpt-4o\".\n");
    out.push_str("# base_url: chat-completions API root URL.\n");
    out.push_str(
        "# custom_headers: optional extra HTTP headers for enterprise proxies or API gateways.\n",
    );
    out.push_str("#                 format: [\"Header-Name: value\", \"Another-Header: value\"]\n");
    out.push_str("#                 note: Authorization and Content-Type are reserved and cannot be overridden.\n\n");
    out.push_str(if cfg.is_enabled() {
        "enabled = true\n"
    } else {
        "enabled = false\n"
    });
    if cfg.api_key().trim().is_empty() {
        out.push_str("# api_key = \"<your_api_key>\"\n");
    } else {
        out.push_str(&format!(
            "api_key = {}\n",
            render_toml_string(cfg.api_key().trim())
        ));
    }
    out.push_str(&format!(
        "model = {}\n",
        render_toml_string(cfg.model().trim())
    ));
    out.push_str(&format!(
        "base_url = {}\n",
        render_toml_string(cfg.base_url().trim())
    ));
    if cfg.custom_headers().is_empty() {
        out.push_str("# custom_headers = [\"X-Customer-ID: your-customer-id\"]\n");
    } else {
        let arr = toml::Value::Array(
            cfg.custom_headers()
                .iter()
                .map(|item| toml::Value::String(item.clone()))
                .collect(),
        );
        out.push_str(&format!("custom_headers = {}\n", arr));
    }
    // Web search: comment out when not configured to keep the file clean.
    let provider = cfg.web_search_provider();
    if provider == "none" || provider.is_empty() {
        out.push_str("# web_search_provider = \"brave\"  # or pipellm, tavily\n");
        out.push_str("# web_search_api_key = \"...\"\n");
    } else {
        out.push_str(&format!(
            "web_search_provider = {}\n",
            render_toml_string(provider)
        ));
        if cfg.web_search_api_key().trim().is_empty() {
            out.push_str("# web_search_api_key = \"...\"\n");
        } else {
            out.push_str(&format!(
                "web_search_api_key = {}\n",
                render_toml_string(cfg.web_search_api_key().trim())
            ));
        }
    }
    write_atomic(path, out.as_bytes()).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn save_kaku_assistant_field(field_key: &str, new_val: &str) -> anyhow::Result<()> {
    let path = assistant_config::ensure_assistant_toml_exists()?;
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            log::debug!(
                "assistant config missing when saving; recreating {}",
                path.display()
            );
            String::new()
        }
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::PermissionDenied | io::ErrorKind::InvalidData
            ) =>
        {
            log::warn!("failed to read assistant config {}: {}", path.display(), e);
            push_ui_error(format!(
                "cannot read {}. Check file permission or encoding.",
                path.display()
            ));
            String::new()
        }
        Err(e) => {
            log::debug!("failed to read assistant config {}: {}", path.display(), e);
            String::new()
        }
    };
    let cfg = parse_kaku_assistant_config(&raw);

    // Build updated config based on which field changed.
    // Every arm must round-trip ALL fields to avoid losing values not in the changed arm.
    let updated = match field_key {
        "Enabled" => {
            let enabled = matches!(new_val.trim(), "On" | "on" | "true" | "1");
            KakuAssistantConfig::new(enabled, cfg.api_key(), cfg.model(), cfg.base_url())
                .with_provider(cfg.provider())
                .with_custom_headers(cfg.custom_headers().to_vec())
                .with_web_search(cfg.web_search_provider(), cfg.web_search_api_key())
        }
        "Provider" => {
            // When provider changes, auto-fill base_url and default model.
            let provider_name = new_val.trim();
            if let Some(preset) = assistant_config::provider_preset(provider_name) {
                let new_base_url = if preset.base_url.is_empty() {
                    cfg.base_url()
                } else {
                    preset.base_url
                };
                let new_model = preset
                    .models
                    .first()
                    .copied()
                    .unwrap_or_else(|| cfg.model());
                KakuAssistantConfig::new(cfg.is_enabled(), cfg.api_key(), new_model, new_base_url)
                    .with_provider(provider_name)
                    .with_custom_headers(cfg.custom_headers().to_vec())
                    .with_web_search(cfg.web_search_provider(), cfg.web_search_api_key())
            } else {
                return Ok(());
            }
        }
        "Model" => {
            let model = if new_val.trim().is_empty() || new_val == "—" {
                assistant_config::DEFAULT_MODEL
            } else {
                new_val.trim()
            };
            KakuAssistantConfig::new(cfg.is_enabled(), cfg.api_key(), model, cfg.base_url())
                .with_provider(cfg.provider())
                .with_custom_headers(cfg.custom_headers().to_vec())
                .with_web_search(cfg.web_search_provider(), cfg.web_search_api_key())
        }
        "Base URL" => {
            let base_url = if new_val.trim().is_empty() || new_val == "—" {
                assistant_config::DEFAULT_BASE_URL
            } else {
                new_val.trim()
            };
            KakuAssistantConfig::new(cfg.is_enabled(), cfg.api_key(), cfg.model(), base_url)
                .with_provider(assistant_config::detect_provider(base_url))
                .with_custom_headers(cfg.custom_headers().to_vec())
                .with_web_search(cfg.web_search_provider(), cfg.web_search_api_key())
        }
        "API Key" => KakuAssistantConfig::new(
            cfg.is_enabled(),
            new_val.trim(),
            cfg.model(),
            cfg.base_url(),
        )
        .with_provider(cfg.provider())
        .with_custom_headers(cfg.custom_headers().to_vec())
        .with_web_search(cfg.web_search_provider(), cfg.web_search_api_key()),
        "Web Search" => {
            const VALID: &[&str] = &["none", "brave", "pipellm", "tavily"];
            let provider = if VALID.contains(&new_val.trim()) {
                new_val.trim()
            } else {
                "none"
            };
            // Clearing to "none" also wipes the key to avoid stale credentials.
            let key = if provider == "none" {
                String::new()
            } else {
                cfg.web_search_api_key().to_string()
            };
            KakuAssistantConfig::new(cfg.is_enabled(), cfg.api_key(), cfg.model(), cfg.base_url())
                .with_provider(cfg.provider())
                .with_custom_headers(cfg.custom_headers().to_vec())
                .with_web_search(provider, key)
        }
        "Search Key" => {
            KakuAssistantConfig::new(cfg.is_enabled(), cfg.api_key(), cfg.model(), cfg.base_url())
                .with_provider(cfg.provider())
                .with_custom_headers(cfg.custom_headers().to_vec())
                .with_web_search(cfg.web_search_provider(), new_val.trim())
        }
        _ => return Ok(()),
    };

    write_kaku_assistant_config(&path, &updated)
}

/// Get Gemini account email from google_accounts.json
fn get_gemini_account() -> Option<String> {
    let path = config::HOME_DIR
        .join(".gemini")
        .join("google_accounts.json");

    let parsed = read_json_file_with_debug(&path, "gemini account")?;

    // Extract "active" field
    parsed.get("active")?.as_str().map(|s| s.to_string())
}

/// Get Codex account email from JWT token in auth.json
fn get_codex_account() -> Option<String> {
    let auth_path = config::HOME_DIR.join(".codex").join("auth.json");
    let auth_json = read_json_file_with_debug(&auth_path, "codex account")?;

    // Extract access_token from tokens object
    let token = auth_json.get("tokens")?.get("access_token")?.as_str()?;

    let jwt_data = decode_jwt_payload_with_debug(token, "codex account")?;

    // OpenAI JWT payload contains email in custom claim
    jwt_data
        .get("https://api.openai.com/profile")?
        .get("email")?
        .as_str()
        .map(|s| s.to_string())
}

/// Get GitHub Copilot username from gh CLI
fn get_copilot_account() -> Option<String> {
    let output = match std::process::Command::new("gh")
        .args(["api", "user", "-q", ".login"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
    {
        Ok(output) => output,
        Err(e) => {
            log::debug!("gh account probe failed to launch: {}", e);
            return None;
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log::debug!(
            "gh account probe failed with status {}: {}",
            output.status,
            stderr.trim()
        );
        return None;
    }

    String::from_utf8(output.stdout)
        .map_err(|e| log::debug!("gh account probe returned non-utf8 stdout: {}", e))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn read_claude_auth_status_json() -> Option<serde_json::Value> {
    let output = match std::process::Command::new("claude")
        .args(["auth", "status"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
    {
        Ok(output) => output,
        Err(e) => {
            log::debug!("claude auth status probe failed to launch: {}", e);
            return None;
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log::debug!(
            "claude auth status probe failed with status {}: {}",
            output.status,
            stderr.trim()
        );
        return None;
    }

    let json_str = String::from_utf8(output.stdout)
        .map_err(|e| log::debug!("claude auth status probe returned non-utf8 stdout: {}", e))
        .ok()?;
    let parsed: serde_json::Value = serde_json::from_str(&json_str)
        .map_err(|e| log::debug!("failed to parse claude auth status json: {}", e))
        .ok()?;
    Some(parsed)
}

fn parse_claude_auth_status(parsed: &serde_json::Value) -> Option<String> {
    let logged_in = parsed.get("loggedIn").and_then(|value| value.as_bool());
    if matches!(logged_in, Some(false)) {
        return Some("✗ not signed in".into());
    }

    let account = parsed
        .get("email")
        .and_then(|value| value.as_str())
        .map(|value| value.to_string());
    let auth_method = parsed
        .get("authMethod")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("oauth");

    if logged_in == Some(true) || account.is_some() {
        Some(format_auth_status(account, auth_method))
    } else {
        None
    }
}

fn read_claude_auth_status() -> Option<String> {
    read_claude_auth_status_json()
        .as_ref()
        .and_then(parse_claude_auth_status)
        .or_else(|| read_claude_oauth_access_token().map(|_| format_auth_status(None, "oauth")))
}

fn kimi_credentials_path() -> PathBuf {
    config::HOME_DIR
        .join(".kimi")
        .join("credentials")
        .join("kimi-code.json")
}

fn read_kimi_auth_status_from_path(path: &Path) -> String {
    let Some(auth) = read_json_file_with_debug(path, "kimi credentials") else {
        return "✗ not signed in".into();
    };

    let has_token = auth
        .get("access_token")
        .and_then(|value| value.as_str())
        .is_some_and(|value| !value.is_empty());
    let expires_at = auth.get("expires_at").and_then(|value| value.as_f64());
    let has_refresh_token = auth
        .get("refresh_token")
        .and_then(|value| value.as_str())
        .is_some_and(|value| !value.is_empty());
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0);

    if has_refresh_token || (has_token && expires_at.is_some_and(|expires_at| expires_at > now)) {
        "✓ oauth".into()
    } else {
        "✗ login required".into()
    }
}

fn kimi_config_model_options(parsed: &toml::Value) -> Vec<String> {
    parsed
        .get("models")
        .and_then(|value| value.as_table())
        .map(|table| table.keys().cloned().collect::<Vec<_>>())
        .filter(|models| !models.is_empty())
        .unwrap_or_else(|| vec!["kimi-code/kimi-for-coding".into()])
}

fn read_kimi_auth_status() -> String {
    read_kimi_auth_status_from_path(&kimi_credentials_path())
}

/// Format auth status, with account fallback to auth method
fn format_auth_status(account: Option<String>, fallback_method: &str) -> String {
    match account {
        Some(acc) if !acc.is_empty() => format!("✓ {}", acc),
        _ => format!("✓ {}", fallback_method),
    }
}

/// Fetch models.dev data, cached to ~/.cache/kaku/models.json.
/// No TTL — use `r` key in TUI to force refresh.
fn load_models_dev_json() -> Option<serde_json::Value> {
    let cache_dir = config::HOME_DIR.join(".cache").join("kaku");
    let cache_path = cache_dir.join("models.json");

    // Use cache if exists
    if let Some(raw) = read_text_with_debug(&cache_path, "models.dev cache") {
        if let Some(v) = parse_json_value_with_debug(&raw, "models.dev cache") {
            return Some(v);
        }
    }

    // Fetch from API via curl (macOS built-in)
    fetch_models_dev_json()
}

/// Force fetch from models.dev and update cache.
fn fetch_models_dev_json() -> Option<serde_json::Value> {
    let cache_dir = config::HOME_DIR.join(".cache").join("kaku");
    let cache_path = cache_dir.join("models.json");

    let output = match std::process::Command::new("curl")
        .args(["-sS", "--max-time", "10", "https://models.dev/api.json"])
        .output()
    {
        Ok(output) => output,
        Err(e) => {
            log::debug!("models.dev fetch failed to launch curl: {}", e);
            return None;
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log::debug!(
            "models.dev fetch failed with status {}: {}",
            output.status,
            stderr.trim()
        );
        return None;
    }

    let raw = String::from_utf8(output.stdout)
        .map_err(|e| log::debug!("models.dev fetch returned non-utf8 stdout: {}", e))
        .ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| log::debug!("failed to parse models.dev json: {}", e))
        .ok()?;
    if let Err(e) = config::create_user_owned_dirs(&cache_dir) {
        log::debug!("Failed to create cache directory: {}", e);
    }
    if let Err(e) = std::fs::write(&cache_path, &raw) {
        log::debug!("Failed to write models cache: {}", e);
    }
    Some(v)
}

/// Read model IDs from models.dev for a given provider.
/// Returns latest models sorted by release_date (newest first), deduped to
/// only keep the short alias (e.g. "claude-sonnet-4-5") and skip dated
/// variants (e.g. "claude-sonnet-4-5-20250929").
fn read_models_dev(provider_id: &str) -> Vec<String> {
    let parsed = match load_models_dev_json() {
        Some(v) => v,
        None => return Vec::new(),
    };
    let models = match parsed
        .get(provider_id)
        .and_then(|p| p.get("models"))
        .and_then(|m| m.as_object())
    {
        Some(m) => m,
        None => return Vec::new(),
    };

    // Collect (id, release_date) pairs, skip embedding/tts/image-only models
    let mut items: Vec<(&str, &str)> = models
        .iter()
        .filter_map(|(id, m)| {
            // Skip non-chat models
            let name = m.get("name").and_then(|v| v.as_str()).unwrap_or("");
            if name.contains("Embedding")
                || name.contains("TTS")
                || name.contains("Image")
                || name.contains("Live")
            {
                return None;
            }
            // Skip dated variants (e.g. "claude-opus-4-5-20251101", "gemini-2.5-flash-preview-06-17")
            if id.chars().rev().take(8).all(|c| c.is_ascii_digit()) {
                return None;
            }
            // Skip dated preview variants with "xx-xx" suffix (e.g. "09-2025", "06-17")
            // Require both segments ≥ 2 chars to avoid filtering version numbers like "4-5"
            if let Some(last_dash) = id.rfind('-') {
                let suffix = &id[last_dash + 1..];
                if let Some(second_dash) = id[..last_dash].rfind('-') {
                    let prev = &id[second_dash + 1..last_dash];
                    if prev.len() >= 2
                        && prev.len() <= 4
                        && suffix.len() >= 2
                        && suffix.len() <= 4
                        && prev.chars().all(|c| c.is_ascii_digit())
                        && suffix.chars().all(|c| c.is_ascii_digit())
                    {
                        return None;
                    }
                }
            }
            // Skip "-latest" aliases (e.g. "gemini-flash-latest")
            if id.ends_with("-latest") {
                return None;
            }
            let rd = m.get("release_date").and_then(|v| v.as_str()).unwrap_or("");
            Some((id.as_str(), rd))
        })
        .collect();

    items.sort_by(|a, b| b.1.cmp(a.1));
    items
        .into_iter()
        .take(4)
        .map(|(id, _)| id.to_string())
        .collect()
}

fn extract_claude_code_fields(val: &serde_json::Value) -> Vec<FieldEntry> {
    let model = json_str(val, "model");

    let model_options = read_models_dev("anthropic");

    let display_value = if model.is_empty() {
        // Show the latest model name as default hint
        model_options
            .first()
            .cloned()
            .unwrap_or_else(|| "default".into())
    } else {
        model
    };

    let mut fields = vec![FieldEntry {
        key: "Model".into(),
        value: display_value,
        options: model_options,
        ..Default::default()
    }];

    if let Some(auth_status) = read_claude_auth_status() {
        fields.push(FieldEntry {
            key: "Auth".into(),
            value: auth_status,
            options: vec![],
            editable: false,
        });
    }

    // Show env-based provider config if present (e.g. OpenRouter, Pipe AI, Kimi)
    if let Some(env) = val.get("env").and_then(|e| e.as_object()) {
        let base_url = env
            .get("ANTHROPIC_BASE_URL")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let auth_token = env
            .get("ANTHROPIC_AUTH_TOKEN")
            .or_else(|| env.get("ANTHROPIC_API_KEY"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if !base_url.is_empty() {
            fields.push(FieldEntry {
                key: "Base URL".into(),
                value: base_url,
                options: vec![],
                ..Default::default()
            });
        }
        if !auth_token.is_empty() {
            fields.push(FieldEntry {
                key: "API Key".into(),
                value: mask_key(&auth_token),
                options: vec![],
                ..Default::default()
            });
        }
    }

    fields
}

fn extract_codex_fields(raw: &str) -> Vec<FieldEntry> {
    let mut fields = Vec::new();
    let mut has_model = false;

    // Read available models from Codex model cache
    let model_options = read_codex_model_options();

    // Parse TOML manually for the fields we care about
    for line in raw.lines() {
        let line = line.trim();
        if line.starts_with('[') || line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, val)) = line.split_once('=') {
            let key = key.trim().trim_matches('"');
            let val = val.trim().trim_matches('"');
            match key {
                "model" => {
                    has_model = true;
                    fields.push(FieldEntry {
                        key: "Model".into(),
                        value: val.to_string(),
                        options: model_options.clone(),
                        ..Default::default()
                    });
                }
                _ => {}
            }
        }
    }

    if !has_model {
        let default_model = model_options
            .first()
            .cloned()
            .unwrap_or_else(|| "default".into());
        fields.insert(
            0,
            FieldEntry {
                key: "Model".into(),
                value: default_model,
                options: model_options.clone(),
                ..Default::default()
            },
        );
    }

    // Check auth status from auth.json
    let auth_path = config::HOME_DIR.join(".codex").join("auth.json");
    if let Some(auth) = read_json_file_with_debug(&auth_path, "codex auth status") {
        let auth_mode = auth.get("auth_mode").and_then(|v| v.as_str()).unwrap_or("");
        if !auth_mode.is_empty() {
            let account = get_codex_account();
            fields.push(FieldEntry {
                key: "Auth".into(),
                value: format_auth_status(account, auth_mode),
                options: vec![],
                editable: false,
            });
        }
    }

    fields
}

fn extract_kimi_fields(raw: &str) -> Vec<FieldEntry> {
    let parsed = raw.parse::<toml::Value>().ok();
    let model_options = parsed
        .as_ref()
        .map(kimi_config_model_options)
        .unwrap_or_else(|| vec!["kimi-code/kimi-for-coding".into()]);
    let model = parsed
        .as_ref()
        .and_then(|parsed| parsed.get("default_model"))
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .to_string();

    let display_value = if model.is_empty() {
        model_options
            .first()
            .cloned()
            .unwrap_or_else(|| "kimi-code/kimi-for-coding".into())
    } else {
        model
    };

    vec![
        FieldEntry {
            key: "Model".into(),
            value: display_value,
            options: model_options,
            ..Default::default()
        },
        FieldEntry {
            key: "Auth".into(),
            value: read_kimi_auth_status(),
            options: vec![],
            editable: false,
        },
    ]
}

fn extract_antigravity_fields(snapshot: Option<&AntigravityUsageSnapshot>) -> Vec<FieldEntry> {
    let auth = read_antigravity_auth_status();
    let account = auth.as_ref().and_then(|status| {
        status
            .get("email")
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .map(|value| value.to_string())
            .or_else(|| {
                status
                    .get("name")
                    .and_then(|value| value.as_str())
                    .filter(|value| !value.is_empty())
                    .map(|value| value.to_string())
            })
    });

    let mut fields = Vec::new();

    if let Some(snapshot) = snapshot {
        if let Some(label) = &snapshot.selected_model_label {
            fields.push(FieldEntry {
                key: "Model".into(),
                value: label.clone(),
                options: vec![],
                editable: false,
            });
        }
    }

    if auth.is_some() {
        fields.push(FieldEntry {
            key: "Auth".into(),
            value: format_auth_status(account, "oauth"),
            options: vec![],
            editable: false,
        });
    }

    fields
}

/// Read model slugs from Codex's own cache, or from models.dev.
fn read_codex_model_options() -> Vec<String> {
    let cache_path = config::HOME_DIR.join(".codex").join("models_cache.json");
    if let Some(parsed) = read_json_file_with_debug(&cache_path, "codex model cache") {
        let mut models: Vec<(String, usize)> = parsed
            .get("models")
            .and_then(|m| m.as_array())
            .map(|arr| {
                arr.iter()
                    .filter(|m| {
                        m.get("visibility")
                            .and_then(|v| v.as_str())
                            .map(|v| v == "list")
                            .unwrap_or(false)
                    })
                    .filter_map(|m| {
                        let slug = m.get("slug").and_then(|v| v.as_str())?;
                        let priority =
                            m.get("priority").and_then(|v| v.as_u64()).unwrap_or(999) as usize;
                        Some((slug.to_string(), priority))
                    })
                    .collect()
            })
            .unwrap_or_default();
        if !models.is_empty() {
            models.sort_by_key(|(_, p)| *p);
            return models.into_iter().map(|(s, _)| s).collect();
        }
    }

    read_models_dev("openai")
}

fn extract_gemini_fields(val: &serde_json::Value) -> Vec<FieldEntry> {
    let mut fields = Vec::new();

    let model = val
        .get("model")
        .and_then(|m| {
            m.get("name")
                .and_then(|n| n.as_str())
                .or_else(|| m.as_str())
        })
        .unwrap_or("")
        .to_string();
    let model_options = read_models_dev("google");

    let display_value = if model.is_empty() {
        model_options
            .first()
            .cloned()
            .unwrap_or_else(|| "default".into())
    } else {
        model
    };

    fields.push(FieldEntry {
        key: "Model".into(),
        value: display_value,
        options: model_options,
        ..Default::default()
    });

    let auth_type = val
        .get("security")
        .and_then(|s| s.get("auth"))
        .and_then(|a| a.get("selectedType"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if !auth_type.is_empty() {
        let account = get_gemini_account();
        fields.push(FieldEntry {
            key: "Auth".into(),
            value: format_auth_status(account, auth_type),
            options: vec![],
            editable: false,
        });
    }

    fields
}

/// Read model choices from `copilot --help` output, fallback to models.dev.
fn read_copilot_model_options() -> Vec<String> {
    let output = match std::process::Command::new("copilot").arg("--help").output() {
        Ok(output) => output,
        Err(e) => {
            log::debug!("copilot --help probe failed to launch: {}", e);
            return read_models_dev("anthropic");
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log::debug!(
            "copilot --help probe failed with status {}: {}",
            output.status,
            stderr.trim()
        );
        return read_models_dev("anthropic");
    }

    let text = String::from_utf8_lossy(&output.stdout);
    // Find "--model" first, then parse the choices after it
    if let Some(model_pos) = text.find("--model") {
        let after_model = &text[model_pos..];
        if let Some(choices_pos) = after_model.find("choices:") {
            let rest = &after_model[choices_pos + "choices:".len()..];
            if let Some(end) = rest.find(')') {
                let choices_str = &rest[..end];
                let models: Vec<String> = choices_str
                    .split(',')
                    .map(|s| s.trim().trim_matches('"').trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                if !models.is_empty() {
                    return models;
                }
            }
        }
    }

    log::debug!("copilot --help output did not expose model choices");
    read_models_dev("anthropic")
}

fn extract_copilot_fields(val: &serde_json::Value) -> Vec<FieldEntry> {
    let model = json_str(val, "model");

    let model_options = read_copilot_model_options();

    let mut fields = vec![FieldEntry {
        key: "Model".into(),
        value: if model.is_empty() {
            "default".into()
        } else {
            model
        },
        options: model_options,
        ..Default::default()
    }];

    // Copilot authenticates via GitHub OAuth; session files indicate auth
    let session_dir = config::HOME_DIR.join(".copilot").join("session-state");
    if session_dir.exists() {
        let account = get_copilot_account();
        fields.push(FieldEntry {
            key: "Auth".into(),
            value: format_auth_status(account, "github"),
            options: vec![],
            editable: false,
        });
    }

    fields
}

fn extract_factory_droid_fields(val: &serde_json::Value) -> Vec<FieldEntry> {
    let session_defaults = val
        .get("sessionDefaultSettings")
        .and_then(|v| v.as_object());
    let model = json_nested_or_top_str(val, "sessionDefaultSettings", "model", "model");
    let reasoning = json_nested_or_top_str(
        val,
        "sessionDefaultSettings",
        "reasoningEffort",
        "reasoningEffort",
    );
    let autonomy = session_defaults
        .and_then(|s| s.get("autonomyMode").or_else(|| s.get("autonomyLevel")))
        .and_then(|v| v.as_str())
        .or_else(|| {
            val.get("autonomyMode")
                .or_else(|| val.get("autonomyLevel"))
                .and_then(|v| v.as_str())
        })
        .unwrap_or("")
        .to_string();

    vec![
        FieldEntry {
            key: "Model".into(),
            value: if model.is_empty() {
                FACTORY_DROID_DEFAULT_MODEL.into()
            } else {
                model
            },
            options: vec![],
            editable: true,
        },
        FieldEntry {
            key: "Reasoning Effort".into(),
            value: if reasoning.is_empty() {
                FACTORY_DROID_DEFAULT_REASONING.into()
            } else {
                reasoning
            },
            options: string_options(&FACTORY_DROID_REASONING_OPTIONS),
            editable: true,
        },
        FieldEntry {
            key: "Autonomy Level".into(),
            value: if autonomy.is_empty() {
                FACTORY_DROID_DEFAULT_AUTONOMY.into()
            } else {
                autonomy
            },
            options: string_options(&FACTORY_DROID_AUTONOMY_OPTIONS),
            editable: true,
        },
    ]
}

fn factory_droid_session_key(field_key: &str) -> Option<&'static str> {
    match field_key {
        "Model" => Some("model"),
        "Reasoning Effort" => Some("reasoningEffort"),
        // Normalize writes to `autonomyMode` while still accepting legacy `autonomyLevel`.
        "Autonomy Level" => Some("autonomyMode"),
        _ => None,
    }
}

fn save_factory_droid_field(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    field_key: &str,
    new_val: &str,
) -> anyhow::Result<()> {
    let Some(target_key) = factory_droid_session_key(field_key) else {
        return Ok(());
    };

    let session_defaults = obj
        .entry("sessionDefaultSettings")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .context("sessionDefaultSettings is not an object")?;

    if new_val == "—" || new_val.is_empty() {
        session_defaults.remove(target_key);
    } else {
        session_defaults.insert(
            target_key.to_string(),
            serde_json::Value::String(new_val.to_string()),
        );
    }

    Ok(())
}

fn extract_openclaw_fields(val: &serde_json::Value) -> Vec<FieldEntry> {
    let agents = val.get("agents").unwrap_or(&serde_json::Value::Null);
    let defaults = agents.get("defaults").unwrap_or(&serde_json::Value::Null);
    let model = defaults.get("model").unwrap_or(&serde_json::Value::Null);

    let primary = model
        .get("primary")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // Collect all model IDs from all providers for the Primary Model selector
    let mut all_model_ids: Vec<String> = Vec::new();

    let mut provider_fields: Vec<FieldEntry> = Vec::new();

    // Dynamically enumerate all providers
    if let Some(providers) = val
        .get("models")
        .and_then(|m| m.get("providers"))
        .and_then(|p| p.as_object())
    {
        for (name, prov) in providers {
            let url = prov
                .get("baseUrl")
                .or_else(|| prov.get("baseURL"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let key = prov
                .get("apiKey")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            // Collect models from this provider
            let model_ids: Vec<String> = prov
                .get("models")
                .and_then(|m| m.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|m| m.get("id").and_then(|v| v.as_str()))
                        .map(|id| format!("{}/{}", name, id))
                        .collect()
                })
                .unwrap_or_default();

            // Also check agents.defaults.models for any registered models with this provider
            if let Some(agent_models) = defaults.get("models").and_then(|m| m.as_object()) {
                for model_key in agent_models.keys() {
                    if model_key.starts_with(&format!("{}/", name))
                        && !all_model_ids.contains(model_key)
                    {
                        all_model_ids.push(model_key.clone());
                    }
                }
            }

            for mid in &model_ids {
                if !all_model_ids.contains(mid) {
                    all_model_ids.push(mid.clone());
                }
            }

            let models_display = if model_ids.is_empty() {
                // Show from agent defaults if provider models array is empty
                all_model_ids
                    .iter()
                    .filter(|m| m.starts_with(&format!("{}/", name)))
                    .map(|m| m.split('/').last().unwrap_or(m).to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            } else {
                model_ids
                    .iter()
                    .map(|m| m.split('/').last().unwrap_or(m).to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            };

            // Provider header
            let api_type = prov
                .get("api")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            provider_fields.push(FieldEntry {
                key: name.clone(),
                value: if api_type.is_empty() {
                    "provider".into()
                } else {
                    api_type
                },
                options: vec![],
                editable: false,
            });

            provider_fields.push(FieldEntry {
                key: format!("{} ▸ Base URL", name),
                value: if url.is_empty() { "—".into() } else { url },
                options: vec![],
                ..Default::default()
            });
            provider_fields.push(FieldEntry {
                key: format!("{} ▸ API Key", name),
                value: mask_key(&key),
                options: vec![],
                ..Default::default()
            });
            if !models_display.is_empty() {
                provider_fields.push(FieldEntry {
                    key: format!("{} ▸ Models", name),
                    value: models_display,
                    options: vec![],
                    editable: false,
                });
            }
        }
    }

    // Fallback: if no providers defined, collect models from agents.defaults.models
    if all_model_ids.is_empty() {
        if let Some(agent_models) = defaults.get("models").and_then(|m| m.as_object()) {
            for model_key in agent_models.keys() {
                all_model_ids.push(model_key.clone());
            }
        }
    }

    let mut fields = vec![FieldEntry {
        key: "Primary Model".into(),
        value: if primary.is_empty() {
            "—".into()
        } else {
            primary
        },
        options: all_model_ids,
        ..Default::default()
    }];

    fields.extend(provider_fields);

    let plugins = val
        .get("plugins")
        .and_then(|p| p.get("entries"))
        .and_then(|e| e.as_object())
        .map(|obj| {
            obj.keys()
                .filter(|k| {
                    obj.get(*k)
                        .and_then(|v| v.get("enabled"))
                        .and_then(|v| v.as_bool())
                        .unwrap_or(true)
                })
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();

    fields.push(FieldEntry {
        key: "Plugins".into(),
        value: if plugins.is_empty() {
            "—".into()
        } else {
            plugins
        },
        options: vec![],
        editable: false,
    });

    fields
}

#[derive(Clone, Copy, PartialEq)]
enum Focus {
    ToolList,
    Editor,
}

#[derive(Clone, PartialEq, Eq)]
enum AppMode {
    Browsing,
    Editing {
        field_idx: usize,
        buffer: String,
        cursor: usize,
    },
    Selecting {
        field_idx: usize,
        options: Vec<String>,
        selected: usize,
    },
}

struct App {
    tools: Vec<ToolState>,
    tool_index: usize,
    field_index: usize,
    assistant_collapsed: bool,
    usage_update_rx: Option<Receiver<UsageSummaryUpdate>>,
    usage_update_tx: Option<Sender<UsageSummaryUpdate>>,
    usage_update_generation: Arc<AtomicUsize>,
    focus: Focus,
    mode: AppMode,
    status_msg: Option<String>,
    status_expire: Option<Instant>,
    toast_msg: Option<String>,
    toast_expire: Option<Instant>,
    last_error: Option<String>,
    error_expire: Option<Instant>,
    should_quit: bool,
}

fn status_value_for_display(field_key: &str, new_val: &str) -> String {
    if field_key.contains("API Key") {
        return if new_val.trim().is_empty() {
            "—".into()
        } else {
            mask_key(new_val.trim())
        };
    }
    new_val.to_string()
}

impl App {
    fn new() -> Self {
        let tools = Self::load_tools();
        let first = tools.iter().position(|t| !t.fields.is_empty()).unwrap_or(0);
        let mut app = App {
            tools,
            tool_index: first,
            field_index: 0,
            assistant_collapsed: false,
            usage_update_rx: None,
            usage_update_tx: None,
            usage_update_generation: Arc::new(AtomicUsize::new(0)),
            focus: Focus::ToolList,
            mode: AppMode::Browsing,
            status_msg: None,
            status_expire: None,
            toast_msg: None,
            toast_expire: None,
            last_error: None,
            error_expire: None,
            should_quit: false,
        };
        app.restart_usage_loading();
        let _ = app.sync_transient_errors();
        app
    }

    fn load_tools() -> Vec<ToolState> {
        ALL_TOOLS
            .iter()
            .map(|t| ToolState::load_without_remote_usage(*t))
            .filter(|t| t.tool == Tool::KakuAssistant || t.installed)
            .collect()
    }

    fn tool_row_count(&self, tool_index: usize) -> usize {
        let Some(tool) = self.tools.get(tool_index) else {
            return 0;
        };
        if tool.tool == Tool::KakuAssistant {
            return 1 + if self.assistant_collapsed {
                0
            } else {
                tool.fields.len()
            };
        }
        tool.fields.len()
    }

    fn current_tool(&self) -> Option<&ToolState> {
        self.tools.get(self.tool_index)
    }

    fn current_tool_mut(&mut self) -> Option<&mut ToolState> {
        self.tools.get_mut(self.tool_index)
    }

    fn tool_is_collapsed(&self, tool_index: usize) -> bool {
        self.tools
            .get(tool_index)
            .is_some_and(|tool| tool.tool == Tool::KakuAssistant && self.assistant_collapsed)
    }

    fn selected_field_index(&self) -> Option<usize> {
        let tool = self.current_tool()?;
        if tool.fields.is_empty() {
            return None;
        }

        if tool.tool == Tool::KakuAssistant {
            self.field_index.checked_sub(1)
        } else {
            Some(self.field_index)
        }
    }

    fn display_index_for_field(&self, tool: Tool, field_index: usize) -> usize {
        if tool == Tool::KakuAssistant {
            field_index + 1
        } else {
            field_index
        }
    }

    fn toggle_kaku_assistant_collapsed(&mut self) {
        if !self
            .tools
            .iter()
            .any(|tool| tool.tool == Tool::KakuAssistant && !tool.fields.is_empty())
        {
            return;
        }

        self.assistant_collapsed = !self.assistant_collapsed;
        if self
            .current_tool()
            .is_some_and(|tool| tool.tool == Tool::KakuAssistant)
        {
            self.field_index = 0;
        }
        self.set_status(if self.assistant_collapsed {
            "Kaku Assistant hidden"
        } else {
            "Kaku Assistant expanded"
        });
    }

    fn total_rows(&self) -> usize {
        (0..self.tools.len())
            .map(|idx| self.tool_row_count(idx))
            .sum()
    }

    fn rendered_tool_row_count(&self) -> usize {
        self.tools
            .iter()
            .map(|tool| {
                1 + if tool.tool == Tool::KakuAssistant && self.assistant_collapsed {
                    1
                } else {
                    tool.fields.len() + 1
                }
            })
            .sum()
    }

    fn flatten_index(&self) -> usize {
        let mut idx = 0;
        for (ti, _) in self.tools.iter().enumerate() {
            let count = self.tool_row_count(ti);
            if ti == self.tool_index {
                return idx + self.field_index.min(count.saturating_sub(1));
            }
            idx += count;
        }
        idx
    }

    fn is_editing(&self) -> bool {
        matches!(self.mode, AppMode::Editing { .. })
    }

    fn is_selecting(&self) -> bool {
        matches!(self.mode, AppMode::Selecting { .. })
    }

    fn editing_view(&self) -> Option<(usize, &str, usize)> {
        match &self.mode {
            AppMode::Editing {
                field_idx,
                buffer,
                cursor,
            } => Some((*field_idx, buffer.as_str(), *cursor)),
            _ => None,
        }
    }

    fn editing_mut(&mut self) -> Option<(&mut String, &mut usize)> {
        match &mut self.mode {
            AppMode::Editing { buffer, cursor, .. } => Some((buffer, cursor)),
            _ => None,
        }
    }

    fn selecting_view(&self) -> Option<(usize, &[String], usize)> {
        match &self.mode {
            AppMode::Selecting {
                field_idx,
                options,
                selected,
            } => Some((*field_idx, options.as_slice(), *selected)),
            _ => None,
        }
    }

    fn set_error(&mut self, message: impl Into<String>) {
        self.last_error = Some(message.into());
        self.error_expire = Some(Instant::now() + UI_ERROR_TTL);
    }

    fn set_status(&mut self, message: impl Into<String>) {
        self.status_msg = Some(message.into());
        self.status_expire = Some(Instant::now() + UI_STATUS_TTL);
    }

    fn set_toast(&mut self, message: impl Into<String>) {
        self.toast_msg = Some(message.into());
        self.toast_expire = Some(Instant::now() + UI_STATUS_TTL);
    }

    fn toast_message(&self) -> Option<&str> {
        self.toast_msg.as_deref()
    }

    fn open_antigravity_app(&mut self) {
        #[cfg(test)]
        {
            return;
        }

        #[cfg(not(test))]
        match std::process::Command::new("open")
            .args(["-a", "Antigravity"])
            .status()
        {
            Ok(status) if status.success() => {}
            Ok(status) => self.set_error(format!(
                "Failed to open Antigravity (exit status: {})",
                status
            )),
            Err(err) => self.set_error(format!("Failed to open Antigravity: {}", err)),
        }
    }

    fn sync_transient_errors(&mut self) -> bool {
        let mut changed = self.drain_usage_updates();

        while let Some(error) = pop_ui_error() {
            self.set_error(error);
            changed = true;
        }

        let now = Instant::now();
        if self.status_expire.is_some_and(|t| now >= t) {
            self.status_msg = None;
            self.status_expire = None;
            changed = true;
        }

        if self.toast_expire.is_some_and(|t| now >= t) {
            self.toast_msg = None;
            self.toast_expire = None;
            changed = true;
        }

        if self.error_expire.is_some_and(|t| now >= t) {
            self.last_error = None;
            self.error_expire = None;
            changed = true;
        }

        changed
    }

    fn restart_usage_loading(&mut self) {
        self.usage_update_rx = None;
        self.usage_update_tx = None;

        let (tx, rx) = mpsc::channel();
        let mut spawned = false;
        let generation = self.usage_update_generation.fetch_add(1, Ordering::Relaxed) + 1;

        for tool in self
            .tools
            .iter()
            .filter(|tool| tool.installed && supports_remote_usage(tool.tool))
        {
            spawned = true;
            let tx = tx.clone();
            let tool_kind = tool.tool;
            let active_generation = Arc::clone(&self.usage_update_generation);
            std::thread::spawn(move || {
                if active_generation.load(Ordering::Relaxed) != generation {
                    return;
                }
                let update = load_usage_update(tool_kind);
                if active_generation.load(Ordering::Relaxed) != generation {
                    return;
                }
                let _ = tx.send(update);
            });
        }

        if spawned {
            self.usage_update_rx = Some(rx);
            self.usage_update_tx = Some(tx);
        } else {
            self.usage_update_rx = None;
            self.usage_update_tx = None;
        }
    }

    fn schedule_usage_reload(&self, tool: Tool) {
        if !supports_remote_usage(tool) {
            return;
        }
        if !self
            .tools
            .iter()
            .any(|entry| entry.tool == tool && entry.installed)
        {
            return;
        }
        let Some(tx) = self.usage_update_tx.clone() else {
            return;
        };
        let generation = self.usage_update_generation.load(Ordering::Relaxed);
        let active_generation = Arc::clone(&self.usage_update_generation);
        std::thread::spawn(move || {
            if active_generation.load(Ordering::Relaxed) != generation {
                return;
            }
            let update = load_usage_update(tool);
            if active_generation.load(Ordering::Relaxed) != generation {
                return;
            }
            let _ = tx.send(update);
        });
    }

    fn drain_usage_updates(&mut self) -> bool {
        let Some(rx) = &self.usage_update_rx else {
            return false;
        };

        let mut updates = Vec::new();
        while let Ok(update) = rx.try_recv() {
            updates.push(update);
        }

        let changed = !updates.is_empty();
        for update in updates {
            if let Some(index) = self.tools.iter().position(|tool| tool.tool == update.tool) {
                if let Some(fields) = update.fields {
                    self.tools[index].fields = fields;
                }
                let installed = self.tools[index].installed;
                let summary = summarize_tool_fields(
                    update.tool,
                    installed,
                    &self.tools[index].fields,
                    update.summary.as_deref(),
                );
                self.tools[index].summary = summary;
            }
        }
        changed
    }

    fn set_from_flat(&mut self, flat: usize) {
        let previous_tool = self.current_tool().map(|tool| tool.tool);
        let mut remaining = flat;
        for (ti, _) in self.tools.iter().enumerate() {
            let count = self.tool_row_count(ti);
            if count == 0 {
                continue;
            }
            if remaining < count {
                self.tool_index = ti;
                self.field_index = remaining;
                if self.current_tool().map(|tool| tool.tool) != previous_tool {
                    if let Some(tool) = self.current_tool().map(|tool| tool.tool) {
                        self.schedule_usage_reload(tool);
                    }
                }
                return;
            }
            remaining -= count;
        }
    }

    fn move_up(&mut self) {
        let flat = self.flatten_index();
        if flat > 0 {
            self.set_from_flat(flat - 1);
        }
    }

    fn move_down(&mut self) {
        let flat = self.flatten_index();
        if flat + 1 < self.total_rows() {
            self.set_from_flat(flat + 1);
        }
    }

    fn move_select_up(&mut self) {
        if let AppMode::Selecting { selected, .. } = &mut self.mode {
            if *selected > 0 {
                *selected -= 1;
            }
        }
    }

    fn move_select_down(&mut self) {
        if let AppMode::Selecting {
            selected, options, ..
        } = &mut self.mode
        {
            if *selected + 1 < options.len() {
                *selected += 1;
            }
        }
    }

    fn start_edit(&mut self) {
        if self
            .current_tool()
            .is_some_and(|tool| tool.tool == Tool::KakuAssistant && self.field_index == 0)
        {
            self.toggle_kaku_assistant_collapsed();
            return;
        }

        let Some(tool) = self.current_tool() else {
            return;
        };
        if !tool.installed || tool.fields.is_empty() {
            return;
        }
        let Some(selected_field_idx) = self.selected_field_index() else {
            return;
        };
        if selected_field_idx >= tool.fields.len() {
            return;
        }
        let field = &tool.fields[selected_field_idx];

        if tool.tool == Tool::Antigravity && field.key == "Model" {
            self.set_toast("Change the model in Antigravity settings.");
            self.open_antigravity_app();
            return;
        }

        // Show OAuth re-authentication command for non-editable auth fields
        if !field.editable {
            if field.key == "Auth" || (field.value.starts_with('✓') && !field.key.contains(" ▸ "))
            {
                let cmd = match tool.tool {
                    Tool::KakuAssistant => None,
                    Tool::Gemini => Some("gemini auth login"),
                    Tool::Codex => Some("codex auth login"),
                    Tool::Kimi => Some("kimi login"),
                    Tool::Antigravity => Some("open -a Antigravity"),
                    Tool::Copilot => Some("gh auth login"),
                    Tool::FactoryDroid => Some("droid"),
                    Tool::ClaudeCode => Some("claude auth login"),
                    Tool::OpenClaw => None,
                };

                if let Some(auth_cmd) = cmd {
                    self.open_in_terminal(auth_cmd);
                } else if tool.tool == Tool::OpenClaw {
                    self.set_status("OpenClaw uses API keys, check config file");
                }
            }
            return;
        }

        if !field.options.is_empty() {
            self.mode = AppMode::Selecting {
                field_idx: selected_field_idx,
                options: field.options.clone(),
                selected: field
                    .options
                    .iter()
                    .position(|o| *o == field.value)
                    .unwrap_or(0),
            };
            self.focus = Focus::Editor;
            return;
        }

        let edit_buf = if field.value == "—" {
            // Empty placeholder
            String::new()
        } else if tool.tool == Tool::KakuAssistant && field.key == "API Key" {
            get_kaku_assistant_api_key().unwrap_or_else(String::new)
        } else {
            field.value.clone()
        };
        let edit_cursor = edit_buf.len(); // Start cursor at end (always a valid byte boundary)
        self.mode = AppMode::Editing {
            field_idx: selected_field_idx,
            buffer: edit_buf,
            cursor: edit_cursor,
        };
        self.focus = Focus::Editor;
    }

    fn confirm_select(&mut self) {
        let AppMode::Selecting {
            field_idx,
            options,
            selected,
        } = std::mem::replace(&mut self.mode, AppMode::Browsing)
        else {
            return;
        };
        self.focus = Focus::ToolList;

        if selected >= options.len() {
            return;
        }

        let Some((tool_kind, field_key, old_val)) = self.current_tool().and_then(|tool| {
            tool.fields
                .get(field_idx)
                .map(|field| (tool.tool, field.key.clone(), field.value.clone()))
        }) else {
            return;
        };
        self.field_index = self.display_index_for_field(tool_kind, field_idx);
        let new_val = options[selected].clone();

        if new_val == old_val {
            return;
        }

        if let Some(tool) = self.current_tool_mut() {
            tool.fields[field_idx].value = new_val.clone();
        }
        let status_val = status_value_for_display(&field_key, &new_val);
        match save_field(tool_kind, &field_key, &new_val) {
            Ok(()) => self.set_status(format!("Saved {} → {}", field_key, status_val)),
            Err(e) => {
                self.set_status(format!("Save failed: {}", e));
                self.set_error(format!("Save failed: {}", e));
            }
        }
        self.reload_current_tool();
    }

    fn cancel_select(&mut self) {
        self.mode = AppMode::Browsing;
        self.focus = Focus::ToolList;
    }

    fn confirm_edit(&mut self) {
        let AppMode::Editing {
            field_idx,
            buffer,
            cursor: _,
        } = std::mem::replace(&mut self.mode, AppMode::Browsing)
        else {
            return;
        };
        self.focus = Focus::ToolList;

        let Some((tool_kind, field_key, old_val)) = self.current_tool().and_then(|tool| {
            tool.fields
                .get(field_idx)
                .map(|field| (tool.tool, field.key.clone(), field.value.clone()))
        }) else {
            return;
        };
        self.field_index = if tool_kind == Tool::KakuAssistant {
            field_idx + 1
        } else {
            field_idx
        };
        let new_val = buffer.trim().to_string();

        // Empty input on API Key fields means cancel, not clear
        if new_val.is_empty() && field_key.contains("API Key") {
            return;
        }

        if new_val == old_val || (new_val.is_empty() && old_val == "—") {
            return;
        }

        if let Some(tool) = self.current_tool_mut() {
            tool.fields[field_idx].value = new_val.clone();
        }

        let status_val = status_value_for_display(&field_key, &new_val);
        match save_field(tool_kind, &field_key, &new_val) {
            Ok(()) => self.set_status(format!("Saved {} → {}", field_key, status_val)),
            Err(e) => {
                self.set_status(format!("Save failed: {}", e));
                self.set_error(format!("Save failed: {}", e));
            }
        }
        self.reload_current_tool();
    }

    fn reload_current_tool(&mut self) {
        let Some(tool_type) = self.current_tool().map(|tool| tool.tool) else {
            return;
        };
        let refreshed = ToolState::load_without_remote_usage(tool_type);
        if let Some(tool) = self.current_tool_mut() {
            *tool = refreshed;
        }
        if tool_type == Tool::KakuAssistant {
            self.assistant_collapsed = false;
            self.field_index = 0;
        }
        self.schedule_usage_reload(tool_type);
    }

    fn cancel_edit(&mut self) {
        self.mode = AppMode::Browsing;
        self.focus = Focus::ToolList;
    }

    fn config_path_to_open(&self) -> Option<PathBuf> {
        let tool = self.current_tool()?;
        let path = tool.tool.config_path();
        path.exists().then_some(path)
    }

    fn refresh_models(&mut self) {
        let codex_usage_cache = codex_usage_cache_path();
        if let Err(e) = std::fs::remove_file(&codex_usage_cache) {
            log::trace!("Could not remove codex usage cache: {}", e);
        }
        let claude_usage_cache = claude_usage_cache_path();
        if let Err(e) = std::fs::remove_file(&claude_usage_cache) {
            log::trace!("Could not remove claude usage cache: {}", e);
        }
        let copilot_usage_cache = copilot_usage_cache_path();
        if let Err(e) = std::fs::remove_file(&copilot_usage_cache) {
            log::trace!("Could not remove copilot usage cache: {}", e);
        }
        let kimi_usage_cache = kimi_usage_cache_path();
        if let Err(e) = std::fs::remove_file(&kimi_usage_cache) {
            log::trace!("Could not remove kimi usage cache: {}", e);
        }
        let antigravity_usage_cache = antigravity_usage_cache_path();
        if let Err(e) = std::fs::remove_file(&antigravity_usage_cache) {
            log::trace!("Could not remove antigravity usage cache: {}", e);
        }

        let models_refreshed = fetch_models_dev_json().is_some();
        let assistant_cache = assistant_models_cache_path();
        if let Err(e) = std::fs::remove_file(&assistant_cache) {
            log::trace!("Could not remove assistant models cache: {}", e);
        }
        self.tools = Self::load_tools();
        self.tool_index = self
            .tools
            .iter()
            .position(|tool| !tool.fields.is_empty())
            .unwrap_or(0);
        self.field_index = 0;
        self.restart_usage_loading();
        if models_refreshed {
            self.set_status("AI settings refreshed");
        } else {
            self.set_status("Usage refreshed");
            self.set_error("Models refresh failed. Kept local model cache.");
        }
        let _ = self.sync_transient_errors();
    }

    /// Open a shell command in a new Kaku tab (preferred) or fall back to Terminal.app.
    fn open_in_terminal(&mut self, cmd: &str) {
        // Prefer a new tab in the current Kaku window.
        // kaku cli spawn reads WEZTERM_PANE from the environment to target the right window.
        // Append `exec $SHELL` so the pane stays alive after the command finishes.
        let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/zsh".to_string());
        let shell_cmd = format!("{}; exec \"{}\"", cmd, shell);
        let kaku_status = std::process::Command::new("kaku")
            .args(["cli", "spawn", "--", &shell, "-c", &shell_cmd])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        if kaku_status.as_ref().is_ok_and(|status| status.success()) {
            self.set_status("Opening in new Kaku tab...");
            return;
        }

        // Fallback: open in macOS Terminal.app via osascript.
        let script = format!("tell application \"Terminal\" to do script \"{}\"", cmd);
        match std::process::Command::new("osascript")
            .arg("-e")
            .arg(&script)
            .spawn()
        {
            Ok(_) => self.set_status("Opening in new terminal window..."),
            Err(_) => self.set_status(format!("Failed to open terminal. Run '{}' manually", cmd)),
        }
    }
}

fn save_field(tool: Tool, field_key: &str, new_val: &str) -> anyhow::Result<()> {
    if tool == Tool::KakuAssistant {
        return save_kaku_assistant_field(field_key, new_val);
    }

    // Codex uses TOML; delegate immediately before any JSON parsing attempt.
    if tool == Tool::Codex {
        return save_codex_field(field_key, new_val);
    }
    if tool == Tool::Kimi {
        return save_kimi_field(field_key, new_val);
    }

    let path = tool.config_path();
    if !path.exists() {
        anyhow::bail!("config file not found: {}", path.display());
    }

    let raw = std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let mut parsed: serde_json::Value =
        parse_json_or_jsonc(&raw).with_context(|| format!("parse {}", path.display()))?;

    match tool {
        Tool::KakuAssistant => unreachable!("Kaku Assistant is handled before JSON parsing"),
        Tool::Antigravity => return Ok(()),
        Tool::Gemini => {
            if field_key == "Model" {
                if let Some(obj) = parsed.as_object_mut() {
                    if new_val == "—" || new_val.is_empty() {
                        obj.remove("model");
                    } else {
                        let keep_string_shape = obj.get("model").is_some_and(|m| m.is_string());
                        if keep_string_shape {
                            obj.insert(
                                "model".to_string(),
                                serde_json::Value::String(new_val.to_string()),
                            );
                        } else {
                            obj.insert("model".to_string(), serde_json::json!({"name": new_val}));
                        }
                    }
                }
            } else {
                return Ok(());
            }
        }
        Tool::Copilot => {
            if field_key == "Model" {
                if let Some(obj) = parsed.as_object_mut() {
                    if new_val == "—" || new_val.is_empty() {
                        obj.remove("model");
                    } else {
                        obj.insert(
                            "model".to_string(),
                            serde_json::Value::String(new_val.to_string()),
                        );
                    }
                }
            } else {
                return Ok(());
            }
        }
        Tool::FactoryDroid => {
            let obj = parsed.as_object_mut().context("root is not object")?;
            save_factory_droid_field(obj, field_key, new_val)?;
        }
        Tool::ClaudeCode => {
            let env_key = match field_key {
                "Base URL" => Some("ANTHROPIC_BASE_URL"),
                "API Key" => Some("ANTHROPIC_AUTH_TOKEN"),
                _ => None,
            };
            let top_key = match field_key {
                "Model" => Some("model"),
                _ => None,
            };

            if let Some(ek) = env_key {
                let obj = parsed.as_object_mut().context("root is not object")?;
                let env = obj.entry("env").or_insert_with(|| serde_json::json!({}));
                if let Some(env_obj) = env.as_object_mut() {
                    if new_val == "—" || new_val.is_empty() {
                        env_obj.remove(ek);
                    } else {
                        env_obj.insert(
                            ek.to_string(),
                            serde_json::Value::String(new_val.to_string()),
                        );
                    }
                }
            } else if let Some(tk) = top_key {
                if let Some(obj) = parsed.as_object_mut() {
                    if new_val == "—" || new_val.is_empty() {
                        obj.remove(tk);
                    } else {
                        obj.insert(
                            tk.to_string(),
                            serde_json::Value::String(new_val.to_string()),
                        );
                    }
                }
            } else {
                return Ok(());
            }
        }
        Tool::OpenClaw => {
            // Parse "provider_name ▸ Base URL" or "provider_name ▸ API Key"
            if let Some(sep_pos) = field_key.find(" ▸ ") {
                let provider_name = &field_key[..sep_pos];
                let sub_field = &field_key[sep_pos + " ▸ ".len()..];

                if sub_field == "Models" {
                    // Rename model key in agents.defaults.models
                    // old_display: "claude-opus-4-5-20251101" → full key: "provider/claude-opus-4-5-20251101"
                    // new_val:     "claude-opus-4-6"          → full key: "provider/claude-opus-4-6"
                    let defaults = parsed
                        .as_object_mut()
                        .context("root not object")?
                        .entry("agents")
                        .or_insert_with(|| serde_json::json!({}))
                        .as_object_mut()
                        .context("agents not object")?
                        .entry("defaults")
                        .or_insert_with(|| serde_json::json!({}))
                        .as_object_mut()
                        .context("defaults not object")?;

                    // Collect old→new key mappings first
                    let prefix = format!("{}/", provider_name);
                    let mut renames: Vec<(String, String)> = Vec::new();

                    if let Some(models_obj) =
                        defaults.get_mut("models").and_then(|m| m.as_object_mut())
                    {
                        let old_keys: Vec<String> = models_obj
                            .keys()
                            .filter(|k| k.starts_with(&prefix))
                            .cloned()
                            .collect();

                        for old_key in old_keys {
                            if let Some(val) = models_obj.remove(&old_key) {
                                let new_full = format!("{}/{}", provider_name, new_val);
                                models_obj.insert(new_full.clone(), val);
                                renames.push((old_key, new_full));
                            }
                        }
                    }

                    // Sync model.primary if it pointed to an old key
                    if let Some(model_obj) =
                        defaults.get_mut("model").and_then(|m| m.as_object_mut())
                    {
                        for (old_key, new_full) in &renames {
                            if model_obj.get("primary").and_then(|v| v.as_str()) == Some(old_key) {
                                model_obj.insert(
                                    "primary".to_string(),
                                    serde_json::Value::String(new_full.clone()),
                                );
                            }
                        }
                    }
                } else {
                    let json_field = match sub_field {
                        "Base URL" => "baseUrl",
                        "API Key" => "apiKey",
                        _ => return Ok(()),
                    };

                    let providers = parsed
                        .as_object_mut()
                        .context("root not object")?
                        .entry("models")
                        .or_insert_with(|| serde_json::json!({}))
                        .as_object_mut()
                        .context("models not object")?
                        .entry("providers")
                        .or_insert_with(|| serde_json::json!({}))
                        .as_object_mut()
                        .context("providers not object")?;

                    let prov = providers
                        .entry(provider_name)
                        .or_insert_with(|| serde_json::json!({}));

                    if let Some(obj) = prov.as_object_mut() {
                        if new_val == "—" || new_val.is_empty() {
                            obj.remove(json_field);
                        } else {
                            obj.insert(
                                json_field.to_string(),
                                serde_json::Value::String(new_val.to_string()),
                            );
                        }
                    }
                }
            } else if field_key == "Primary Model" {
                let model = parsed
                    .as_object_mut()
                    .context("root not object")?
                    .entry("agents")
                    .or_insert_with(|| serde_json::json!({}))
                    .as_object_mut()
                    .context("agents not object")?
                    .entry("defaults")
                    .or_insert_with(|| serde_json::json!({}))
                    .as_object_mut()
                    .context("defaults not object")?
                    .entry("model")
                    .or_insert_with(|| serde_json::json!({}));

                if let Some(obj) = model.as_object_mut() {
                    if new_val == "—" || new_val.is_empty() {
                        obj.remove("primary");
                    } else {
                        obj.insert(
                            "primary".to_string(),
                            serde_json::Value::String(new_val.to_string()),
                        );
                    }
                }
            } else {
                return Ok(());
            }
        }
        Tool::Codex | Tool::Kimi => {
            unreachable!("TOML-backed tools are handled before JSON parsing")
        }
    }

    let output = serde_json::to_string_pretty(&parsed).context("serialize config")?;
    if is_jsonc_path(&path) {
        log::info!(
            "Note: {} comments will be removed when Kaku rewrites this file.",
            path.display()
        );
    }
    write_atomic(&path, output.as_bytes()).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Save a field to Codex TOML config (~/.codex/config.toml)
fn save_codex_field(field_key: &str, new_val: &str) -> anyhow::Result<()> {
    let path = Tool::Codex.config_path();
    save_codex_field_at(&path, field_key, new_val)
}

fn save_kimi_field(field_key: &str, new_val: &str) -> anyhow::Result<()> {
    let path = Tool::Kimi.config_path();
    save_kimi_field_at(&path, field_key, new_val)
}

fn save_codex_field_at(path: &Path, field_key: &str, new_val: &str) -> anyhow::Result<()> {
    let toml_key = match field_key {
        "Model" => "model",
        "Reasoning Effort" => "model_reasoning_effort",
        _ => return Ok(()),
    };

    let raw = if path.exists() {
        std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?
    } else {
        String::new()
    };

    let mut lines: Vec<String> = raw.lines().map(|l| l.to_string()).collect();
    let target = format!("{} = ", toml_key);
    let new_line = format!("{} = \"{}\"", toml_key, new_val);

    let mut found = false;
    let mut in_top_level = true;
    for line in &mut lines {
        let trimmed = line.trim_start();
        // Entering a table section: [section] or [[array-of-tables]]
        if trimmed.starts_with('[') {
            in_top_level = false;
        }
        if in_top_level && trimmed.starts_with(&target) {
            if new_val == "—" || new_val.is_empty() {
                *line = String::new();
            } else {
                *line = new_line.clone();
            }
            found = true;
            break;
        }
    }

    if !found && !new_val.is_empty() && new_val != "—" {
        // Insert before the first [section] or at the end
        let insert_pos = lines
            .iter()
            .position(|l| l.trim_start().starts_with('['))
            .unwrap_or(lines.len());
        lines.insert(insert_pos, new_line);
    }

    // Remove empty lines that resulted from deletion
    let output: Vec<&str> = lines.iter().map(|l| l.as_str()).collect();
    let result = output.join("\n");
    write_atomic(path, result.as_bytes()).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn save_kimi_field_at(path: &Path, field_key: &str, new_val: &str) -> anyhow::Result<()> {
    let toml_key = match field_key {
        "Model" => "default_model",
        _ => return Ok(()),
    };

    let raw = if path.exists() {
        std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?
    } else {
        String::new()
    };

    let mut lines: Vec<String> = raw.lines().map(|line| line.to_string()).collect();
    let target = format!("{toml_key} = ");
    let new_line = format!("{toml_key} = \"{new_val}\"");

    let mut found = false;
    let mut in_top_level = true;
    for line in &mut lines {
        let trimmed = line.trim_start();
        // Entering a table section: [section] or [[array-of-tables]]
        if trimmed.starts_with('[') {
            in_top_level = false;
        }
        if in_top_level && trimmed.starts_with(&target) {
            if new_val == "—" || new_val.is_empty() {
                *line = String::new();
            } else {
                *line = new_line.clone();
            }
            found = true;
            break;
        }
    }

    if !found && !new_val.is_empty() && new_val != "—" {
        let insert_pos = lines
            .iter()
            .position(|line| line.trim_start().starts_with('['))
            .unwrap_or(lines.len());
        lines.insert(insert_pos, new_line);
    }

    let output: Vec<&str> = lines.iter().map(|line| line.as_str()).collect();
    let result = output.join("\n");
    write_atomic(path, result.as_bytes()).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

pub fn run() -> anyhow::Result<()> {
    struct TerminalGuard {
        raw_mode: bool,
        bracketed_paste: bool,
        alternate_screen: bool,
    }

    impl TerminalGuard {
        fn new() -> Self {
            Self {
                raw_mode: false,
                bracketed_paste: false,
                alternate_screen: false,
            }
        }
    }

    impl Drop for TerminalGuard {
        fn drop(&mut self) {
            if self.raw_mode {
                let _ = disable_raw_mode();
            }

            let mut stdout = io::stdout();
            if self.bracketed_paste {
                let _ = stdout.execute(DisableBracketedPaste);
            }
            if self.alternate_screen {
                let _ = stdout.execute(LeaveAlternateScreen);
            }
        }
    }

    let mut guard = TerminalGuard::new();
    enable_raw_mode().context("enable raw mode")?;
    guard.raw_mode = true;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, EnableBracketedPaste).context("enable bracketed paste")?;
    guard.bracketed_paste = true;
    stdout
        .execute(EnterAlternateScreen)
        .context("enter alternate screen")?;
    guard.alternate_screen = true;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("create terminal")?;

    terminal
        .draw(ui::loading_ui)
        .context("draw loading screen")?;

    let mut app = App::new();
    let result = run_loop(&mut terminal, &mut app);

    terminal.show_cursor().context("show cursor")?;

    result
}

fn is_confirm_key(code: KeyCode) -> bool {
    matches!(
        code,
        KeyCode::Enter | KeyCode::Char('\n') | KeyCode::Char('\r') | KeyCode::Char(' ')
    )
}

fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> anyhow::Result<()> {
    let mut needs_redraw = true;
    loop {
        if app.sync_transient_errors() {
            needs_redraw = true;
        }

        if needs_redraw {
            terminal.draw(|frame| ui::ui(frame, app))?;
            needs_redraw = false;
        }

        if !event::poll(EVENT_POLL_INTERVAL).context("poll event")? {
            continue;
        }

        needs_redraw = true;
        match event::read().context("read event")? {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                app.status_msg = None;
                app.status_expire = None;
                if app.is_selecting() {
                    match key.code {
                        code if is_confirm_key(code) => app.confirm_select(),
                        KeyCode::Esc => app.cancel_select(),
                        KeyCode::Up | KeyCode::Char('k') => app.move_select_up(),
                        KeyCode::Down | KeyCode::Char('j') => app.move_select_down(),
                        _ => {}
                    }
                    continue;
                }

                if app.is_editing() {
                    match key.code {
                        KeyCode::Enter | KeyCode::Char('\n') | KeyCode::Char('\r') => {
                            app.confirm_edit()
                        }
                        KeyCode::Esc => app.cancel_edit(),
                        KeyCode::Left => {
                            if let Some((edit_buf, edit_cursor)) = app.editing_mut() {
                                if *edit_cursor > 0 {
                                    *edit_cursor = prev_char_boundary(edit_buf, *edit_cursor);
                                }
                            }
                        }
                        KeyCode::Right => {
                            if let Some((edit_buf, edit_cursor)) = app.editing_mut() {
                                if *edit_cursor < edit_buf.len() {
                                    *edit_cursor = next_char_boundary(edit_buf, *edit_cursor);
                                }
                            }
                        }
                        KeyCode::Home => {
                            if let Some((_, edit_cursor)) = app.editing_mut() {
                                *edit_cursor = 0;
                            }
                        }
                        KeyCode::End => {
                            if let Some((edit_buf, edit_cursor)) = app.editing_mut() {
                                *edit_cursor = edit_buf.len();
                            }
                        }
                        KeyCode::Backspace => {
                            if key.modifiers.contains(KeyModifiers::CONTROL)
                                || key.modifiers.contains(KeyModifiers::SUPER)
                            {
                                // Cmd+Backspace (macOS) or Ctrl+Backspace - clear all
                                if let Some((edit_buf, edit_cursor)) = app.editing_mut() {
                                    edit_buf.clear();
                                    *edit_cursor = 0;
                                }
                            } else if let Some((edit_buf, edit_cursor)) = app.editing_mut() {
                                if *edit_cursor > 0 {
                                    edit_backspace(edit_buf, edit_cursor);
                                }
                            }
                        }
                        KeyCode::Delete => {
                            if let Some((edit_buf, edit_cursor)) = app.editing_mut() {
                                if *edit_cursor < edit_buf.len() {
                                    edit_delete(edit_buf, *edit_cursor);
                                }
                            }
                        }
                        KeyCode::Char(c) => {
                            // Handle Ctrl+U (clear line) - macOS Cmd+Backspace may also send this
                            if (key.modifiers.contains(KeyModifiers::CONTROL)
                                || key.modifiers.contains(KeyModifiers::SUPER))
                                && c == 'u'
                            {
                                if let Some((edit_buf, edit_cursor)) = app.editing_mut() {
                                    edit_buf.clear();
                                    *edit_cursor = 0;
                                }
                            }
                            // Ignore other control characters
                            else if !key.modifiers.contains(KeyModifiers::CONTROL)
                                && !key.modifiers.contains(KeyModifiers::SUPER)
                            {
                                if let Some((edit_buf, edit_cursor)) = app.editing_mut() {
                                    edit_insert_char(edit_buf, edit_cursor, c);
                                }
                            }
                        }
                        _ => {}
                    }
                    continue;
                }

                match key.code {
                    KeyCode::Esc => app.should_quit = true,
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        app.should_quit = true
                    }
                    KeyCode::Up | KeyCode::Char('k') => app.move_up(),
                    KeyCode::Down | KeyCode::Char('j') => app.move_down(),
                    code if is_confirm_key(code) => app.start_edit(),
                    KeyCode::Char('o') => {
                        if let Some(path) = app.config_path_to_open() {
                            match with_terminal_suspended(terminal, || open_path_in_editor(&path)) {
                                Ok(()) => app.set_status("Opened config file"),
                                Err(err) => {
                                    log::debug!("Failed to open config file: {}", err);
                                    app.set_status(format!("Failed to open: {}", err));
                                }
                            }
                        }
                    }
                    KeyCode::Char('r') => app.refresh_models(),
                    _ => {}
                }
            }
            Event::Mouse(mouse) => match mouse.kind {
                MouseEventKind::ScrollUp => app.move_up(),
                MouseEventKind::ScrollDown => app.move_down(),
                _ => {}
            },
            Event::Paste(text) => {
                if !app.is_editing() || text.is_empty() {
                    continue;
                }

                let ends_with_newline = text.ends_with('\n') || text.ends_with('\r');

                // Clipboard paste may include a trailing newline from terminal copy.
                // Strip line breaks so paste doesn't break single-line inputs.
                let cleaned: String = text.chars().filter(|c| *c != '\r' && *c != '\n').collect();
                if !cleaned.is_empty() {
                    if let Some((edit_buf, edit_cursor)) = app.editing_mut() {
                        for c in cleaned.chars() {
                            edit_insert_char(edit_buf, edit_cursor, c);
                        }
                    }
                }

                // If the pasted text (often from voice typing tools) ends with a newline, auto-submit
                if ends_with_newline {
                    app.confirm_edit();
                }
            }
            _ => {}
        }

        if app.should_quit {
            return Ok(());
        }
    }
}

fn with_terminal_suspended<F>(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    func: F,
) -> anyhow::Result<()>
where
    F: FnOnce() -> anyhow::Result<()>,
{
    disable_raw_mode().context("disable raw mode")?;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, DisableBracketedPaste).context("disable bracketed paste")?;
    stdout
        .execute(LeaveAlternateScreen)
        .context("leave alternate screen")?;

    let action_result = func();

    let restore_result = (|| -> anyhow::Result<()> {
        enable_raw_mode().context("enable raw mode")?;
        let mut stdout = io::stdout();
        crossterm::execute!(stdout, EnableBracketedPaste).context("enable bracketed paste")?;
        stdout
            .execute(EnterAlternateScreen)
            .context("enter alternate screen")?;
        terminal.clear().context("clear terminal")?;
        Ok(())
    })();

    action_result.and(restore_result)
}

fn prev_char_boundary(buf: &str, cursor: usize) -> usize {
    let cursor = cursor.min(buf.len());
    if cursor == 0 {
        return 0;
    }
    buf[..cursor]
        .char_indices()
        .next_back()
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn next_char_boundary(buf: &str, cursor: usize) -> usize {
    let cursor = cursor.min(buf.len());
    if cursor >= buf.len() {
        return buf.len();
    }
    cursor + buf[cursor..].chars().next().map_or(0, |c| c.len_utf8())
}

fn edit_backspace(buf: &mut String, cursor: &mut usize) {
    if *cursor == 0 || buf.is_empty() {
        return;
    }
    let prev = prev_char_boundary(buf, *cursor);
    buf.remove(prev);
    *cursor = prev;
}

fn edit_delete(buf: &mut String, cursor: usize) {
    if cursor >= buf.len() || buf.is_empty() {
        return;
    }
    buf.remove(cursor);
}

fn edit_insert_char(buf: &mut String, cursor: &mut usize, c: char) {
    let at = (*cursor).min(buf.len());
    buf.insert(at, c);
    *cursor = at + c.len_utf8();
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
    use tempfile::tempdir;

    fn encode_varint(mut value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if value == 0 {
                return out;
            }
        }
    }

    fn encode_len_delimited(field_number: u64, payload: &[u8]) -> Vec<u8> {
        let mut out = encode_varint((field_number << 3) | 2);
        out.extend(encode_varint(payload.len() as u64));
        out.extend(payload);
        out
    }

    fn encode_int32_state_value(value: i32) -> String {
        let mut out = encode_varint(2 << 3);
        out.extend(encode_varint(value as u64));
        base64::engine::general_purpose::STANDARD.encode(out)
    }

    fn encode_antigravity_unified_state(entries: &[(&str, &str)]) -> String {
        let mut outer = Vec::new();
        for (key, value) in entries {
            let nested_value = encode_len_delimited(1, value.as_bytes());
            let mut entry = encode_len_delimited(1, key.as_bytes());
            entry.extend(encode_len_delimited(2, &nested_value));
            outer.extend(encode_len_delimited(1, &entry));
        }
        base64::engine::general_purpose::STANDARD.encode(outer)
    }

    #[test]
    fn codex_save_round_trip_for_model_and_reasoning_effort() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");

        std::fs::write(
            &path,
            "model = \"old\"\nmodel_reasoning_effort = \"low\"\n\n[projects.\"/tmp\"]\ntrust_level = \"trusted\"\n",
        )
        .expect("seed config");

        save_codex_field_at(&path, "Model", "gpt-5").expect("update model");
        save_codex_field_at(&path, "Reasoning Effort", "high").expect("update effort");
        let saved = std::fs::read_to_string(&path).expect("read config");
        assert!(saved.contains("model = \"gpt-5\""));
        assert!(saved.contains("model_reasoning_effort = \"high\""));
        assert!(saved.contains("[projects.\"/tmp\"]"));

        save_codex_field_at(&path, "Model", "").expect("remove model");
        let saved = std::fs::read_to_string(&path).expect("read config");
        assert!(!saved.contains("model = \"gpt-5\""));
        assert!(saved.contains("model_reasoning_effort = \"high\""));
        assert!(saved.contains("[projects.\"/tmp\"]"));
    }

    #[test]
    fn codex_save_creates_new_top_level_entry_before_sections() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[profiles.default]\nfoo = \"bar\"\n").expect("seed config");

        save_codex_field_at(&path, "Model", "gpt-5").expect("insert model");
        let saved = std::fs::read_to_string(&path).expect("read config");
        let model_pos = saved.find("model = \"gpt-5\"").expect("model line");
        let section_pos = saved.find("[profiles.default]").expect("section");
        assert!(model_pos < section_pos);
    }

    #[test]
    fn unicode_edit_helpers_keep_cursor_on_char_boundaries() {
        let mut buf = "你a好".to_string();
        let mut cursor = buf.len();

        cursor = prev_char_boundary(&buf, cursor);
        assert_eq!(cursor, "你a".len());
        cursor = prev_char_boundary(&buf, cursor);
        assert_eq!(cursor, "你".len());
        cursor = prev_char_boundary(&buf, cursor);
        assert_eq!(cursor, 0);

        cursor = next_char_boundary(&buf, cursor);
        assert_eq!(cursor, "你".len());
        cursor = next_char_boundary(&buf, cursor);
        assert_eq!(cursor, "你a".len());

        edit_insert_char(&mut buf, &mut cursor, '界');
        assert_eq!(buf, "你a界好");
        assert_eq!(cursor, "你a界".len());

        edit_backspace(&mut buf, &mut cursor);
        assert_eq!(buf, "你a好");
        assert_eq!(cursor, "你a".len());

        cursor = prev_char_boundary(&buf, cursor);
        assert_eq!(cursor, "你".len());
        edit_delete(&mut buf, cursor);
        assert_eq!(buf, "你好");
        assert_eq!(cursor, "你".len());
    }

    #[test]
    fn factory_droid_extract_prefers_session_defaults() {
        let parsed = serde_json::json!({
            "model": "legacy-model",
            "reasoningEffort": "low",
            "autonomyMode": "spec",
            "sessionDefaultSettings": {
                "model": "fd-model",
                "reasoningEffort": "medium",
                "autonomyLevel": "auto-low"
            }
        });

        let fields = extract_factory_droid_fields(&parsed);
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].key, "Model");
        assert_eq!(fields[0].value, "fd-model");
        assert!(fields[0].options.is_empty());

        assert_eq!(fields[1].key, "Reasoning Effort");
        assert_eq!(fields[1].value, "medium");
        assert_eq!(
            fields[1].options,
            string_options(&FACTORY_DROID_REASONING_OPTIONS)
        );

        assert_eq!(fields[2].key, "Autonomy Level");
        assert_eq!(fields[2].value, "auto-low");
        assert_eq!(
            fields[2].options,
            string_options(&FACTORY_DROID_AUTONOMY_OPTIONS)
        );
    }

    #[test]
    fn factory_droid_extract_falls_back_to_top_level_and_defaults() {
        let parsed = serde_json::json!({
            "model": "legacy-model",
            "reasoningEffort": "high",
            "autonomyLevel": "spec"
        });

        let fields = extract_factory_droid_fields(&parsed);
        assert_eq!(fields[0].value, "legacy-model");
        assert_eq!(fields[1].value, "high");
        assert_eq!(fields[2].value, "spec");

        let empty_fields = extract_factory_droid_fields(&serde_json::json!({}));
        assert_eq!(empty_fields[0].value, FACTORY_DROID_DEFAULT_MODEL);
        assert_eq!(empty_fields[1].value, FACTORY_DROID_DEFAULT_REASONING);
        assert_eq!(empty_fields[2].value, FACTORY_DROID_DEFAULT_AUTONOMY);
    }

    #[test]
    fn factory_droid_save_writes_session_defaults_and_normalizes_autonomy_key() {
        let mut obj = serde_json::json!({
            "autonomyLevel": "spec"
        })
        .as_object()
        .cloned()
        .expect("object");

        save_factory_droid_field(&mut obj, "Autonomy Level", "auto-high").expect("save autonomy");
        save_factory_droid_field(&mut obj, "Model", "gpt-5.1-codex").expect("save model");
        save_factory_droid_field(&mut obj, "Reasoning Effort", "").expect("clear reasoning effort");

        let session_defaults = obj
            .get("sessionDefaultSettings")
            .and_then(|v| v.as_object())
            .expect("sessionDefaultSettings object");

        assert_eq!(
            session_defaults
                .get("autonomyMode")
                .and_then(|v| v.as_str()),
            Some("auto-high")
        );
        assert_eq!(session_defaults.get("autonomyLevel"), None);
        assert_eq!(
            session_defaults.get("model").and_then(|v| v.as_str()),
            Some("gpt-5.1-codex")
        );
        assert_eq!(session_defaults.get("reasoningEffort"), None);
    }

    #[test]
    fn kaku_assistant_fields_include_enabled_toggle() {
        let fields = extract_kaku_assistant_fields("enabled = true\n");
        assert!(fields.iter().any(|field| field.key == "Enabled"));
        let fields = extract_kaku_assistant_fields("enabled = false\n");
        assert!(fields.iter().any(|field| field.key == "Enabled"));
    }

    #[test]
    fn summarize_kaku_assistant_requires_api_key() {
        let fields = extract_kaku_assistant_fields("enabled = false\n");
        assert_eq!(
            summarize_tool_fields(Tool::KakuAssistant, true, &fields, None),
            Some(format!(
                "Setup required · {}",
                assistant_config::DEFAULT_MODEL
            ))
        );
    }

    #[test]
    fn summarize_kaku_assistant_prefers_status_and_model() {
        let fields = extract_kaku_assistant_fields(
            "enabled = true\napi_key = \"sk-test\"\nmodel = \"gpt-5-mini\"\n",
        );
        assert_eq!(
            summarize_tool_fields(Tool::KakuAssistant, true, &fields, None),
            Some("Ready · gpt-5-mini".into())
        );
    }

    #[test]
    fn summarize_non_usage_tool_uses_auth_and_model() {
        let fields = vec![
            FieldEntry {
                key: "Model".into(),
                value: "gpt-5".into(),
                options: vec![],
                editable: true,
            },
            FieldEntry {
                key: "Auth".into(),
                value: "✓ user@example.com".into(),
                options: vec![],
                editable: false,
            },
        ];

        assert_eq!(
            summarize_tool_fields(Tool::FactoryDroid, true, &fields, None),
            Some("user@example.com · gpt-5".into())
        );
    }

    #[test]
    fn summarize_codex_prefers_usage_only() {
        assert_eq!(
            summarize_tool_fields(
                Tool::Codex,
                true,
                &[FieldEntry {
                    key: "Auth".into(),
                    value: "✓ user@example.com".into(),
                    options: vec![],
                    editable: false,
                }],
                Some("5h remain 75% · reset 4h0m  |  7d remain 93% · reset 6d0h"),
            ),
            Some("5h remain 75% · reset 4h0m  |  7d remain 93% · reset 6d0h".into())
        );
    }

    #[test]
    fn summarize_claude_prefers_usage_only() {
        assert_eq!(
            summarize_tool_fields(
                Tool::ClaudeCode,
                true,
                &[],
                Some("5h remain 94% · reset 2h0m")
            ),
            Some("5h remain 94% · reset 2h0m".into())
        );
    }

    #[test]
    fn summarize_kimi_prefers_usage_only() {
        assert_eq!(
            summarize_tool_fields(
                Tool::Kimi,
                true,
                &[],
                Some("5h remain 94% · reset 14m  |  7d remain 67% · reset 4d11h")
            ),
            Some("5h remain 94% · reset 14m  |  7d remain 67% · reset 4d11h".into())
        );
    }

    #[test]
    fn summarize_antigravity_uses_usage_summary() {
        let fields = vec![
            FieldEntry {
                key: "Plan".into(),
                value: "Google AI Pro".into(),
                options: vec![],
                editable: false,
            },
            FieldEntry {
                key: "Auth".into(),
                value: "✓ hitw93@gmail.com".into(),
                options: vec![],
                editable: false,
            },
        ];

        assert_eq!(
            summarize_tool_fields(
                Tool::Antigravity,
                true,
                &fields,
                Some("remain 60% · reset 3d4h"),
            ),
            Some("remain 60% · reset 3d4h".into())
        );
    }

    #[test]
    fn summarize_antigravity_falls_back_to_sync_message_when_model_present() {
        let fields = vec![FieldEntry {
            key: "Model".into(),
            value: "Gemini 3.1 Pro".into(),
            options: vec![],
            editable: false,
        }];

        assert_eq!(
            summarize_tool_fields(Tool::Antigravity, true, &fields, None),
            Some("Open Antigravity to sync quota".into())
        );
    }

    #[test]
    fn antigravity_unified_state_decodes_int32_credit_values() {
        let available = encode_int32_state_value(128);
        let minimum = encode_int32_state_value(4);
        let raw = encode_antigravity_unified_state(&[
            ("availableCreditsSentinelKey", &available),
            ("minimumCreditAmountForUsageKey", &minimum),
        ]);

        let entries = parse_antigravity_unified_state(&raw).expect("entries");
        let parsed_available = entries
            .iter()
            .find(|(key, _)| key == "availableCreditsSentinelKey")
            .and_then(|(_, value)| decode_antigravity_int32_value(value));
        let parsed_minimum = entries
            .iter()
            .find(|(key, _)| key == "minimumCreditAmountForUsageKey")
            .and_then(|(_, value)| decode_antigravity_int32_value(value));

        assert_eq!(parsed_available, Some(128));
        assert_eq!(parsed_minimum, Some(4));
    }

    #[test]
    fn antigravity_model_label_from_sentinel_value_maps_known_models() {
        assert_eq!(
            antigravity_model_label_from_sentinel_value(37),
            Some("Gemini 3.1 Pro")
        );
        assert_eq!(
            antigravity_model_label_from_sentinel_value(35),
            Some("Claude Sonnet 4.6")
        );
    }

    #[test]
    fn antigravity_model_id_from_sentinel_maps_placeholder_suffix() {
        let model_ids = vec![
            "MODEL_PLACEHOLDER_M37".to_string(),
            "MODEL_PLACEHOLDER_M35".to_string(),
            "MODEL_PLACEHOLDER_M18".to_string(),
        ];
        assert_eq!(
            antigravity_model_id_from_sentinel(1035, &model_ids),
            Some("MODEL_PLACEHOLDER_M35".into())
        );
    }

    #[test]
    fn antigravity_process_info_parses_pid_csrf_and_extension_port() {
        let line = "34643 /Applications/Antigravity.app/Contents/Resources/app/extensions/antigravity/bin/language_server_macos_arm --enable_lsp --csrf_token abc --extension_server_port 56502 --extension_server_csrf_token def --random_port --app_data_dir antigravity";
        let parsed = parse_antigravity_process_info_line(line).expect("process info");
        assert_eq!(parsed.pid, 34643);
        assert_eq!(parsed.csrf_token, "abc");
        assert_eq!(parsed.extension_server_port, Some(56502));
    }

    #[test]
    fn antigravity_process_info_rejects_non_app_server_processes() {
        let line = "34643 /tmp/language_server_macos_arm --enable_lsp --csrf_token abc --extension_server_port 56502";
        assert!(parse_antigravity_process_info_line(line).is_none());
    }

    #[test]
    fn antigravity_unified_state_returns_none_for_malformed_payload() {
        let raw = base64::engine::general_purpose::STANDARD.encode([0x0a, 0x05, 0x01]);
        assert!(parse_antigravity_unified_state(&raw).is_none());
    }

    #[test]
    fn antigravity_usage_snapshot_parses_live_quota_windows() {
        let data = serde_json::json!({
            "user_status": {
                "userStatus": {
                    "cascadeModelConfigData": {
                        "clientModelConfigs": [
                            {
                                "label": "Claude Sonnet 4.6 (Thinking)",
                                "modelOrAlias": {
                                    "model": "claude-sonnet-thinking"
                                },
                                "quotaInfo": {
                                    "remainingFraction": 0.6,
                                    "resetTime": "2100-03-12T07:44:33Z"
                                }
                            },
                            {
                                "label": "Gemini 3 Flash",
                                "quotaInfo": {
                                    "remainingFraction": 1.0
                                }
                            }
                        ]
                    }
                }
            },
            "command_model_configs": {
                "configs": [
                    {
                        "modelId": "claude-sonnet-thinking",
                        "modelDisplayName": "Claude Sonnet 4.6 (Thinking)"
                    }
                ]
            }
        });

        let snapshot = parse_antigravity_usage_snapshot(&data).expect("snapshot");
        assert_eq!(
            snapshot.selected_model_label,
            Some("Claude Sonnet 4.6".into())
        );
        assert!(snapshot
            .summary
            .as_deref()
            .is_some_and(|summary| summary.starts_with("remain 60%")));
    }

    #[test]
    fn antigravity_usage_snapshot_ignores_unrelated_nested_quota_entries() {
        let data = serde_json::json!({
            "user_status": {
                "userStatus": {
                    "cascadeModelConfigData": {
                        "defaultOverrideModelConfig": {
                            "modelOrAlias": {
                                "model": "MODEL_PLACEHOLDER_M18"
                            }
                        },
                        "clientModelConfigs": [
                            {
                                "label": "Gemini 3 Flash",
                                "modelOrAlias": {
                                    "model": "MODEL_PLACEHOLDER_M18"
                                },
                                "quotaInfo": {
                                    "remainingFraction": 1.0
                                }
                            }
                        ]
                    },
                    "someOtherSection": {
                        "history": [
                            {
                                "label": "Gemini 3 Flash",
                                "modelOrAlias": {
                                    "model": "MODEL_PLACEHOLDER_M18"
                                },
                                "quotaInfo": {
                                    "remainingFraction": 0.1
                                }
                            }
                        ]
                    }
                }
            }
        });

        let snapshot = parse_antigravity_usage_snapshot(&data).expect("snapshot");
        assert_eq!(snapshot.selected_model_label, Some("Gemini 3 Flash".into()));
        assert_eq!(snapshot.summary.as_deref(), Some("remain 100%"));
    }

    #[test]
    fn antigravity_usage_snapshot_falls_back_to_first_model_when_unselected() {
        let data = serde_json::json!({
            "user_status": {
                "userStatus": {
                    "cascadeModelConfigData": {
                        "clientModelConfigs": [
                            { "label": "Claude Opus 4.6 (Thinking)", "quotaInfo": { "remainingFraction": 1.0 } },
                            { "label": "Claude Sonnet 4.6 (Thinking)", "quotaInfo": { "remainingFraction": 1.0 } },
                            { "label": "Gemini 3 Flash", "quotaInfo": { "remainingFraction": 1.0 } },
                            { "label": "Gemini 3.1 Pro (High)", "quotaInfo": { "remainingFraction": 1.0 } }
                        ]
                    }
                }
            }
        });

        let snapshot = parse_antigravity_usage_snapshot(&data).expect("snapshot");
        assert_eq!(
            snapshot.selected_model_label,
            Some("Claude Opus 4.6".into())
        );
        assert_eq!(snapshot.summary.as_deref(), Some("remain 100%"));
    }

    #[test]
    fn antigravity_usage_snapshot_falls_back_to_first_non_selected_model() {
        let data = serde_json::json!({
            "user_status": {
                "userStatus": {
                    "cascadeModelConfigData": {
                        "clientModelConfigs": [
                            { "label": "Claude Sonnet 4.6 (Thinking)", "quotaInfo": { "remainingFraction": 0.6 } },
                            { "label": "Gemini 3 Flash", "quotaInfo": { "remainingFraction": 1.0 } },
                            { "label": "GPT-OSS 120B (Medium)", "quotaInfo": { "remainingFraction": 0.6 } }
                        ]
                    }
                }
            }
        });

        let snapshot = parse_antigravity_usage_snapshot(&data).expect("snapshot");
        assert_eq!(
            snapshot.selected_model_label,
            Some("Claude Sonnet 4.6".into())
        );
        assert_eq!(snapshot.summary.as_deref(), Some("remain 60%"));
    }

    #[test]
    fn antigravity_usage_snapshot_prefers_default_override_model_when_available() {
        let data = serde_json::json!({
            "user_status": {
                "userStatus": {
                    "cascadeModelConfigData": {
                        "defaultOverrideModelConfig": {
                            "modelOrAlias": {
                                "model": "MODEL_PLACEHOLDER_M18"
                            }
                        },
                        "clientModelConfigs": [
                            { "label": "Gemini 3 Flash", "modelOrAlias": { "model": "MODEL_PLACEHOLDER_M18" }, "quotaInfo": { "remainingFraction": 1.0 } },
                            { "label": "Claude Sonnet 4.6 (Thinking)", "modelOrAlias": { "model": "MODEL_PLACEHOLDER_M35" }, "quotaInfo": { "remainingFraction": 0.6 } },
                            { "label": "Gemini 3.1 Pro (High)", "modelOrAlias": { "model": "MODEL_PLACEHOLDER_M37" }, "quotaInfo": { "remainingFraction": 1.0 } }
                        ]
                    }
                }
            }
        });

        let snapshot = parse_antigravity_usage_snapshot(&data).expect("snapshot");
        assert_eq!(snapshot.selected_model_label, Some("Gemini 3 Flash".into()));
        assert_eq!(snapshot.summary.as_deref(), Some("remain 100%"));
    }

    #[test]
    fn antigravity_usage_snapshot_prefers_live_default_override_over_stale_sentinel() {
        let data = serde_json::json!({
            "user_status": {
                "userStatus": {
                    "cascadeModelConfigData": {
                        "defaultOverrideModelConfig": {
                            "modelOrAlias": {
                                "model": "MODEL_PLACEHOLDER_M37"
                            }
                        },
                        "clientModelConfigs": [
                            { "label": "Gemini 3.1 Pro (High)", "modelOrAlias": { "model": "MODEL_PLACEHOLDER_M37" }, "quotaInfo": { "remainingFraction": 1.0 } },
                            { "label": "Claude Sonnet 4.6 (Thinking)", "modelOrAlias": { "model": "MODEL_PLACEHOLDER_M35" }, "quotaInfo": { "remainingFraction": 0.6 } }
                        ]
                    }
                }
            }
        });

        let snapshot = parse_antigravity_usage_snapshot(&data).expect("snapshot");
        assert_eq!(snapshot.selected_model_label, Some("Gemini 3.1 Pro".into()));
        assert_eq!(snapshot.summary.as_deref(), Some("remain 100%"));
    }

    #[test]
    fn antigravity_usage_snapshot_prefers_default_override_over_command_model_list() {
        let data = serde_json::json!({
            "user_status": {
                "userStatus": {
                    "cascadeModelConfigData": {
                        "defaultOverrideModelConfig": {
                            "modelOrAlias": {
                                "model": "MODEL_PLACEHOLDER_M37"
                            }
                        },
                        "clientModelConfigs": [
                            { "label": "Gemini 3.1 Pro (High)", "modelOrAlias": { "model": "MODEL_PLACEHOLDER_M37" }, "quotaInfo": { "remainingFraction": 1.0 } },
                            { "label": "Gemini 3 Flash", "modelOrAlias": { "model": "MODEL_PLACEHOLDER_M18" }, "quotaInfo": { "remainingFraction": 1.0 } }
                        ]
                    }
                }
            },
            "command_model_configs": {
                "clientModelConfigs": [
                    { "label": "Gemini 3 Flash", "modelOrAlias": { "model": "MODEL_PLACEHOLDER_M18" } }
                ]
            }
        });

        let snapshot = parse_antigravity_usage_snapshot(&data).expect("snapshot");
        assert_eq!(snapshot.selected_model_label, Some("Gemini 3.1 Pro".into()));
        assert_eq!(snapshot.summary.as_deref(), Some("remain 100%"));
    }

    #[test]
    fn antigravity_usage_snapshot_ignores_ambiguous_command_model_configs() {
        let data = serde_json::json!({
            "user_status": {
                "userStatus": {
                    "cascadeModelConfigData": {
                        "defaultOverrideModelConfig": {
                            "modelOrAlias": {
                                "model": "MODEL_PLACEHOLDER_M37"
                            }
                        },
                        "clientModelConfigs": [
                            { "label": "Gemini 3.1 Pro (High)", "modelOrAlias": { "model": "MODEL_PLACEHOLDER_M37" }, "quotaInfo": { "remainingFraction": 1.0 } },
                            { "label": "Gemini 3 Flash", "modelOrAlias": { "model": "MODEL_PLACEHOLDER_M18" }, "quotaInfo": { "remainingFraction": 1.0 } }
                        ]
                    }
                }
            },
            "command_model_configs": {
                "clientModelConfigs": [
                    { "label": "Gemini 3 Flash", "modelOrAlias": { "model": "MODEL_PLACEHOLDER_M18" } },
                    { "label": "Gemini 3.1 Pro (High)", "modelOrAlias": { "model": "MODEL_PLACEHOLDER_M37" } }
                ]
            }
        });

        let snapshot = parse_antigravity_usage_snapshot(&data).expect("snapshot");
        assert_eq!(snapshot.selected_model_label, Some("Gemini 3.1 Pro".into()));
        assert_eq!(snapshot.summary.as_deref(), Some("remain 100%"));
    }

    #[test]
    fn antigravity_usage_snapshot_prefers_explicit_command_selected_model() {
        let data = serde_json::json!({
            "user_status": {
                "userStatus": {
                    "cascadeModelConfigData": {
                        "defaultOverrideModelConfig": {
                            "modelOrAlias": {
                                "model": "MODEL_PLACEHOLDER_M37"
                            }
                        },
                        "clientModelConfigs": [
                            { "label": "Gemini 3.1 Pro (High)", "modelOrAlias": { "model": "MODEL_PLACEHOLDER_M37" }, "quotaInfo": { "remainingFraction": 1.0 } },
                            { "label": "Gemini 3 Flash", "modelOrAlias": { "model": "MODEL_PLACEHOLDER_M18" }, "quotaInfo": { "remainingFraction": 0.8 } }
                        ]
                    }
                }
            },
            "command_model_configs": {
                "selectedModelConfig": {
                    "modelOrAlias": {
                        "model": "MODEL_PLACEHOLDER_M18"
                    }
                },
                "clientModelConfigs": [
                    { "label": "Gemini 3 Flash", "modelOrAlias": { "model": "MODEL_PLACEHOLDER_M18" } },
                    { "label": "Gemini 3.1 Pro (High)", "modelOrAlias": { "model": "MODEL_PLACEHOLDER_M37" } }
                ]
            }
        });

        let snapshot = parse_antigravity_usage_snapshot(&data).expect("snapshot");
        assert_eq!(snapshot.selected_model_label, Some("Gemini 3 Flash".into()));
        assert_eq!(snapshot.summary.as_deref(), Some("remain 80%"));
    }

    #[test]
    fn extract_antigravity_fields_include_model_and_auth_only() {
        let snapshot = AntigravityUsageSnapshot {
            summary: Some("remain 60%".into()),
            selected_model_label: Some("Claude Sonnet 4.6".into()),
        };

        let fields = extract_antigravity_fields(Some(&snapshot));
        assert!(fields
            .iter()
            .any(|field| field.key == "Model" && field.value == "Claude Sonnet 4.6"));
        assert!(fields.iter().all(|field| field.key != "Quota"));
        assert!(fields
            .iter()
            .all(|field| !field.key.starts_with("Quota · ")));
        assert!(fields.iter().all(|field| field.key != "Prompt Credits"));
        assert!(fields.iter().all(|field| field.key != "Flow Credits"));
        assert!(fields.iter().all(|field| field.key != "Plan"));
    }

    #[test]
    fn antigravity_model_click_shows_toast_instead_of_editing() {
        let mut app = App {
            tools: vec![ToolState {
                tool: Tool::Antigravity,
                installed: true,
                fields: vec![FieldEntry {
                    key: "Model".into(),
                    value: "Gemini 3.1 Pro".into(),
                    options: vec![],
                    editable: false,
                }],
                summary: Some("remain 100%".into()),
            }],
            tool_index: 0,
            field_index: 0,
            assistant_collapsed: false,
            usage_update_rx: None,
            usage_update_tx: None,
            usage_update_generation: Arc::new(AtomicUsize::new(0)),
            focus: Focus::ToolList,
            mode: AppMode::Browsing,
            status_msg: None,
            status_expire: None,
            toast_msg: None,
            toast_expire: None,
            last_error: None,
            error_expire: None,
            should_quit: false,
        };

        app.start_edit();

        assert_eq!(
            app.toast_message(),
            Some("Change the model in Antigravity settings.")
        );
        assert!(!app.is_editing());
        assert!(!app.is_selecting());
    }

    #[test]
    fn confirm_key_accepts_space_and_enter() {
        assert!(is_confirm_key(KeyCode::Char(' ')));
        assert!(is_confirm_key(KeyCode::Enter));
    }

    #[test]
    fn antigravity_plan_name_extracts_google_ai_tier() {
        let raw = base64::engine::general_purpose::STANDARD
            .encode(b"\0\0Google AI Pro\0ignored model text");
        assert_eq!(
            extract_antigravity_plan_name(&raw),
            Some("Google AI Pro".into())
        );
    }

    #[test]
    fn summarize_gemini_falls_back_to_auth_and_model_when_quota_missing() {
        let fields = vec![
            FieldEntry {
                key: "Model".into(),
                value: "gemini-3.1-flash-lite-preview".into(),
                options: vec![],
                editable: true,
            },
            FieldEntry {
                key: "Auth".into(),
                value: "✓ user@example.com".into(),
                options: vec![],
                editable: false,
            },
        ];

        assert_eq!(
            summarize_tool_fields(Tool::Gemini, true, &fields, None),
            Some("user@example.com · gemini-3.1-flash-lite-preview".into())
        );
    }

    #[test]
    fn claude_usage_snapshot_shows_reauth_required_for_invalid_refresh_state() {
        let parsed = serde_json::json!({
            "type": "error",
            "error": {
                "type": "authentication_error",
                "details": {
                    "error_code": "token_expired"
                }
            }
        });

        let snapshot = parse_claude_usage_snapshot(&parsed).expect("snapshot");
        assert_eq!(snapshot.summary, Some("Re-auth required".into()));
    }

    #[test]
    fn parse_claude_usage_error_detects_auth_failure() {
        let parsed = serde_json::json!({
            "type": "error",
            "error": {
                "type": "authentication_error",
                "details": {
                    "error_code": "token_expired"
                }
            }
        });

        assert_eq!(
            parse_claude_usage_error(&parsed).as_deref(),
            Some("Re-auth required")
        );
    }

    #[test]
    fn parse_claude_keychain_account_extracts_account_name() {
        let raw = r#"
keychain: "/Users/tw93/Library/Keychains/login.keychain-db"
version: 512
class: "genp"
attributes:
    "acct"<blob>="tw93"
    "svce"<blob>="Claude Code-credentials"
"#;

        assert_eq!(parse_claude_keychain_account(raw), Some("tw93".into()));
    }

    #[test]
    fn gemini_quota_summary_uses_auth_type() {
        let parsed = serde_json::json!({
            "security": {
                "auth": {
                    "selectedType": "oauth-personal"
                }
            }
        });
        assert_eq!(
            gemini_quota_summary(&parsed),
            Some("Quota 1000/day · 60/min".into())
        );
    }

    #[test]
    fn copilot_usage_snapshot_prefers_remaining_count() {
        let parsed = serde_json::json!({
            "quota_reset_date_utc": "2100-04-01T00:00:00.000Z",
            "quota_snapshots": {
                "premium_interactions": {
                    "remaining": 298,
                    "quota_remaining": 298.34
                }
            }
        });
        let snapshot = parse_copilot_usage_snapshot(&parsed).expect("snapshot");
        let summary = snapshot.summary.expect("summary");
        assert!(summary.starts_with("298 left this month · reset "));
    }

    #[test]
    fn codex_extract_adds_default_model_when_missing() {
        let fields = extract_codex_fields("");
        let model = fields
            .iter()
            .find(|field| field.key == "Model")
            .expect("model field");
        assert!(!model.value.is_empty());
        assert!(!model.value.ends_with(" (default)"));
    }

    #[test]
    fn kimi_extract_reads_model_and_auth_field() {
        let fields = extract_kimi_fields(
            r#"
default_model = "kimi-code/kimi-for-coding"

[models."kimi-code/kimi-for-coding"]
provider = "managed:kimi-code"
"#,
        );
        assert_eq!(fields[0].key, "Model");
        assert_eq!(fields[0].value, "kimi-code/kimi-for-coding");
        assert_eq!(fields[1].key, "Auth");
    }

    #[test]
    fn kimi_save_round_trip_for_default_model() {
        let path = std::env::temp_dir().join(format!(
            "kaku-kimi-{}.toml",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("unix epoch")
                .as_nanos()
        ));
        std::fs::write(
            &path,
            "default_model = \"kimi-code/kimi-for-coding\"\n[models.\"kimi-code/kimi-for-coding\"]\nprovider = \"managed:kimi-code\"\n",
        )
        .expect("write temp config");

        save_kimi_field_at(&path, "Model", "kimi-code/new-model").expect("save model");
        let updated = std::fs::read_to_string(&path).expect("read temp config");
        assert!(updated.contains("default_model = \"kimi-code/new-model\""));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn kimi_usage_snapshot_formats_current_and_weekly_windows() {
        let parsed = serde_json::json!({
            "usage": {
                "limit": "100",
                "used": "33",
                "remaining": "67",
                "resetTime": "2100-03-12T14:43:16.853067Z"
            },
            "limits": [
                {
                    "window": {
                        "duration": 300,
                        "timeUnit": "TIME_UNIT_MINUTE"
                    },
                    "detail": {
                        "limit": "100",
                        "used": "6",
                        "remaining": "94",
                        "resetTime": "2100-03-08T03:43:16.853067Z"
                    }
                }
            ]
        });

        let snapshot = parse_kimi_usage_snapshot(&parsed).expect("snapshot");
        let summary = snapshot.summary.expect("summary");
        assert!(summary.starts_with("5h remain 94% · reset "));
        assert!(summary.contains("  |  7d remain 67% · reset "));
    }

    #[test]
    fn kimi_auth_status_accepts_refreshable_session() {
        let auth = serde_json::json!({
            "access_token": "",
            "refresh_token": "refresh-token",
            "expires_at": 0.0
        });
        let dir = tempdir().expect("tempdir");
        let path = dir
            .path()
            .join(".kimi")
            .join("credentials")
            .join("kimi-code.json");
        let parent = path.parent().expect("credentials dir");
        std::fs::create_dir_all(parent).expect("create credentials dir");
        std::fs::write(&path, serde_json::to_vec(&auth).expect("serialize auth"))
            .expect("write credentials");

        assert_eq!(read_kimi_auth_status_from_path(&path), "✓ oauth");
    }

    #[test]
    fn claude_auth_status_prefers_email_when_logged_in() {
        let parsed = serde_json::json!({
            "loggedIn": true,
            "authMethod": "claude.ai",
            "email": "user@example.com",
        });

        assert_eq!(
            parse_claude_auth_status(&parsed),
            Some("✓ user@example.com".into())
        );
    }

    #[test]
    fn claude_auth_status_reports_signed_out() {
        let parsed = serde_json::json!({
            "loggedIn": false,
            "authMethod": "claude.ai",
        });

        assert_eq!(
            parse_claude_auth_status(&parsed),
            Some("✗ not signed in".into())
        );
    }

    #[test]
    fn kaku_assistant_fields_include_provider_dropdown() {
        let fields = extract_kaku_assistant_fields(
            "enabled = true\napi_key = \"sk-test\"\nmodel = \"gpt-5.4-mini\"\nbase_url = \"https://api.openai.com/v1\"\n",
        );
        let provider = fields
            .iter()
            .find(|f| f.key == "Provider")
            .expect("Provider field");
        assert_eq!(provider.value, "OpenAI");
        assert!(provider.options.contains(&"OpenAI".to_string()));
        assert!(provider.options.contains(&"Custom".to_string()));

        let model = fields
            .iter()
            .find(|f| f.key == "Model")
            .expect("Model field");
        assert_eq!(model.value, "gpt-5.4-mini");
    }

    #[test]
    fn kaku_assistant_auto_detects_provider_from_base_url() {
        let fields = extract_kaku_assistant_fields(
            "enabled = true\nmodel = \"gpt-5.4-mini\"\nbase_url = \"https://api.openai.com/v1\"\n",
        );
        let provider = fields
            .iter()
            .find(|f| f.key == "Provider")
            .expect("Provider field");
        assert_eq!(provider.value, "OpenAI");
    }

    #[test]
    fn kaku_assistant_provider_defaults_to_openai() {
        let fields = extract_kaku_assistant_fields("enabled = true\n");
        let provider = fields
            .iter()
            .find(|f| f.key == "Provider")
            .expect("Provider field");
        assert_eq!(provider.value, "OpenAI");
    }

    #[test]
    fn kaku_assistant_custom_url_sets_custom_provider() {
        let fields = extract_kaku_assistant_fields(
            "enabled = true\nbase_url = \"https://my-proxy.example.com/v1\"\n",
        );
        let provider = fields
            .iter()
            .find(|f| f.key == "Provider")
            .expect("Provider field");
        assert_eq!(provider.value, "Custom");

        let model = fields
            .iter()
            .find(|f| f.key == "Model")
            .expect("Model field");
        assert!(
            model.options.is_empty(),
            "Custom provider should have no model presets"
        );
    }

    #[test]
    fn kaku_assistant_save_provider_updates_base_url_and_model() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("assistant.toml");
        std::fs::write(
            &path,
            "enabled = true\nmodel = \"gpt-5.4-mini\"\nbase_url = \"https://api.openai.com/v1\"\n",
        )
        .expect("write temp config");

        // Parse, change to Custom provider, and write
        let raw = std::fs::read_to_string(&path).expect("read");
        let cfg = parse_kaku_assistant_config(&raw);
        let updated = KakuAssistantConfig::new(
            cfg.is_enabled(),
            cfg.api_key(),
            "my-model",
            "https://my-proxy.example.com/v1",
        )
        .with_provider("Custom")
        .with_custom_headers(cfg.custom_headers().to_vec());
        write_kaku_assistant_config(&path, &updated).expect("write config");

        let saved = std::fs::read_to_string(&path).expect("read saved");
        assert!(saved.contains("model = \"my-model\""));
        assert!(saved.contains("base_url = \"https://my-proxy.example.com/v1\""));
    }

    #[test]
    fn kaku_assistant_provider_round_trip_preserves_headers() {
        let raw = "enabled = true\napi_key = \"sk-test\"\nmodel = \"gpt-5.4-mini\"\nbase_url = \"https://api.openai.com/v1\"\ncustom_headers = [\"X-Foo: bar\"]\n";
        let cfg = parse_kaku_assistant_config(raw);
        assert_eq!(cfg.provider(), "OpenAI");
        assert_eq!(cfg.custom_headers(), &["X-Foo: bar"]);

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("assistant.toml");
        write_kaku_assistant_config(&path, &cfg).expect("write config");
        let saved = std::fs::read_to_string(&path).expect("read saved");
        // The AI provider preset (e.g. "OpenAI") must not be persisted because
        // it is derived from base_url at load time, not a stored field.
        // Check only uncommented lines to avoid tripping on commented-out web_search_provider.
        assert!(
            !saved.lines().any(|l| {
                // Strip comments: only look at the text before the first '#'.
                l.split('#').next().unwrap_or("").contains("provider =")
            }),
            "provider preset must not be written to TOML"
        );
        assert!(saved.contains("custom_headers = [\"X-Foo: bar\"]"));
    }

    #[test]
    fn codex_usage_snapshot_prefers_current_window() {
        let parsed = serde_json::json!({
            "rate_limit": {
                "primary_window": {
                    "used_percent": 62.7,
                    "reset_at": 4_102_444_800i64
                },
                "secondary_window": {
                    "used_percent": 18.0,
                    "reset_at": 4_102_704_000i64
                }
            }
        });

        let snapshot = parse_codex_usage_snapshot(&parsed).expect("snapshot");
        let summary = snapshot.summary.expect("summary");
        assert!(summary.starts_with("5h remain 37.3% · reset "));
        assert!(summary.contains("  |  7d remain 82% · reset "));
    }
}
