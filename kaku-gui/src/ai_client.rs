//! AI client for Kaku's built-in chat overlay.
//!
//! Reads API config from `~/.config/kaku/assistant.toml` and provides
//! a synchronous streaming chat completion client (OpenAI-compatible API).
//! Supports function/tool calling for agentic workflows.
//!
//! Runs on a plain OS thread (inside overlay), so blocking I/O is fine.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

const DEFAULT_MODEL: &str = "gpt-5.4-mini";
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Configuration loaded from `assistant.toml`.
#[derive(Clone)]
pub struct AssistantConfig {
    pub api_key: String,
    /// Chat overlay model. Falls back to `model` from assistant.toml when omitted.
    pub chat_model: String,
    /// Optional user-curated model list for the chat overlay. When set, the chat
    /// overlay cycles only through these via Shift+Tab and skips the auto-fetch step.
    pub chat_model_choices: Vec<String>,
    pub base_url: String,
    /// When false, the `tools` field is omitted from chat requests.
    /// Set `chat_tools_enabled = false` in assistant.toml for providers that do not
    /// support function calling (e.g. some Kimi or local-model variants).
    pub chat_tools_enabled: bool,
    /// Web search provider: "brave", "pipellm", or "tavily". None = disabled.
    pub web_search_provider: Option<String>,
    /// API key for web_search_provider. None = search tool not registered.
    pub web_search_api_key: Option<String>,
    /// Hidden escape hatch: path to a custom fetch script (not in TUI or template).
    /// Script receives the URL as $1 and must print Markdown to stdout.
    pub web_fetch_script: Option<String>,
    /// Optional dedicated model for background memory curation. Falls back to
    /// `chat_model` when unset. Point at a cheaper/faster model to reduce cost.
    pub memory_curator_model: Option<String>,
}

impl AssistantConfig {
    pub fn load() -> Result<Self> {
        let path = assistant_toml_path()?;
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("Cannot read {}", path.display()))?;
        let parsed: toml::Value = raw.parse().context("Invalid assistant.toml")?;

        let api_key = parsed
            .get("api_key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "api_key not set in {}. Run `kaku ai` to configure.",
                    path.display()
                )
            })?
            .to_string();

        let model = parsed
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_MODEL)
            .to_string();

        // chat_model defaults to `model` to preserve current behavior for users
        // who haven't set an explicit chat model.
        let chat_model = parsed
            .get("chat_model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| model.clone());

        let chat_model_choices = parsed
            .get("chat_model_choices")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let base_url = parsed
            .get("base_url")
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_BASE_URL)
            .trim_end_matches('/')
            .to_string();

        let chat_tools_enabled = parsed
            .get("chat_tools_enabled")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let web_search_provider = parsed
            .get("web_search_provider")
            .and_then(|v| v.as_str())
            .filter(|s| matches!(*s, "brave" | "pipellm" | "tavily"))
            .map(String::from);

        let web_search_api_key = parsed
            .get("web_search_api_key")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(String::from);

        let web_fetch_script = parsed
            .get("web_fetch_script")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| expand_tilde(s));

        let memory_curator_model = parsed
            .get("memory_curator_model")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(String::from);

        Ok(Self {
            api_key,
            chat_model,
            chat_model_choices,
            base_url,
            chat_tools_enabled,
            web_search_provider,
            web_search_api_key,
            web_fetch_script,
            memory_curator_model,
        })
    }

    /// Returns true when a web_search provider and its API key are both configured.
    pub fn web_search_ready(&self) -> bool {
        self.web_search_provider.is_some() && self.web_search_api_key.is_some()
    }
}

fn expand_tilde(s: &str) -> String {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return Path::new(&home).join(rest).to_string_lossy().into_owned();
        }
    }
    s.to_string()
}

fn assistant_toml_path() -> Result<PathBuf> {
    let user_config_path = config::user_config_path();
    let config_dir = user_config_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("invalid user config path"))?;
    Ok(config_dir.join("assistant.toml"))
}

// ─── Message types ────────────────────────────────────────────────────────────

/// A single message in API format. Stored as a raw JSON value so it can represent
/// any role (system, user, assistant, tool) including tool_calls and tool results.
#[derive(Clone)]
pub struct ApiMessage(pub serde_json::Value);

impl ApiMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self(serde_json::json!({ "role": "system", "content": content.into() }))
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self(serde_json::json!({ "role": "user", "content": content.into() }))
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self(serde_json::json!({ "role": "assistant", "content": content.into() }))
    }
    /// Assistant turn that requested tool calls (content is null per the OpenAI spec).
    pub fn assistant_tool_calls(tool_calls: serde_json::Value) -> Self {
        Self(serde_json::json!({
            "role": "assistant",
            "content": null,
            "tool_calls": tool_calls
        }))
    }
    /// Tool result message returned after executing a function call.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self(serde_json::json!({
            "role": "tool",
            "tool_call_id": tool_call_id.into(),
            "content": content.into()
        }))
    }
}

// ─── Tool calling ─────────────────────────────────────────────────────────────

/// A fully assembled tool call returned by the model after streaming is complete.
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Complete JSON-encoded arguments string, e.g. `{"path": "~/Downloads"}`.
    pub arguments: String,
}

// ─── Client ───────────────────────────────────────────────────────────────────

/// Synchronous AI client for use inside overlay threads.
/// Clone is cheap: reqwest::blocking::Client is Arc-backed internally.
#[derive(Clone)]
pub struct AiClient {
    config: AssistantConfig,
    client: reqwest::blocking::Client,
}

impl AiClient {
    pub fn new(config: AssistantConfig) -> Self {
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .unwrap_or_else(|e| {
                log::warn!("Failed to build HTTP client: {e}; falling back to default client");
                reqwest::blocking::Client::new()
            });
        Self { config, client }
    }

    /// Whether this client will include tools in chat requests.
    pub fn tools_enabled(&self) -> bool {
        self.config.chat_tools_enabled
    }

    /// Returns a reference to the loaded assistant configuration.
    pub fn config(&self) -> &AssistantConfig {
        &self.config
    }

    /// Single-shot (non-streaming) completion for short tasks like title generation.
    ///
    /// Internally uses `chat_step` with an empty tools list and accumulates all tokens
    /// into a String. The returned text is trimmed of leading/trailing whitespace.
    pub fn complete_once(&self, model: &str, messages: &[ApiMessage]) -> Result<String> {
        let cancelled = AtomicBool::new(false);
        let mut text = String::new();
        self.chat_step(model, messages, &[], &cancelled, &mut |tok| {
            text.push_str(tok);
        })?;
        Ok(text.trim().to_string())
    }

    /// Fetch available chat models from `{base_url}/models`.
    /// Filters out non-chat models (embeddings, TTS, image, etc.).
    pub fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/models", self.config.base_url);
        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .send()
            .context("GET /models failed")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            anyhow::bail!("models API {}: {}", status, body);
        }
        let v: serde_json::Value = resp.json().context("parse /models response")?;
        let arr = v
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| anyhow::anyhow!("missing `data` array in /models response"))?;
        let mut out: Vec<String> = arr
            .iter()
            .filter_map(|m| m.get("id").and_then(|s| s.as_str()).map(String::from))
            .filter(|id| kaku_ai_utils::is_chat_model_id(id))
            .collect();
        out.sort();
        out.dedup();
        out.truncate(30);
        Ok(out)
    }

    /// Single chat step with optional tool support.
    ///
    /// Streams text tokens via `on_token`. If the model responds by requesting
    /// tool calls instead of (or before) text, returns those calls for the
    /// caller to execute and loop. Returns an empty vec when the step is text-only.
    ///
    /// The caller must set `cancelled` to `true` to abort mid-stream.
    pub fn chat_step(
        &self,
        model: &str,
        messages: &[ApiMessage],
        tools: &[serde_json::Value],
        cancelled: &AtomicBool,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<Vec<ToolCall>> {
        let url = format!("{}/chat/completions", self.config.base_url);

        let mut body = serde_json::json!({
            "model": model,
            "messages": messages.iter().map(|m| m.0.clone()).collect::<Vec<_>>(),
            "stream": true,
        });
        if !tools.is_empty() && self.config.chat_tools_enabled {
            body["tools"] = serde_json::Value::Array(tools.to_vec());
        }

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("Accept-Encoding", "identity")
            .json(&body)
            .send()
            .context("HTTP request failed")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            anyhow::bail!("API error {}: {}", status, body);
        }

        let reader = BufReader::new(response);
        // Accumulate tool call fragments by index; each index is one pending call.
        // BTreeMap keeps indices sorted so we process them in order.
        let mut tc_buf: BTreeMap<usize, ToolCallBuf> = BTreeMap::new();
        let mut finish_reason = String::new();

        for line in reader.lines() {
            if cancelled.load(Ordering::Relaxed) {
                break;
            }
            let line = line.context("read SSE line")?;
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data.trim() == "[DONE]" {
                break;
            }
            let chunk = match serde_json::from_str::<serde_json::Value>(data) {
                Ok(v) => v,
                Err(e) => {
                    log::warn!("Failed to parse SSE chunk: {e}");
                    continue;
                }
            };

            let choice = &chunk["choices"][0];

            // Capture finish_reason when present.
            if let Some(fr) = choice["finish_reason"].as_str() {
                if !fr.is_empty() && fr != "null" {
                    finish_reason = fr.to_string();
                }
            }

            let delta = &choice["delta"];

            // Text delta (standard) and reasoning delta (DeepSeek et al.).
            if let Some(reasoning) = delta["reasoning_content"]
                .as_str()
                .or_else(|| choice["reasoning"].as_str())
            {
                if !reasoning.is_empty() {
                    on_token(&format!("<think>\n{}\n</think>\n\n", reasoning));
                }
            }
            if let Some(content) = delta["content"].as_str() {
                on_token(content);
            }

            // Tool call deltas: accumulate arguments by index.
            if let Some(tc_arr) = delta["tool_calls"].as_array() {
                for tc in tc_arr {
                    let idx = tc["index"].as_u64().unwrap_or(0) as usize;
                    let entry = tc_buf.entry(idx).or_default();
                    if let Some(id) = tc["id"].as_str() {
                        entry.id = id.to_string();
                    }
                    if let Some(name) = tc["function"]["name"].as_str() {
                        entry.name = name.to_string();
                    }
                    if let Some(args) = tc["function"]["arguments"].as_str() {
                        entry.arguments.push_str(args);
                    }
                }
            }
        }

        // Build ToolCall results only when the model explicitly requested tool use.
        if finish_reason == "tool_calls" {
            let calls = tc_buf
                .into_values()
                .map(|b| ToolCall {
                    id: b.id,
                    name: b.name,
                    arguments: b.arguments,
                })
                .collect();
            Ok(calls)
        } else {
            Ok(vec![])
        }
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Buffer for accumulating streamed tool call fragments.
#[derive(Default)]
struct ToolCallBuf {
    id: String,
    name: String,
    arguments: String,
}

// Delegated to kaku-ai-utils crate to avoid cross-binary drift.
