use crate::ai_client::{AiClient, ApiMessage};
use crate::ai_conversations;
use crate::overlay::ai_chat::{approval_summary, StreamMsg};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;

/// Generate a short title for a conversation (≤ 40 chars). Runs on a background thread.
pub(crate) fn generate_summary(
    client: &AiClient,
    messages: &[ai_conversations::PersistedMessage],
) -> anyhow::Result<String> {
    let model = client.config().chat_model.clone();
    // Take up to the last 20 messages to keep the prompt short.
    let window = if messages.len() > 20 {
        &messages[messages.len() - 20..]
    } else {
        messages
    };
    let mut api_msgs = vec![ApiMessage::system(
        "You are a titler. Summarize the following conversation in a short phrase \
         (max 40 characters). Use the same language as the conversation. \
         Return only the phrase, no quotes.",
    )];
    for m in window {
        if m.role == "user" {
            api_msgs.push(ApiMessage::user(&m.content));
        } else {
            api_msgs.push(ApiMessage::assistant(&m.content));
        }
    }
    let summary = client.complete_once(&model, &api_msgs)?;
    let truncated: String = summary.chars().take(40).collect();
    Ok(truncated)
}

// ─── Agent loop ──────────────────────────────────────────────────────────────

/// Background thread: runs chat_step in a loop, executing tool calls until the
/// model produces a text-only response or the round limit is reached.
pub(crate) fn run_agent(
    client: AiClient,
    model: String,
    mut messages: Vec<ApiMessage>,
    tools: Vec<serde_json::Value>,
    mut cwd: String,
    cancel: Arc<AtomicBool>,
    tx: Sender<StreamMsg>,
) {
    // ai_conversations used via fully-qualified path below
    use crate::ai_tools;
    const MAX_ROUNDS: usize = 15;

    for _ in 0..MAX_ROUNDS {
        if cancel.load(Ordering::Relaxed) {
            break;
        }

        let tx_c = tx.clone();
        let mut sent_start = false;
        let tool_calls = match client.chat_step(&model, &messages, &tools, &cancel, &mut |token| {
            if !sent_start {
                let _ = tx_c.send(StreamMsg::AssistantStart);
                sent_start = true;
            }
            let _ = tx_c.send(StreamMsg::Token(token.to_string()));
        }) {
            Ok(tc) => tc,
            Err(e) => {
                let _ = tx.send(StreamMsg::Err(e.to_string()));
                return;
            }
        };

        if tool_calls.is_empty() {
            // Text-only response: agent is done.
            let _ = tx.send(StreamMsg::Done);
            return;
        }

        // Record the assistant's tool-call turn in the conversation.
        let tc_json: Vec<serde_json::Value> = tool_calls
            .iter()
            .map(|tc| {
                serde_json::json!({
                    "id": tc.id,
                    "type": "function",
                    "function": { "name": tc.name, "arguments": tc.arguments }
                })
            })
            .collect();
        messages.push(ApiMessage::assistant_tool_calls(serde_json::Value::Array(
            tc_json,
        )));

        // Execute each tool call and collect results back into the conversation.
        for tc in &tool_calls {
            if cancel.load(Ordering::Relaxed) {
                break;
            }

            let args: serde_json::Value = serde_json::from_str(&tc.arguments).unwrap_or_default();
            // Extract a clean display hint. Priority: "query" (web_search/grep), "path", first value.
            let args_preview = args
                .get("query")
                .or_else(|| args.get("path"))
                .or_else(|| args.get("url"))
                .or_else(|| args.get("pattern"))
                .or_else(|| args.get("command"))
                .or_else(|| args.as_object().and_then(|o| o.values().next()))
                .and_then(|v| v.as_str())
                .map(|s| s.chars().take(40).collect::<String>())
                .unwrap_or_default();
            // All state-mutating tools require user approval before running.
            if let Some(summary) = approval_summary(&tc.name, &args) {
                let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel::<bool>(0);
                let _ = tx.send(StreamMsg::ApprovalRequired { summary, reply_tx });
                // Block until the user responds or cancels.
                let approved = reply_rx.recv().unwrap_or(false);
                if !approved {
                    let _ = tx.send(StreamMsg::ToolFailed {
                        error: "Operation rejected by user.".into(),
                    });
                    messages.push(ApiMessage::tool_result(
                        tc.id.clone(),
                        "Error: user rejected the operation.".to_string(),
                    ));
                    continue;
                }
            }

            let _ = tx.send(StreamMsg::ToolStart {
                name: tc.name.clone(),
                args_preview,
            });

            match ai_tools::execute(&tc.name, &args, &mut cwd, client.config()) {
                Ok(result) => {
                    let _ = tx.send(StreamMsg::ToolDone {
                        result_preview: String::new(),
                    });
                    messages.push(ApiMessage::tool_result(tc.id.clone(), result));
                }
                Err(e) => {
                    let err_str = e.to_string();
                    let _ = tx.send(StreamMsg::ToolFailed {
                        error: err_str.clone(),
                    });
                    // Feed the error back as the tool result so the model can recover.
                    messages.push(ApiMessage::tool_result(
                        tc.id.clone(),
                        format!("Error: {}", err_str),
                    ));
                }
            }
        }
    }

    // Exceeded max rounds without a text-only response.
    let _ = tx.send(StreamMsg::Err(
        "Reached the maximum number of tool-call rounds (15).".to_string(),
    ));
    let _ = tx.send(StreamMsg::Done);
}
