//! Small shared utilities for Kaku's AI-related binaries.

/// Returns false for model IDs that are clearly not conversational (embeddings,
/// TTS, image generation, ASR, moderation). Everything else is assumed to be a
/// chat model.
pub fn is_chat_model_id(id: &str) -> bool {
    const BLOCK: &[&str] = &[
        "whisper",
        "tts",
        "dall-e",
        "dalle",
        "embedding",
        "moderation",
        "audio",
        "image",
        "davinci",
        "babbage",
        "ada-",
    ];
    let lower = id.to_ascii_lowercase();
    !BLOCK.iter().any(|p| lower.contains(p))
}
