use crate::overlay::ai_chat::{InlineSpan, InlineStyle, MdBlock};
use termwiz::cell::unicode_column_width;
use unicode_segmentation::UnicodeSegmentation;

pub(crate) fn parse_markdown_blocks(content: &str) -> Vec<MdBlock> {
    let mut out = Vec::new();
    let mut in_fence = false;
    for line in content.split('\n') {
        let trimmed_start = line.trim_start();
        // Fence open/close: ``` or ~~~ on their own (possibly with info string).
        if trimmed_start.starts_with("```") || trimmed_start.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            out.push(MdBlock::CodeLine(line.to_string()));
            continue;
        }
        if trimmed_start.is_empty() {
            out.push(MdBlock::Blank);
            continue;
        }
        let tight = trimmed_start.trim_end();
        // Horizontal rule: 3+ of `-`, `*`, or `_` with nothing else.
        if tight.len() >= 3
            && (tight.chars().all(|c| c == '-')
                || tight.chars().all(|c| c == '*')
                || tight.chars().all(|c| c == '_'))
        {
            out.push(MdBlock::Hr);
            continue;
        }
        // ATX headings (#, ##, ###, ####). Five+ levels collapse to 4.
        if let Some((level, rest)) = parse_heading_prefix(trimmed_start) {
            out.push(MdBlock::Heading {
                level,
                text: rest.to_string(),
            });
            continue;
        }
        // Blockquote.
        if let Some(rest) = trimmed_start.strip_prefix("> ") {
            out.push(MdBlock::Quote(rest.to_string()));
            continue;
        }
        if trimmed_start == ">" {
            out.push(MdBlock::Quote(String::new()));
            continue;
        }
        // Unordered list: `- `, `* `, `+ `.
        if let Some(rest) = trimmed_start
            .strip_prefix("- ")
            .or_else(|| trimmed_start.strip_prefix("* "))
            .or_else(|| trimmed_start.strip_prefix("+ "))
        {
            out.push(MdBlock::ListItem {
                marker: "• ".to_string(),
                text: rest.to_string(),
            });
            continue;
        }
        // Ordered list: `<digits>. `.
        if let Some((num, rest)) = split_numbered_list(trimmed_start) {
            out.push(MdBlock::ListItem {
                marker: format!("{}. ", num),
                text: rest.to_string(),
            });
            continue;
        }
        out.push(MdBlock::Paragraph(trimmed_start.to_string()));
    }
    out
}

fn parse_heading_prefix(s: &str) -> Option<(u8, &str)> {
    for level in (1u8..=4).rev() {
        let marker_len = level as usize + 1; // "# " ... "#### "
        let pounds = "#".repeat(level as usize);
        let prefix = format!("{} ", pounds);
        if let Some(rest) = s.strip_prefix(&prefix) {
            // Don't match if preceded by a # (that would be a higher level).
            // Since we iterate highest first, this handles "##### " → level 4 + remainder.
            return Some((level, rest));
        }
        let _ = marker_len;
    }
    None
}

fn split_numbered_list(s: &str) -> Option<(String, &str)> {
    let end = s.find(|c: char| !c.is_ascii_digit())?;
    if end == 0 || end > 3 {
        // No digits, or absurdly long number (not a list).
        return None;
    }
    let rest = &s[end..];
    let after = rest.strip_prefix(". ")?;
    Some((s[..end].to_string(), after))
}

/// Walk a single line and split it into styled spans. Pairs are matched
/// greedily left-to-right; an unclosed opener renders as literal (matching
/// termimad's behavior under streaming).
pub(crate) fn tokenize_inline(text: &str) -> Vec<InlineSpan> {
    let mut out: Vec<InlineSpan> = Vec::new();
    let mut plain = String::new();
    let flush_plain = |out: &mut Vec<InlineSpan>, plain: &mut String| {
        if !plain.is_empty() {
            merge_push(
                out,
                InlineSpan {
                    text: std::mem::take(plain),
                    style: InlineStyle::Plain,
                },
            );
        }
    };
    let mut rest = text;
    while !rest.is_empty() {
        // **bold** (also matches __bold__)
        if let Some((inner, after)) = match_paired(rest, "**").or_else(|| match_paired(rest, "__"))
        {
            flush_plain(&mut out, &mut plain);
            merge_push(
                &mut out,
                InlineSpan {
                    text: inner.to_string(),
                    style: InlineStyle::Bold,
                },
            );
            rest = after;
            continue;
        }
        // `code`
        if let Some((inner, after)) = match_paired(rest, "`") {
            flush_plain(&mut out, &mut plain);
            merge_push(
                &mut out,
                InlineSpan {
                    text: inner.to_string(),
                    style: InlineStyle::Code,
                },
            );
            rest = after;
            continue;
        }
        // ~~strike~~ → drop markers, keep inner as plain
        if let Some((inner, after)) = match_paired(rest, "~~") {
            plain.push_str(inner);
            rest = after;
            continue;
        }
        // *italic* (single star, not part of **); avoid matching when the
        // opening star is immediately followed by whitespace (that's usually
        // a stray `*`, not emphasis).
        if rest.starts_with('*') && !rest.starts_with("**") {
            let after_star = &rest['*'.len_utf8()..];
            if !after_star.starts_with(' ') && !after_star.starts_with('\t') {
                if let Some((inner, after)) = match_single_italic(rest, '*') {
                    flush_plain(&mut out, &mut plain);
                    merge_push(
                        &mut out,
                        InlineSpan {
                            text: inner.to_string(),
                            style: InlineStyle::Italic,
                        },
                    );
                    rest = after;
                    continue;
                }
            }
        }
        // [label](url) → keep label as plain, drop url.
        if let Some((label, after)) = match_link(rest) {
            plain.push_str(label);
            rest = after;
            continue;
        }
        // Default: consume one char (handles UTF-8 boundaries).
        let mut chars = rest.char_indices();
        let (_, ch) = chars.next().expect("rest is non-empty");
        let next = chars.next().map(|(b, _)| b).unwrap_or(rest.len());
        plain.push(ch);
        rest = &rest[next..];
    }
    flush_plain(&mut out, &mut plain);
    out
}

/// Append `span` to `out`, merging with the last span if styles match. Keeps
/// the run count low, which matters for render throughput.
/// CJK kinsoku shori: characters that must NOT appear at the start of a line.
/// Covers fullwidth punctuation (Chinese/Japanese) and common closing brackets.
fn is_cjk_no_line_start(g: &str) -> bool {
    let c = match g.chars().next() {
        Some(c) => c,
        None => return false,
    };
    matches!(
        c,
        '：' | '，'
            | '。'
            | '、'
            | '；'
            | '？'
            | '！'
            | '）'
            | '】'
            | '》'
            | '」'
            | '』'
            | '〉'
            | '〕'
            | '…'
            | '‥'
            | '～'
            | ')'
            | ']'
            | '}'
            | '.'
            | ','
            | ':'
            | ';'
            | '?'
            | '!'
    )
}

fn merge_push(out: &mut Vec<InlineSpan>, span: InlineSpan) {
    if span.text.is_empty() {
        return;
    }
    if let Some(last) = out.last_mut() {
        if last.style == span.style {
            last.text.push_str(&span.text);
            return;
        }
    }
    out.push(span);
}

/// If `s` starts with `delim`, try to find a matching closing `delim` on the
/// same line, returning `(inner, rest_after)`. Returns None if empty content
/// or the closer isn't found.
fn match_paired<'a>(s: &'a str, delim: &str) -> Option<(&'a str, &'a str)> {
    let after_open = s.strip_prefix(delim)?;
    let close = after_open.find(delim)?;
    if close == 0 {
        return None;
    }
    let inner = &after_open[..close];
    if inner.contains('\n') {
        return None;
    }
    Some((inner, &after_open[close + delim.len()..]))
}

/// Match `*italic*` where the closer `*` is not immediately followed by
/// another `*` (that would be a bold opener).
fn match_single_italic(s: &str, delim: char) -> Option<(&str, &str)> {
    let mut chars = s.char_indices();
    let (_, first) = chars.next()?;
    if first != delim {
        return None;
    }
    let after_open_byte = chars.next().map(|(b, _)| b).unwrap_or(s.len());
    let after_open = &s[after_open_byte..];
    if after_open.is_empty() {
        return None;
    }
    // Search for a closing delim that is not part of a doubled pair.
    let mut search_from = 0;
    while search_from < after_open.len() {
        let rel = after_open[search_from..].find(delim)?;
        let abs = search_from + rel;
        let next = abs + delim.len_utf8();
        let is_double = after_open[next..].starts_with(delim);
        if is_double {
            search_from = next + delim.len_utf8();
            continue;
        }
        if abs == 0 {
            return None;
        }
        let inner = &after_open[..abs];
        if inner.contains('\n') {
            return None;
        }
        return Some((inner, &after_open[next..]));
    }
    None
}

/// Match `[label](url)`, returning `(label, rest_after)`. Rejects nested
/// brackets and multi-line content.
fn match_link(s: &str) -> Option<(&str, &str)> {
    let after_open = s.strip_prefix('[')?;
    let close_label = after_open.find(']')?;
    let label = &after_open[..close_label];
    if label.contains('\n') || label.contains('[') {
        return None;
    }
    let after_label = &after_open[close_label + 1..];
    let after_paren_open = after_label.strip_prefix('(')?;
    let close_paren = after_paren_open.find(')')?;
    if after_paren_open[..close_paren].contains('\n') {
        return None;
    }
    Some((label, &after_paren_open[close_paren + 1..]))
}

pub(crate) fn segments_to_plain(segments: &[InlineSpan]) -> String {
    let mut s = String::new();
    for seg in segments {
        s.push_str(&seg.text);
    }
    s
}

/// Word-wrap a list of styled spans into one or more wrapped lines. Preserves
/// span boundaries: a wrapped line contains a subset of the input spans, split
/// at whitespace where possible. If a single token exceeds `width`, it stays
/// on its own (possibly overflowing) line rather than being grapheme-split.
pub(crate) fn wrap_segments(segments: &[InlineSpan], width: usize) -> Vec<Vec<InlineSpan>> {
    if width == 0 {
        return vec![segments.to_vec()];
    }
    // Tokenize into (text, style, visual_width, is_whitespace).
    let mut tokens: Vec<(String, InlineStyle, usize, bool)> = Vec::new();
    for seg in segments {
        let mut buf = String::new();
        let mut buf_ws: Option<bool> = None;
        for g in seg.text.graphemes(true) {
            let g_is_ws = g.chars().all(|c| c == ' ' || c == '\t');
            match buf_ws {
                Some(prev) if prev == g_is_ws => buf.push_str(g),
                Some(_) => {
                    let w = unicode_column_width(&buf, None);
                    tokens.push((std::mem::take(&mut buf), seg.style, w, buf_ws.unwrap()));
                    buf.push_str(g);
                    buf_ws = Some(g_is_ws);
                }
                None => {
                    buf.push_str(g);
                    buf_ws = Some(g_is_ws);
                }
            }
        }
        if !buf.is_empty() {
            let w = unicode_column_width(&buf, None);
            tokens.push((buf, seg.style, w, buf_ws.unwrap_or(false)));
        }
    }

    let mut lines: Vec<Vec<InlineSpan>> = Vec::new();
    let mut current: Vec<InlineSpan> = Vec::new();
    let mut current_w = 0usize;

    for (text, style, w, is_ws) in tokens {
        // Skip leading whitespace on a fresh line.
        if current_w == 0 && is_ws {
            continue;
        }
        if current_w + w > width && !current.is_empty() {
            lines.push(std::mem::take(&mut current));
            current_w = 0;
            if is_ws {
                continue;
            }
        }
        // Hard-break oversized non-whitespace tokens (long URLs, CJK runs) by
        // slicing at grapheme boundaries so they never exceed `width`.
        // CJK kinsoku: never break before "no-line-start" punctuation (：，。etc.)
        // even if it overshoots `width` by one character.
        if !is_ws && w > width {
            let mut chunk = String::new();
            let mut chunk_w = 0usize;
            let graphemes: Vec<&str> = text.graphemes(true).collect();
            for (gi, g) in graphemes.iter().enumerate() {
                let gw = unicode_column_width(g, None);
                if chunk_w + gw > width && !chunk.is_empty() {
                    // Kinsoku: if this grapheme is a no-line-start char, absorb it
                    // into the current chunk rather than letting it start a new line.
                    if is_cjk_no_line_start(g) {
                        chunk.push_str(g);
                        // chunk_w is reset to 0 below; the increment is skipped intentionally.
                        // Now flush the chunk that includes the punctuation.
                        if !current.is_empty() {
                            lines.push(std::mem::take(&mut current));
                            current_w = 0;
                        }
                        lines.push(vec![InlineSpan {
                            text: std::mem::take(&mut chunk),
                            style,
                        }]);
                        chunk_w = 0;
                        continue;
                    }
                    // Also check: if the NEXT grapheme is a no-line-start char,
                    // include the current grapheme AND the next one on this line.
                    if gi + 1 < graphemes.len() && is_cjk_no_line_start(graphemes[gi + 1]) {
                        chunk.push_str(g);
                        chunk_w += gw;
                        continue; // let the next iteration absorb it via the branch above.
                    }
                    // Normal break: flush the current chunk.
                    if !current.is_empty() {
                        lines.push(std::mem::take(&mut current));
                        current_w = 0;
                    }
                    lines.push(vec![InlineSpan {
                        text: std::mem::take(&mut chunk),
                        style,
                    }]);
                    chunk_w = 0;
                }
                chunk.push_str(g);
                chunk_w += gw;
            }
            // Remaining partial chunk continues on current line.
            if !chunk.is_empty() {
                merge_push(&mut current, InlineSpan { text: chunk, style });
                current_w += chunk_w;
            }
            continue;
        }
        merge_push(&mut current, InlineSpan { text, style });
        current_w += w;
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(Vec::new());
    }
    lines
}
