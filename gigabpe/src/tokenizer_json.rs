//! Render a trained model as a HuggingFace `tokenizer.json`.
//!
//! Written by hand rather than with serde: the document shape is fixed and
//! small, and the crate stays dependency-free. Only the escaping needs care,
//! and that is covered by round-trip tests that load the output back through
//! `tokenizers` (see scripts/check_bindings.py).

use crate::trainer::TrainResult;

/// Escape a string as a JSON string body (without surrounding quotes).
///
/// Matches `serde_json` exactly — the two-character escapes `\b`/`\f` for
/// U+0008/U+000C and `\uXXXX` for the remaining control characters — so
/// output is byte-comparable with HuggingFace's own serializer for the tokens
/// a BPE vocabulary can contain. (U+0008 is reachable: it is neither `\w` nor
/// whitespace, so it survives whitespace pretokenization.)
fn escape(s: &str, out: &mut String) {
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
}

fn quoted(s: &str, out: &mut String) {
    out.push('"');
    escape(s, out);
    out.push('"');
}

/// Serialize `s` as a JSON string literal, surrounding quotes included.
///
/// Exposed so the CLI's `--vocab-out` uses exactly the escaping the
/// `tokenizer.json` writer does, rather than a second implementation that
/// could drift from it.
pub fn json_string(s: &str) -> String {
    let mut out = String::new();
    quoted(s, &mut out);
    out
}

/// Serialize `result` as a `tokenizer.json` document.
///
/// `special_tokens` are emitted as added tokens so they survive a round trip,
/// and `bytelevel` selects the matching pre_tokenizer/decoder so the file is
/// usable for encoding, not just inspection.
pub fn render(
    result: &TrainResult,
    special_tokens: &[String],
    bytelevel: bool,
    continuing_subword_prefix: Option<&str>,
    end_of_word_suffix: Option<&str>,
) -> String {
    let mut s = String::with_capacity(result.vocab.len() * 24);
    s.push_str("{\n  \"version\": \"1.0\",\n  \"truncation\": null,\n  \"padding\": null,\n");

    // added_tokens: the special tokens, with the ids they were assigned.
    // Deduplicated: the trainer assigns one id per distinct special, and HF's
    // AddedVocabulary emits one entry per token, so a repeated --special must
    // not produce two identical objects here.
    s.push_str("  \"added_tokens\": [");
    let mut first = true;
    let mut seen: Vec<&str> = Vec::with_capacity(special_tokens.len());
    for token in special_tokens {
        if seen.contains(&token.as_str()) {
            continue;
        }
        seen.push(token);
        if let Some(id) = result.vocab.iter().position(|t| t == token) {
            if !first {
                s.push(',');
            }
            first = false;
            s.push_str("\n    {\"id\": ");
            s.push_str(&id.to_string());
            s.push_str(", \"content\": ");
            quoted(token, &mut s);
            s.push_str(", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": true}");
        }
    }
    s.push_str(if first { "],\n" } else { "\n  ],\n" });

    s.push_str("  \"normalizer\": null,\n");
    if bytelevel {
        s.push_str(concat!(
            "  \"pre_tokenizer\": {\"type\": \"ByteLevel\", \"add_prefix_space\": false, ",
            "\"trim_offsets\": true, \"use_regex\": true},\n",
            "  \"post_processor\": {\"type\": \"ByteLevel\", \"add_prefix_space\": true, ",
            "\"trim_offsets\": true, \"use_regex\": true},\n",
            "  \"decoder\": {\"type\": \"ByteLevel\", \"add_prefix_space\": true, ",
            "\"trim_offsets\": true, \"use_regex\": true},\n",
        ));
    } else {
        s.push_str(concat!(
            "  \"pre_tokenizer\": {\"type\": \"WhitespaceSplit\"},\n",
            "  \"post_processor\": null,\n",
            "  \"decoder\": null,\n",
        ));
    }

    s.push_str("  \"model\": {\n    \"type\": \"BPE\",\n");
    s.push_str("    \"dropout\": null,\n    \"unk_token\": null,\n");
    for (field, value) in [
        ("continuing_subword_prefix", continuing_subword_prefix),
        ("end_of_word_suffix", end_of_word_suffix),
    ] {
        s.push_str("    \"");
        s.push_str(field);
        s.push_str("\": ");
        match value {
            Some(v) => quoted(v, &mut s),
            None => s.push_str("null"),
        }
        s.push_str(",\n");
    }
    s.push_str("    \"fuse_unk\": false,\n    \"byte_fallback\": false,\n");
    s.push_str("    \"ignore_merges\": false,\n");

    s.push_str("    \"vocab\": {");
    for (id, token) in result.vocab.iter().enumerate() {
        if id > 0 {
            s.push(',');
        }
        s.push_str("\n      ");
        quoted(token, &mut s);
        s.push_str(": ");
        s.push_str(&id.to_string());
    }
    s.push_str("\n    },\n");

    s.push_str("    \"merges\": [");
    for (i, (a, b)) in result.serialized_merges().iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str("\n      [");
        quoted(a, &mut s);
        s.push_str(", ");
        quoted(b, &mut s);
        s.push(']');
    }
    s.push_str("\n    ]\n  }\n}\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_json_metacharacters() {
        let mut out = String::new();
        escape("a\"b\\c\nd\te\u{1}f", &mut out);
        assert_eq!(out, "a\\\"b\\\\c\\nd\\te\\u0001f");
    }

    #[test]
    fn leaves_normal_text_alone() {
        let mut out = String::new();
        escape("Ġthe中文", &mut out);
        assert_eq!(out, "Ġthe中文");
    }
}
