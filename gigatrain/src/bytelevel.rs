//! ByteLevel pretokenization, matching HF `pre_tokenizers.ByteLevel`.
//!
//! This is the configuration essentially every production BPE tokenizer uses
//! (GPT-2 and descendants), so parity here matters more than for plain
//! whitespace splitting.
//!
//! Two steps:
//!
//! 1. Split on the GPT-2 pattern
//!    `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//!    with leftmost-first alternation.
//! 2. Map each byte of each piece through the GPT-2 byte-to-unicode table, so
//!    that arbitrary bytes become printable chars and the BPE alphabet is
//!    exactly 256 symbols.
//!
//! The regex is implemented directly rather than with a regex engine: the
//! pattern needs the lookahead `(?!\S)`, which the `regex` crate does not
//! support, and taking on `fancy-regex`/`onig` for one fixed pattern would
//! cost a dependency and still need verifying against HF. Correctness is
//! established by differential tests against HF's own output (see
//! scripts/check_bytelevel_parity.py) rather than by inspection.

use crate::unicode_tables::{is_letter, is_number};

/// GPT-2's byte-to-unicode map: 256 distinct printable chars, one per byte.
///
/// Bytes that are already printable ASCII/Latin-1 map to themselves; the rest
/// map to U+0100.. in order, so no piece ever contains whitespace or control
/// characters after mapping.
pub fn byte_to_char() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut used = [false; 256];
    // The three "already printable" runs GPT-2 keeps as-is.
    for b in (b'!'..=b'~').chain(0xA1..=0xAC).chain(0xAE..=0xFF) {
        table[b as usize] = b as char;
        used[b as usize] = true;
    }
    let mut n = 0u32;
    for b in 0..256usize {
        if !used[b] {
            table[b] = char::from_u32(256 + n).expect("valid scalar");
            n += 1;
        }
    }
    table
}

#[inline]
fn is_ws(c: char) -> bool {
    c.is_whitespace()
}

/// Length in bytes of the GPT-2 contraction at the start of `s`, if any.
/// Matches the literal alternatives `'s 't 're 've 'm 'll 'd` (lowercase
/// only, as in the pattern).
fn contraction_len(s: &str) -> Option<usize> {
    for pat in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
        if s.starts_with(pat) {
            return Some(pat.len());
        }
    }
    None
}

/// Byte length of the next GPT-2 token in `s`, which must be non-empty.
fn next_piece_len(s: &str) -> usize {
    if let Some(n) = contraction_len(s) {
        return n;
    }

    let mut it = s.char_indices();
    let (_, first) = it.next().expect("non-empty");

    // ` ?\p{L}+` | ` ?\p{N}+` | ` ?[^\s\p{L}\p{N}]+`: an optional single
    // leading space, then a run of one class. The space is only consumed if a
    // qualifying character follows it.
    let (body_start, lead) = if first == ' ' {
        match it.clone().next() {
            Some((i, c)) => (i, Some(c)),
            None => (s.len(), None),
        }
    } else {
        (0, Some(first))
    };

    if let Some(c) = lead {
        let class = if is_letter(c) {
            Some(0)
        } else if is_number(c) {
            Some(1)
        } else if !is_ws(c) {
            Some(2)
        } else {
            None
        };
        if let Some(class) = class {
            let mut end = body_start;
            for (i, ch) in s[body_start..].char_indices() {
                let ok = match class {
                    0 => is_letter(ch),
                    1 => is_number(ch),
                    _ => !is_ws(ch) && !is_letter(ch) && !is_number(ch),
                };
                if !ok {
                    end = body_start + i;
                    return end;
                }
                end = body_start + i + ch.len_utf8();
            }
            return end;
        }
    }

    // `\s+(?!\S)` then `\s+`: take the whitespace run, but if it is followed
    // by a non-whitespace character, give the last whitespace char back so it
    // can start the next piece (that is what the negative lookahead does).
    debug_assert!(is_ws(first));
    let mut end = 0;
    let mut last_ws_start = 0;
    for (i, ch) in s.char_indices() {
        if !is_ws(ch) {
            // Followed by non-whitespace: the lookahead fails at the full run,
            // so the match backs off one character.
            return if last_ws_start > 0 { last_ws_start } else { i };
        }
        last_ws_start = i;
        end = i + ch.len_utf8();
    }
    end
}

/// Call `f` with each *unmapped* GPT-2 piece of `text`.
///
/// The byte-to-unicode mapping is a bijection, so two pieces are equal after
/// mapping exactly when they are equal before it. Phase 1 therefore hashes and
/// counts raw pieces and maps only the unique words at the end — roughly 9M
/// mappings on a 13 GB corpus instead of one per token occurrence.
pub fn for_each_piece<'a>(text: &'a str, mut f: impl FnMut(&'a str)) {
    let mut rest = text;
    while !rest.is_empty() {
        let n = next_piece_len(rest);
        debug_assert!(n > 0, "zero-length piece would loop forever");
        f(&rest[..n]);
        rest = &rest[n..];
    }
}

/// Apply the byte-to-unicode map to `piece`, writing into `buf`.
#[inline]
pub fn map_bytes(piece: &str, table: &[char; 256], buf: &mut String) {
    buf.clear();
    for &b in piece.as_bytes() {
        buf.push(table[b as usize]);
    }
}

/// Call `f` with each ByteLevel-mapped token of `text`.
///
/// `buf` is reused across calls to avoid per-token allocation.
pub fn for_each_token(text: &str, table: &[char; 256], buf: &mut String, mut f: impl FnMut(&str)) {
    for_each_piece(text, |piece| {
        map_bytes(piece, table, buf);
        f(buf);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(text: &str) -> Vec<String> {
        let table = byte_to_char();
        let mut buf = String::new();
        let mut out = vec![];
        for_each_token(text, &table, &mut buf, |t| out.push(t.to_string()));
        out
    }

    #[test]
    fn byte_table_matches_gpt2() {
        let t = byte_to_char();
        // Printable ASCII maps to itself.
        assert_eq!(t[b'!' as usize], '!');
        assert_eq!(t[b'~' as usize], '~');
        assert_eq!(t[b'A' as usize], 'A');
        // Space and control bytes are remapped above U+0100.
        assert_eq!(t[b' ' as usize], 'Ġ');
        assert_eq!(t[b'\n' as usize], 'Ċ');
        assert_eq!(t[0], 'Ā');
        // All 256 outputs are distinct.
        let mut seen: Vec<char> = t.to_vec();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 256);
    }

    // Expected values captured from HF pre_tokenizers.ByteLevel
    // (add_prefix_space=False, use_regex=True).
    #[test]
    fn matches_hf_reference_cases() {
        assert_eq!(split("Hello world"), vec!["Hello", "Ġworld"]);
        assert_eq!(split("don't stop"), vec!["don", "'t", "Ġstop"]);
        assert_eq!(split("a  b"), vec!["a", "Ġ", "Ġb"]);
        assert_eq!(split("x   "), vec!["x", "ĠĠĠ"]);
        assert_eq!(split("  lead"), vec!["Ġ", "Ġlead"]);
        assert_eq!(split("123abc"), vec!["123", "abc"]);
        assert_eq!(split("a1b2"), vec!["a", "1", "b", "2"]);
        assert_eq!(split("end.  "), vec!["end", ".", "ĠĠ"]);
        assert_eq!(split("café"), vec!["cafÃ©"]);
        assert_eq!(split("中文 text"), vec!["ä¸Ńæĸĩ", "Ġtext"]);
        // U+2167 is Nl: \p{N}, not \p{L}.
        assert_eq!(split("Ⅷ roman"), vec!["âħ§", "Ġroman"]);
        // U+0345 is Other_Alphabetic: neither \p{L} nor \p{N}.
        assert_eq!(split("aͅb"), vec!["a", "Íħ", "b"]);
    }

    #[test]
    fn never_loops_or_drops_bytes() {
        for text in [
            "",
            " ",
            "\n",
            "\t\t",
            "'",
            "''",
            "'s",
            "'S",
            "a'",
            "  ",
            "a  ",
            "\u{00a0}x",
            "\u{3000}",
            "🙂🙂",
            "a🙂b",
            "1.5",
            "-3",
            "e=mc²",
        ] {
            let pieces = split(text);
            // Byte-level mapping is injective, so total mapped chars must
            // equal total input bytes: nothing dropped, nothing duplicated.
            let mapped: usize = pieces.iter().map(|p| p.chars().count()).sum();
            assert_eq!(mapped, text.len(), "byte count mismatch on {text:?}");
        }
    }
}
