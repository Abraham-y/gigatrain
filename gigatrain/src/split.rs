//! Whitespace pretokenization, byte-fast but exactly equivalent to
//! `str::split_whitespace` (and therefore to HF's `WhitespaceSplit`, which
//! splits on `char::is_whitespace` with delimiters removed).
//!
//! `split_whitespace` decodes every char to test `is_whitespace`. Web text is
//! overwhelmingly ASCII, so this scans raw bytes and only decodes when it sees
//! a lead byte that could possibly begin a whitespace char.
//!
//! Unicode White_Space is: ASCII 0x09-0x0D and 0x20; U+0085, U+00A0 (lead
//! byte 0xC2); U+1680 (0xE1); U+2000-200A, U+2028, U+2029, U+202F, U+205F
//! (0xE2); U+3000 (0xE3). No other lead byte can start a whitespace char, so
//! any other non-ASCII byte is unconditionally part of a word.

#[inline(always)]
fn is_ascii_ws(b: u8) -> bool {
    b == b' ' || (0x09..=0x0d).contains(&b)
}

/// Could a UTF-8 char starting with this byte be whitespace?
#[inline(always)]
fn maybe_ws_lead(b: u8) -> bool {
    matches!(b, 0xC2 | 0xE1 | 0xE2 | 0xE3)
}

/// (is_whitespace, utf8_len) for the char starting at `i`, which must be a
/// char boundary.
#[inline]
fn decode_at(text: &str, i: usize) -> (bool, usize) {
    match text[i..].chars().next() {
        Some(c) => (c.is_whitespace(), c.len_utf8()),
        None => (false, 1),
    }
}

/// Call `f` with each whitespace-delimited word, in order. Words borrow from
/// `text`, so callers may collect them.
pub fn for_each_word<'a>(text: &'a str, mut f: impl FnMut(&'a str)) {
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        // Skip run of whitespace.
        while i < n {
            let b = bytes[i];
            if b < 0x80 {
                if !is_ascii_ws(b) {
                    break;
                }
                i += 1;
            } else if maybe_ws_lead(b) {
                let (ws, len) = decode_at(text, i);
                if !ws {
                    break;
                }
                i += len;
            } else {
                break;
            }
        }

        // Consume run of non-whitespace.
        let start = i;
        while i < n {
            let b = bytes[i];
            if b < 0x80 {
                if is_ascii_ws(b) {
                    break;
                }
                i += 1;
            } else if maybe_ws_lead(b) {
                let (ws, len) = decode_at(text, i);
                if ws {
                    break;
                }
                i += len;
            } else {
                i += 1;
            }
        }

        if i > start {
            // Safety: `start` and `i` are both char boundaries — the scan only
            // stops at ASCII bytes or at UTF-8 lead bytes.
            f(unsafe { std::str::from_utf8_unchecked(&bytes[start..i]) });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<&str> {
        let mut out = vec![];
        for_each_word(text, |w| out.push(w));
        out
    }

    fn assert_matches_std(text: &str) {
        let expected: Vec<&str> = text.split_whitespace().collect();
        assert_eq!(words(text), expected, "mismatch on {text:?}");
    }

    #[test]
    fn matches_std_split_whitespace() {
        for text in [
            "",
            "   ",
            "hello world",
            "  leading and trailing  ",
            "tabs\tand\nnewlines\r\nhere",
            "vertical\x0btab\x0cformfeed",
            "unicode\u{00a0}nbsp\u{2003}emspace",
            "next\u{0085}line",
            "ogham\u{1680}space",
            "ideographic\u{3000}space",
            "narrow\u{202f}nbsp\u{205f}medmath",
            "line\u{2028}sep\u{2029}para",
            // Non-whitespace chars sharing the same lead bytes.
            "\u{00e9}accented \u{20ac}euro \u{2014}emdash \u{e000}priv",
            "\u{1234}\u{2600}\u{3042}mixed",
            "emoji\u{1f600}here",
            "中文 文本 测试",
            "\u{00a0}",
            "a\u{00a0}b",
            "\u{2000}\u{2001}\u{2002}",
        ] {
            assert_matches_std(text);
        }
    }

    // Every char whose lead byte we fast-path must be classified correctly.
    #[test]
    fn all_lead_byte_chars_agree_with_std() {
        for cp in 0u32..0x1_0000 {
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            let mut buf = [0u8; 4];
            let s = c.encode_utf8(&mut buf);
            let text = format!("a{s}b");
            assert_matches_std(&text);
        }
    }
}
