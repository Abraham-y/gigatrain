//! Word storage and in-word merge application.
//!
//! Merge semantics must match HF `tokenizers` `models/bpe/word.rs::Word::merge`
//! exactly (see PARITY.md). HF splices a Vec in place and re-reads neighbors
//! post-splice; we use a write-pointer scan that emits the identical multiset
//! of pair-count deltas without O(n^2) splicing.
//!
//! Storage is a flat arena of bare token IDs rather than a `Vec<Symbol>` per
//! word. Two properties make this work:
//!
//! - A merge only ever shrinks a word, so each word's slice start is fixed for
//!   the whole run and only its length shrinks: the arena never reallocates,
//!   never needs compaction, and costs one allocation instead of one per
//!   unique word.
//! - HF stores a per-symbol `len` (chars covered, summed on merge), but that
//!   value is always exactly the char count of the symbol's token string —
//!   initial symbols are one char, and a merge sums the two parts, matching
//!   the concatenated string. So it lives in a dense `token_chars` table
//!   indexed by token ID instead of in every symbol, halving the arena and
//!   the bytes the hot merge scan touches.

pub type Pair = (u32, u32);

#[derive(Default)]
pub struct WordArena {
    symbols: Vec<u32>,
    /// Per word: offset into `symbols` (fixed) and current length (shrinks).
    starts: Vec<u32>,
    lens: Vec<u32>,
}

impl WordArena {
    pub fn with_capacity(words: usize, symbols: usize) -> Self {
        Self {
            symbols: Vec::with_capacity(symbols),
            starts: Vec::with_capacity(words),
            lens: Vec::with_capacity(words),
        }
    }

    /// Append a word built from `ids` (chars already mapped to vocab IDs).
    pub fn push_word(&mut self, ids: impl Iterator<Item = u32>) {
        let start = self.symbols.len();
        assert!(start <= u32::MAX as usize, "symbol arena exceeds 4G symbols");
        self.symbols.extend(ids);
        self.starts.push(start as u32);
        self.lens.push((self.symbols.len() - start) as u32);
    }

    pub fn len(&self) -> usize {
        self.starts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.starts.is_empty()
    }

    #[inline]
    pub fn symbols_of(&self, i: usize) -> &[u32] {
        let start = self.starts[i] as usize;
        &self.symbols[start..start + self.lens[i] as usize]
    }

    /// Merge every non-overlapping occurrence of (c1, c2) in word `i`, left to
    /// right, appending pair-count deltas to `changes` as (pair, delta).
    ///
    /// `token_chars[id]` is the char count of token `id`, used only for the
    /// `max_len` guard. Pass `usize::MAX` as `max_len` to disable it, in which
    /// case `token_chars` is never read.
    ///
    /// Parity notes (PARITY.md): the merged pair itself is never decremented;
    /// the left neighbor is read post-merge (may be a symbol produced earlier
    /// in this same pass); the `max_len` guard is strict `<`.
    pub fn merge(
        &mut self,
        i: usize,
        c1: u32,
        c2: u32,
        new_id: u32,
        max_len: usize,
        token_chars: &[u32],
        changes: &mut Vec<(Pair, i32)>,
    ) {
        let start = self.starts[i] as usize;
        let n = self.lens[i] as usize;
        let syms = &mut self.symbols[start..start + n];
        let unbounded = max_len == usize::MAX;
        let new_chars = if unbounded {
            0
        } else {
            token_chars[new_id as usize] as usize
        };

        let mut w = 0;
        let mut k = 0;
        while k < n {
            if k + 1 < n && syms[k] == c1 && syms[k + 1] == c2 {
                if w > 0 {
                    let prev = syms[w - 1];
                    changes.push(((prev, c1), -1));
                    if unbounded || token_chars[prev as usize] as usize + new_chars < max_len {
                        changes.push(((prev, new_id), 1));
                    }
                }
                if k + 2 < n {
                    let right = syms[k + 2];
                    changes.push(((c2, right), -1));
                    if unbounded || token_chars[right as usize] as usize + new_chars < max_len {
                        changes.push(((new_id, right), 1));
                    }
                }
                syms[w] = new_id;
                w += 1;
                k += 2;
            } else {
                syms[w] = syms[k];
                w += 1;
                k += 1;
            }
        }
        self.lens[i] = w as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // token_chars large enough for the small IDs used below; the tests that
    // exercise max_len set the entries they rely on.
    fn arena_of(ids: &[u32]) -> WordArena {
        let mut a = WordArena::default();
        a.push_word(ids.iter().copied());
        a
    }

    // Port of HF word.rs::tests::test_merge ("hello", merge 'l'+'l' -> 'll').
    #[test]
    fn test_merge() {
        let mut a = arena_of(&[0, 1, 2, 2, 3]);
        let mut changes = vec![];
        a.merge(0, 2, 2, 4, usize::MAX, &[], &mut changes);
        assert_eq!(a.symbols_of(0), &[0, 1, 4, 3]);
        assert_eq!(
            changes,
            &[((1, 2), -1), ((1, 4), 1), ((2, 3), -1), ((4, 3), 1)]
        );
    }

    // Port of HF word.rs::tests::test_merge_max_length (max_length = 2).
    // chars: 'h','e','l','l','o' are 1 each; 'll' (id 4) is 2.
    #[test]
    fn test_merge_max_length() {
        let mut a = arena_of(&[0, 1, 2, 2, 3]);
        let token_chars = [1, 1, 1, 1, 2];
        let mut changes = vec![];
        a.merge(0, 2, 2, 4, 2, &token_chars, &mut changes);
        assert_eq!(a.symbols_of(0), &[0, 1, 4, 3]);
        assert_eq!(changes, &[((1, 2), -1), ((2, 3), -1)]);
    }

    // Overlap handling: "aaaa" merging (a,a) -> [aa, aa], with the second
    // occurrence seeing the first merged symbol as its left neighbor.
    #[test]
    fn test_merge_overlaps() {
        let mut a = arena_of(&[7, 7, 7, 7]);
        let mut changes = vec![];
        a.merge(0, 7, 7, 9, usize::MAX, &[], &mut changes);
        assert_eq!(a.symbols_of(0), &[9, 9]);
        assert_eq!(
            changes,
            &[((7, 7), -1), ((9, 7), 1), ((9, 7), -1), ((9, 9), 1)]
        );
    }

    // "aaa" -> [aa, a]: trailing odd symbol stays.
    #[test]
    fn test_merge_odd_run() {
        let mut a = arena_of(&[7, 7, 7]);
        let mut changes = vec![];
        a.merge(0, 7, 7, 9, usize::MAX, &[], &mut changes);
        assert_eq!(a.symbols_of(0), &[9, 7]);
        assert_eq!(changes, &[((7, 7), -1), ((9, 7), 1)]);
    }

    // Words are independent: merging word 1 leaves words 0 and 2 untouched,
    // and shrunk words never disturb their neighbors' slices.
    #[test]
    fn test_multiple_words_independent() {
        let mut a = WordArena::default();
        a.push_word([5, 5, 6].iter().copied());
        a.push_word([5, 5, 5, 5].iter().copied());
        a.push_word([6, 5, 5].iter().copied());
        let mut changes = vec![];
        a.merge(1, 5, 5, 9, usize::MAX, &[], &mut changes);
        assert_eq!(a.symbols_of(0), &[5, 5, 6]);
        assert_eq!(a.symbols_of(1), &[9, 9]);
        assert_eq!(a.symbols_of(2), &[6, 5, 5]);
    }
}
