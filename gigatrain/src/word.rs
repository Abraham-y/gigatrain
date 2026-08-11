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
//! - HF stores a per-symbol `len` (chars covered, summed on merge). With no
//!   decoration that equals the char count of the symbol's token string, so
//!   symbols can be bare `u32` ids and the length need not be stored at all.
//!
//!   That equivalence **breaks** once `continuing_subword_prefix` or
//!   `end_of_word_suffix` is set, because a token id is then reachable both
//!   as an initial symbol (covering one source char) and as a merged token
//!   (covering several) — via a special token that spells a decorated
//!   symbol, or via merge-time id reuse once the prefix is stripped. HF is
//!   immune because it stores the length per symbol, not per id. An earlier
//!   version of this file kept a per-id table and was wrong for exactly that
//!   reason.
//!
//!   So `sym_lens` is a parallel per-symbol array, populated only when
//!   `max_token_length` is set — the sole consumer of the value.

pub type Pair = (u32, u32);

#[derive(Default)]
pub struct WordArena {
    symbols: Vec<u32>,
    /// Source characters covered by each symbol, parallel to `symbols`. Empty
    /// unless `max_token_length` is in use; see the module comment.
    sym_lens: Vec<u32>,
    /// Per word: offset into `symbols` (fixed) and current length (shrinks).
    starts: Vec<u32>,
    lens: Vec<u32>,
    track_lens: bool,
}

impl WordArena {
    /// `track_lens` must be true whenever `max_token_length` will be applied.
    pub fn with_capacity(words: usize, symbols: usize, track_lens: bool) -> Self {
        Self {
            symbols: Vec::with_capacity(symbols),
            sym_lens: if track_lens {
                Vec::with_capacity(symbols)
            } else {
                Vec::new()
            },
            starts: Vec::with_capacity(words),
            lens: Vec::with_capacity(words),
            track_lens,
        }
    }

    /// Append a word built from `ids` (chars already mapped to vocab IDs).
    pub fn push_word(&mut self, ids: impl Iterator<Item = u32>) {
        let start = self.symbols.len();
        assert!(
            start <= u32::MAX as usize,
            "symbol arena exceeds 4G symbols"
        );
        self.symbols.extend(ids);
        if self.track_lens {
            // Every initial symbol covers exactly one source character,
            // whatever decoration its token string carries.
            self.sym_lens.resize(self.symbols.len(), 1);
        }
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
    /// `max_len` of `usize::MAX` disables the length guard, in which case
    /// per-symbol lengths are neither tracked nor read.
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
        changes: &mut Vec<(Pair, i32)>,
    ) {
        let start = self.starts[i] as usize;
        let n = self.lens[i] as usize;
        let unbounded = max_len == usize::MAX;

        let mut w = 0;
        let mut k = 0;
        while k < n {
            let a = self.symbols[start + k];
            let pair_here = k + 1 < n && a == c1 && self.symbols[start + k + 1] == c2;
            if pair_here {
                // Length of the symbol being formed, summed as HF does.
                let new_chars = if unbounded {
                    0
                } else {
                    (self.sym_lens[start + k] + self.sym_lens[start + k + 1]) as usize
                };
                if w > 0 {
                    let prev = self.symbols[start + w - 1];
                    changes.push(((prev, c1), -1));
                    if unbounded || self.sym_lens[start + w - 1] as usize + new_chars < max_len {
                        changes.push(((prev, new_id), 1));
                    }
                }
                if k + 2 < n {
                    let right = self.symbols[start + k + 2];
                    changes.push(((c2, right), -1));
                    if unbounded || self.sym_lens[start + k + 2] as usize + new_chars < max_len {
                        changes.push(((new_id, right), 1));
                    }
                }
                self.symbols[start + w] = new_id;
                if !unbounded {
                    self.sym_lens[start + w] = new_chars as u32;
                }
                w += 1;
                k += 2;
            } else {
                self.symbols[start + w] = a;
                if !unbounded {
                    self.sym_lens[start + w] = self.sym_lens[start + k];
                }
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

    // track_lens on, so the max_len tests have per-symbol lengths; every
    // initial symbol covers one source char (see push_word).
    fn arena_of(ids: &[u32]) -> WordArena {
        let mut a = WordArena::with_capacity(1, ids.len(), true);
        a.push_word(ids.iter().copied());
        a
    }

    // Port of HF word.rs::tests::test_merge ("hello", merge 'l'+'l' -> 'll').
    #[test]
    fn test_merge() {
        let mut a = arena_of(&[0, 1, 2, 2, 3]);
        let mut changes = vec![];
        a.merge(0, 2, 2, 4, usize::MAX, &mut changes);
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
        let mut changes = vec![];
        a.merge(0, 2, 2, 4, 2, &mut changes);
        assert_eq!(a.symbols_of(0), &[0, 1, 4, 3]);
        assert_eq!(changes, &[((1, 2), -1), ((2, 3), -1)]);
    }

    // Overlap handling: "aaaa" merging (a,a) -> [aa, aa], with the second
    // occurrence seeing the first merged symbol as its left neighbor.
    #[test]
    fn test_merge_overlaps() {
        let mut a = arena_of(&[7, 7, 7, 7]);
        let mut changes = vec![];
        a.merge(0, 7, 7, 9, usize::MAX, &mut changes);
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
        a.merge(0, 7, 7, 9, usize::MAX, &mut changes);
        assert_eq!(a.symbols_of(0), &[9, 7]);
        assert_eq!(changes, &[((7, 7), -1), ((9, 7), 1)]);
    }

    // Words are independent: merging word 1 leaves words 0 and 2 untouched,
    // and shrunk words never disturb their neighbors' slices.
    #[test]
    fn test_multiple_words_independent() {
        let mut a = WordArena::with_capacity(3, 11, true);
        a.push_word([5, 5, 6].iter().copied());
        a.push_word([5, 5, 5, 5].iter().copied());
        a.push_word([6, 5, 5].iter().copied());
        let mut changes = vec![];
        a.merge(1, 5, 5, 9, usize::MAX, &mut changes);
        assert_eq!(a.symbols_of(0), &[5, 5, 6]);
        assert_eq!(a.symbols_of(1), &[9, 9]);
        assert_eq!(a.symbols_of(2), &[6, 5, 5]);
    }
}
