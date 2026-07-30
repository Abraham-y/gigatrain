//! Word representation and in-word merge application.
//!
//! Semantics must match HF `tokenizers` `models/bpe/word.rs::Word::merge`
//! exactly (see PARITY.md). HF splices a Vec in place and re-reads neighbors
//! post-splice; we use a write-pointer scan that emits the identical multiset
//! of pair-count deltas without O(n^2) splicing.

pub type Pair = (u32, u32);

#[derive(Clone, Copy, Debug)]
pub struct Symbol {
    pub c: u32,
    /// Length in chars (HF counts 1 per char; merged symbols sum).
    pub len: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Word {
    pub symbols: Vec<Symbol>,
}

impl Word {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            symbols: Vec::with_capacity(capacity),
        }
    }

    pub fn add(&mut self, c: u32) {
        self.symbols.push(Symbol { c, len: 1 });
    }

    /// Merge every non-overlapping occurrence of (c1, c2), left to right,
    /// appending pair-count deltas to `changes` as (pair, delta).
    ///
    /// Parity notes (PARITY.md): the merged pair itself is never decremented;
    /// the left neighbor is read post-merge (may be a symbol produced earlier
    /// in this same pass); the `max_len` guard is strict `<` and applies only
    /// to the newly formed neighbor pairs.
    pub fn merge(
        &mut self,
        c1: u32,
        c2: u32,
        new_id: u32,
        max_len: usize,
        changes: &mut Vec<(Pair, i32)>,
    ) {
        let syms = &mut self.symbols;
        let n = syms.len();
        let mut w = 0;
        let mut k = 0;
        while k < n {
            if k + 1 < n && syms[k].c == c1 && syms[k + 1].c == c2 {
                let new_len = syms[k].len + syms[k + 1].len;
                if w > 0 {
                    let prev = syms[w - 1];
                    changes.push(((prev.c, c1), -1));
                    if prev.len + new_len < max_len {
                        changes.push(((prev.c, new_id), 1));
                    }
                }
                if k + 2 < n {
                    let right = syms[k + 2];
                    changes.push(((c2, right.c), -1));
                    if right.len + new_len < max_len {
                        changes.push(((new_id, right.c), 1));
                    }
                }
                syms[w] = Symbol {
                    c: new_id,
                    len: new_len,
                };
                w += 1;
                k += 2;
            } else {
                syms[w] = syms[k];
                w += 1;
                k += 1;
            }
        }
        syms.truncate(w);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(word: &Word) -> Vec<u32> {
        word.symbols.iter().map(|s| s.c).collect()
    }

    // Port of HF word.rs::tests::test_merge ("hello", merge 'l'+'l' -> 'll').
    #[test]
    fn test_merge() {
        let mut word = Word::default();
        for c in [0, 1, 2, 2, 3] {
            word.add(c);
        }
        let mut changes = vec![];
        word.merge(2, 2, 4, usize::MAX, &mut changes);
        assert_eq!(chars(&word), &[0, 1, 4, 3]);
        assert_eq!(
            changes,
            &[((1, 2), -1), ((1, 4), 1), ((2, 3), -1), ((4, 3), 1)]
        );
    }

    // Port of HF word.rs::tests::test_merge_max_length (max_length = 2).
    #[test]
    fn test_merge_max_length() {
        let mut word = Word::default();
        for c in [0, 1, 2, 2, 3] {
            word.add(c);
        }
        let mut changes = vec![];
        word.merge(2, 2, 4, 2, &mut changes);
        assert_eq!(chars(&word), &[0, 1, 4, 3]);
        assert_eq!(changes, &[((1, 2), -1), ((2, 3), -1)]);
    }

    // Overlap handling: "aaaa" merging (a,a) -> [aa, aa], with the second
    // occurrence seeing the first merged symbol as its left neighbor.
    #[test]
    fn test_merge_overlaps() {
        let mut word = Word::default();
        for c in [7, 7, 7, 7] {
            word.add(c);
        }
        let mut changes = vec![];
        word.merge(7, 7, 9, usize::MAX, &mut changes);
        assert_eq!(chars(&word), &[9, 9]);
        // occ 1: right neighbor = third 'a': (a,a)-1, (aa,a)+1
        // occ 2: left neighbor = merged 'aa': (aa,a)-1, (aa,aa)+1
        assert_eq!(
            changes,
            &[((7, 7), -1), ((9, 7), 1), ((9, 7), -1), ((9, 9), 1)]
        );
    }

    // "aaa" -> [aa, a]: trailing odd symbol stays.
    #[test]
    fn test_merge_odd_run() {
        let mut word = Word::default();
        for c in [7, 7, 7] {
            word.add(c);
        }
        let mut changes = vec![];
        word.merge(7, 7, 9, usize::MAX, &mut changes);
        assert_eq!(chars(&word), &[9, 7]);
        assert_eq!(changes, &[((7, 7), -1), ((9, 7), 1)]);
    }
}
