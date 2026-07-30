//! Arena-backed word counter: the phase-1 accumulator.
//!
//! A `HashMap<String, u64>` costs one heap allocation per unique word plus a
//! 24-byte header, and on a web corpus the unique-word count runs to tens of
//! millions. Measured on 1 GB of FineWeb (4.1M unique words), the per-worker
//! `HashMap<String, u64>` accumulators peaked at 1.7 GB — the single largest
//! consumer in the whole trainer, larger than the entire merge phase.
//!
//! This stores words in the flat `WordTable` arena and keeps only a
//! `hash -> word id` index with intrusive collision chains, so a word costs
//! its bytes plus ~16 bytes of bookkeeping and no individual allocation.

use crate::fxhash::{FxHashMap, FxHasher};
use crate::wordtable::WordTable;
use std::hash::Hasher;

const NONE: u32 = u32::MAX;

#[derive(Default)]
pub struct WordCounter {
    table: WordTable,
    /// Per word: next word id in the same hash bucket, or NONE.
    next: Vec<u32>,
    /// Full 64-bit hash -> most recently added word id with that hash.
    index: FxHashMap<u64, u32>,
}

#[inline]
pub fn hash_word(word: &str) -> u64 {
    let mut h = FxHasher::default();
    h.write(word.as_bytes());
    h.finish()
}

impl WordCounter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    pub fn add(&mut self, word: &str, count: u64) {
        self.add_hashed(word, hash_word(word), count);
    }

    /// Add with a precomputed `hash_word` value, so callers that already
    /// hashed for shard routing don't hash twice.
    pub fn add_hashed(&mut self, word: &str, h: u64, count: u64) {
        if let Some(&head) = self.index.get(&h) {
            let mut id = head;
            loop {
                if self.table.word(id as usize) == word {
                    self.table.bump(id as usize, count);
                    return;
                }
                let n = self.next[id as usize];
                if n == NONE {
                    break;
                }
                id = n;
            }
            // Distinct word with a colliding hash: prepend to the chain.
            let new_id = self.table.len() as u32;
            self.table.push(word, count);
            self.next.push(head);
            self.index.insert(h, new_id);
        } else {
            let new_id = self.table.len() as u32;
            assert!(new_id != NONE, "more than u32::MAX-1 unique words");
            self.table.push(word, count);
            self.next.push(NONE);
            self.index.insert(h, new_id);
        }
    }

    pub fn total_bytes(&self) -> usize {
        self.table.total_bytes()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, u64)> + '_ {
        self.table.iter()
    }

    /// Drop the index, keeping just the compact table.
    pub fn into_table(self) -> WordTable {
        self.table
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_and_dedups() {
        let mut c = WordCounter::new();
        for w in ["a", "b", "a", "cc", "a", "b"] {
            c.add(w, 1);
        }
        c.add("cc", 10);
        assert_eq!(c.len(), 3);
        let mut got: Vec<(&str, u64)> = c.iter().collect();
        got.sort();
        assert_eq!(got, vec![("a", 3), ("b", 2), ("cc", 11)]);
    }

    #[test]
    fn handles_unicode_and_empty() {
        let mut c = WordCounter::new();
        c.add("héllo→", 2);
        c.add("héllo→", 3);
        c.add("h", 1);
        assert_eq!(c.len(), 2);
        let table = c.into_table();
        assert_eq!(table.len(), 2);
        assert_eq!(table.word(0), "héllo→");
        assert_eq!(table.count(0), 5);
    }

    // Words sharing a hash bucket must stay distinct (chain walk correctness).
    #[test]
    fn many_words_stay_distinct() {
        let mut c = WordCounter::new();
        let words: Vec<String> = (0..5000).map(|i| format!("w{i}")).collect();
        for w in &words {
            c.add(w, 1);
        }
        for w in &words {
            c.add(w, 2);
        }
        assert_eq!(c.len(), 5000);
        assert!(c.iter().all(|(_, count)| count == 3));
    }
}
