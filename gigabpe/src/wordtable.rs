//! Compact word-frequency table: the output of phase 1, the input to phase 2.
//!
//! Words live in one flat byte arena with u32 offsets rather than as
//! individual `String`s. At web scale the table holds tens of millions of
//! unique pretokens, where per-`String` overhead (24-byte header + heap
//! allocation + allocator rounding, roughly 3-5x the payload for short words)
//! is a substantial fraction of peak RSS.
//!
//! The trainer takes this by value and drops it as soon as words are tokenized
//! into symbol IDs, so word strings do not stay resident through the merge
//! loop.

pub struct WordTable {
    arena: Vec<u8>,
    /// n+1 offsets into `arena`; word i is arena[offsets[i]..offsets[i+1]].
    offsets: Vec<u32>,
    counts: Vec<u64>,
}

impl Default for WordTable {
    fn default() -> Self {
        Self::new()
    }
}

impl WordTable {
    pub fn new() -> Self {
        Self {
            arena: Vec::new(),
            offsets: vec![0],
            counts: Vec::new(),
        }
    }

    pub fn with_capacity(words: usize, bytes: usize) -> Self {
        let mut offsets = Vec::with_capacity(words + 1);
        offsets.push(0);
        Self {
            arena: Vec::with_capacity(bytes),
            offsets,
            counts: Vec::with_capacity(words),
        }
    }

    /// Append a word. Caller guarantees uniqueness (phase 1 dedups via hash
    /// map); this type is a store, not a set.
    pub fn push(&mut self, word: &str, count: u64) {
        self.arena.extend_from_slice(word.as_bytes());
        assert!(
            self.arena.len() <= u32::MAX as usize,
            "unique word bytes exceed 4 GB; recompile with u64 offsets"
        );
        self.offsets.push(self.arena.len() as u32);
        self.counts.push(count);
    }

    pub fn len(&self) -> usize {
        self.counts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    pub fn total_bytes(&self) -> usize {
        self.arena.len()
    }

    #[inline]
    pub fn word(&self, i: usize) -> &str {
        let (a, b) = (self.offsets[i] as usize, self.offsets[i + 1] as usize);
        // Safety: the arena only ever receives whole &str, appended
        // contiguously, and offsets are recorded at str boundaries.
        unsafe { std::str::from_utf8_unchecked(&self.arena[a..b]) }
    }

    #[inline]
    pub fn count(&self, i: usize) -> u64 {
        self.counts[i]
    }

    #[inline]
    pub fn bump(&mut self, i: usize, delta: u64) {
        self.counts[i] += delta;
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, u64)> + '_ {
        (0..self.len()).map(move |i| (self.word(i), self.counts[i]))
    }
}

impl FromIterator<(String, u64)> for WordTable {
    fn from_iter<I: IntoIterator<Item = (String, u64)>>(iter: I) -> Self {
        let mut table = WordTable::new();
        for (word, count) in iter {
            table.push(&word, count);
        }
        table
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_words_and_counts() {
        let mut t = WordTable::new();
        t.push("hello", 3);
        t.push("", 1);
        t.push("héllo→", 7);
        assert_eq!(t.len(), 3);
        assert_eq!(t.word(0), "hello");
        assert_eq!(t.word(1), "");
        assert_eq!(t.word(2), "héllo→");
        assert_eq!(t.count(2), 7);
        assert_eq!(
            t.iter().collect::<Vec<_>>(),
            vec![("hello", 3), ("", 1), ("héllo→", 7)]
        );
    }
}
