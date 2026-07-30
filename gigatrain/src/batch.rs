//! Packed batch of routed words, the unit of transfer from scanner threads to
//! shard-owner threads in phase 1.
//!
//! Words are packed into one byte buffer with parallel offset/hash arrays, so
//! shipping a few hundred thousand words between threads costs three
//! allocations rather than one per word.

pub struct WordBatch {
    data: Vec<u8>,
    offsets: Vec<u32>,
    hashes: Vec<u64>,
}

impl Default for WordBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl WordBatch {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            offsets: vec![0],
            hashes: Vec::new(),
        }
    }

    #[inline]
    pub fn push(&mut self, word: &str, hash: u64) {
        self.data.extend_from_slice(word.as_bytes());
        self.offsets.push(self.data.len() as u32);
        self.hashes.push(hash);
    }

    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    pub fn bytes(&self) -> usize {
        self.data.len()
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.offsets.clear();
        self.offsets.push(0);
        self.hashes.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, u64)> + '_ {
        (0..self.len()).map(move |i| {
            let (a, b) = (self.offsets[i] as usize, self.offsets[i + 1] as usize);
            // Safety: only whole &str are appended, at recorded boundaries.
            let s = unsafe { std::str::from_utf8_unchecked(&self.data[a..b]) };
            (s, self.hashes[i])
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_and_round_trips() {
        let mut b = WordBatch::new();
        b.push("alpha", 1);
        b.push("β", 2);
        b.push("", 3);
        assert_eq!(b.len(), 3);
        assert_eq!(
            b.iter().collect::<Vec<_>>(),
            vec![("alpha", 1), ("β", 2), ("", 3)]
        );
        b.clear();
        assert!(b.is_empty());
        assert_eq!(b.iter().count(), 0);
        b.push("again", 9);
        assert_eq!(b.iter().collect::<Vec<_>>(), vec![("again", 9)]);
    }
}
