//! BPE trainer with HF `tokenizers` byte-exact output parity.
//!
//! Every semantic here mirrors `tokenizers/src/models/bpe/trainer.rs` at
//! v0.22.2 — see PARITY.md for the full list of behaviors that must be
//! preserved, including the ones that look like bugs (stale count residue,
//! partial position sets, duplicate merges).
//!
//! Representation choices that differ from HF without affecting output:
//! position lists are Vec<u32> instead of HashSet<usize> (deduped on push:
//! within one merge step words are processed one at a time, so duplicate
//! inserts for a pair are always adjacent); maps use FxHash. Both only
//! change iteration order / memory, which the algorithm is insensitive to.

use crate::fxhash::FxHashMap;
use crate::word::{Pair, Word};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

pub struct TrainerConfig {
    pub vocab_size: usize,
    pub min_frequency: u64,
    pub special_tokens: Vec<String>,
    pub limit_alphabet: Option<usize>,
    pub initial_alphabet: Vec<char>,
    pub max_token_length: Option<usize>,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30000,
            min_frequency: 0,
            special_tokens: vec![],
            limit_alphabet: None,
            initial_alphabet: vec![],
            max_token_length: None,
        }
    }
}

pub struct TrainResult {
    /// Token string per vocab ID.
    pub vocab: Vec<String>,
    /// Raw merge log in creation order; may contain duplicate pairs.
    pub merges: Vec<(Pair, u32)>,
}

impl TrainResult {
    /// The serialized merge list as HF emits it in tokenizer.json:
    /// unique pairs, deduplicated keeping the LAST rank, sorted by rank.
    pub fn serialized_merges(&self) -> Vec<(String, String)> {
        let mut last_rank: FxHashMap<Pair, usize> = FxHashMap::default();
        for (rank, (pair, _)) in self.merges.iter().enumerate() {
            last_rank.insert(*pair, rank);
        }
        let mut out: Vec<(usize, Pair)> = last_rank.into_iter().map(|(p, r)| (r, p)).collect();
        out.sort_unstable_by_key(|&(r, _)| r);
        out.into_iter()
            .map(|(_, (a, b))| {
                (
                    self.vocab[a as usize].clone(),
                    self.vocab[b as usize].clone(),
                )
            })
            .collect()
    }
}

struct MergeEntry {
    pair: Pair,
    count: u64,
    pos: Vec<u32>,
}

impl PartialEq for MergeEntry {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}
impl Eq for MergeEntry {}
impl PartialOrd for MergeEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MergeEntry {
    // HF tie-break: higher count wins; on ties the SMALLER id pair wins.
    fn cmp(&self, other: &Self) -> Ordering {
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            other.pair.cmp(&self.pair)
        }
    }
}

/// Push word index `i`, skipping if it's already the last element. Within a
/// merge step words are processed one at a time, so duplicates are always
/// adjacent and this fully dedups (matching HF's HashSet semantics).
#[inline]
fn push_pos(pos: &mut Vec<u32>, i: u32) {
    if pos.last() != Some(&i) {
        pos.push(i);
    }
}

/// Train BPE over a word-frequency table. `word_counts` order is irrelevant
/// to the output (it only determines internal word indices).
pub fn train(word_counts: &[(String, u64)], config: &TrainerConfig) -> TrainResult {
    let max_token_length = config.max_token_length.unwrap_or(usize::MAX);

    let mut word_to_id: FxHashMap<String, u32> = FxHashMap::default();
    let mut id_to_word: Vec<String> = Vec::with_capacity(config.vocab_size);

    // 1. Special tokens, in order, deduplicated.
    for token in &config.special_tokens {
        if !word_to_id.contains_key(token) {
            id_to_word.push(token.clone());
            word_to_id.insert(token.clone(), (id_to_word.len() - 1) as u32);
        }
    }

    // 2. Alphabet: chars weighted by word count; initial_alphabet forced to
    //    usize::MAX; limit_alphabet drops lowest-count chars; kept chars are
    //    ID-ordered by codepoint.
    {
        let mut alphabet: FxHashMap<char, usize> = FxHashMap::default();
        for (word, count) in word_counts {
            for c in word.chars() {
                *alphabet.entry(c).or_default() += *count as usize;
            }
        }
        for c in &config.initial_alphabet {
            alphabet.insert(*c, usize::MAX);
        }

        let mut kept: Vec<(char, usize)> = alphabet.into_iter().collect();
        let to_remove = config
            .limit_alphabet
            .map(|limit| kept.len().saturating_sub(limit))
            .unwrap_or(0);
        if to_remove > 0 {
            // HF sorts unstably by count only; ties at the cutoff are
            // nondeterministic in HF itself. We add the char as a secondary
            // key so at least our own output is deterministic.
            kept.sort_unstable_by_key(|&(c, count)| (count, c));
            kept.drain(..to_remove);
        }

        kept.sort_unstable_by_key(|&(c, _)| c as u32);
        for (c, _) in kept {
            let s = c.to_string();
            if !word_to_id.contains_key(&s) {
                id_to_word.push(s.clone());
                word_to_id.insert(s, (id_to_word.len() - 1) as u32);
            }
        }
    }

    // Fast char -> id view of the vocab (valid because we support no
    // continuing_subword_prefix/end_of_word_suffix, so tokenization only
    // ever looks up single-char strings; single-char special tokens are
    // in word_to_id and therefore in this map too).
    let char_to_id: FxHashMap<char, u32> = word_to_id
        .iter()
        .filter_map(|(s, &id)| {
            let mut chars = s.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) => Some((c, id)),
                _ => None,
            }
        })
        .collect();

    // 3. Tokenize words into symbol sequences; chars outside the alphabet
    //    (only possible under limit_alphabet) are dropped.
    let mut words: Vec<Word> = Vec::with_capacity(word_counts.len());
    let mut counts: Vec<u64> = Vec::with_capacity(word_counts.len());
    for (word, count) in word_counts {
        counts.push(*count);
        let mut w = Word::with_capacity(word.chars().count());
        for c in word.chars() {
            if let Some(&id) = char_to_id.get(&c) {
                w.add(id);
            }
        }
        words.push(w);
    }

    // 4. Initial pair counts and the inverted index of where each pair lives.
    let mut pair_counts: FxHashMap<Pair, i64> = FxHashMap::default();
    let mut where_to_update: FxHashMap<Pair, Vec<u32>> = FxHashMap::default();
    for (i, word) in words.iter().enumerate() {
        for win in word.symbols.windows(2) {
            let pair = (win[0].c, win[1].c);
            *pair_counts.entry(pair).or_default() += counts[i] as i64;
            push_pos(where_to_update.entry(pair).or_default(), i as u32);
        }
    }

    let mut queue: BinaryHeap<MergeEntry> = BinaryHeap::with_capacity(pair_counts.len());
    for (pair, pos) in where_to_update.drain() {
        let count = pair_counts[&pair];
        if count > 0 {
            queue.push(MergeEntry {
                pair,
                count: count as u64,
                pos,
            });
        }
    }

    // 5. Merge loop.
    let mut merges: Vec<(Pair, u32)> = vec![];
    let mut changes: Vec<(Pair, i32)> = vec![];
    loop {
        if word_to_id.len() >= config.vocab_size {
            break;
        }
        let Some(mut top) = queue.pop() else {
            break;
        };

        let live = *pair_counts.get(&top.pair).unwrap_or(&0);
        if top.count != live as u64 {
            // Stale entry: correct the count and re-push (pos is kept).
            top.count = live as u64;
            queue.push(top);
            continue;
        }
        if top.count < 1 || config.min_frequency > top.count {
            break;
        }

        let part_a = &id_to_word[top.pair.0 as usize];
        let part_b = &id_to_word[top.pair.1 as usize];
        let new_token = format!("{part_a}{part_b}");
        // Reuse the existing ID if this string is already in the vocab.
        let new_token_id = word_to_id
            .get(&new_token)
            .copied()
            .unwrap_or(id_to_word.len() as u32);
        if !word_to_id.contains_key(&new_token) {
            id_to_word.push(new_token.clone());
            word_to_id.insert(new_token, new_token_id);
        }
        merges.push((top.pair, new_token_id));

        // Merge only within this entry's recorded positions (a partial set
        // when the pair re-formed after an earlier merge of it — HF behavior).
        for &i in &top.pos {
            changes.clear();
            words[i as usize].merge(
                top.pair.0,
                top.pair.1,
                new_token_id,
                max_token_length,
                &mut changes,
            );
            for &(pair, change) in &changes {
                *pair_counts.entry(pair).or_default() += change as i64 * counts[i as usize] as i64;
                if change > 0 {
                    push_pos(where_to_update.entry(pair).or_default(), i);
                }
            }
        }

        for (pair, pos) in where_to_update.drain() {
            let count = pair_counts[&pair];
            if count > 0 {
                queue.push(MergeEntry {
                    pair,
                    count: count as u64,
                    pos,
                });
            }
        }
    }

    TrainResult {
        vocab: id_to_word,
        merges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(pairs: &[(&str, u64)]) -> Vec<(String, u64)> {
        pairs.iter().map(|(w, c)| (w.to_string(), *c)).collect()
    }

    // Port of HF trainer.rs::tests::test_train (roses are red...).
    #[test]
    fn test_train_matches_hf() {
        let word_counts = counts(&[
            ("roses", 1),
            ("are", 2),
            ("red", 1),
            ("voilets", 1),
            ("blue", 1),
            ("BERT", 1),
            ("is", 2),
            ("big", 1),
            ("and", 1),
            ("so", 1),
            ("GPT-2", 1),
        ]);
        let config = TrainerConfig {
            min_frequency: 2,
            vocab_size: 30000,
            ..Default::default()
        };
        let result = train(&word_counts, &config);

        let expected_vocab: Vec<&str> = vec![
            "-", "2", "B", "E", "G", "P", "R", "T", "a", "b", "d", "e", "g", "i", "l", "n", "o",
            "r", "s", "t", "u", "v", "re", "are", "is",
        ];
        assert_eq!(result.vocab, expected_vocab);
        // ('r','e')->'re', ('a','re')->'are', ('i','s')->'is'
        assert_eq!(
            result.merges,
            vec![((17, 11), 22), ((8, 22), 23), ((13, 18), 24)]
        );
    }

    // Port of HF trainer.rs::tests::bpe_test_max_token_length_direct_assert.
    #[test]
    fn test_max_token_length_direct_assert() {
        let word_counts = counts(&[
            ("sin", 2),
            ("Sin", 2),
            ("Lon", 2),
            ("Ano", 2),
            ("짧은한", 2),
            ("긴한글", 2),
            ("短字符", 2),
            ("长字符", 2),
            ("短い文", 2),
            ("長い文", 2),
            ("so", 2),
            ("GP", 2),
        ]);
        let config = TrainerConfig {
            max_token_length: Some(2),
            min_frequency: 0,
            vocab_size: 30000,
            ..Default::default()
        };
        let result = train(&word_counts, &config);

        let mut expected: Vec<(&str, u32)> = vec![
            ("短", 12),
            ("n", 6),
            ("i", 5),
            ("s", 8),
            ("字符", 23),
            ("長", 14),
            ("긴", 17),
            ("い文", 22),
            ("L", 2),
            ("in", 21),
            ("o", 7),
            ("은한", 29),
            ("S", 4),
            ("P", 3),
            ("so", 27),
            ("符", 13),
            ("文", 11),
            ("字", 10),
            ("짧", 19),
            ("GP", 25),
            ("글", 16),
            ("G", 1),
            ("An", 24),
            ("长", 15),
            ("A", 0),
            ("Lo", 26),
            ("긴한", 28),
            ("い", 9),
            ("한", 20),
            ("은", 18),
        ];
        expected.sort_by_key(|&(_, id)| id);
        let expected_vocab: Vec<&str> = expected.iter().map(|&(s, _)| s).collect();
        assert_eq!(result.vocab, expected_vocab);
    }

    // max_token_length only filters NEW pairs: 2-char tokens always form.
    #[test]
    fn test_max_token_length_two() {
        let word_counts = counts(&[("aaaa", 4)]);
        let config = TrainerConfig {
            max_token_length: Some(2),
            vocab_size: 30000,
            ..Default::default()
        };
        let result = train(&word_counts, &config);
        assert_eq!(result.vocab, vec!["a", "aa"]);
        assert_eq!(result.merges, vec![((0, 0), 1)]);
    }

    // Tie-break: equal counts pick the smallest ID pair. With words "ba" and
    // "dc" at equal counts, pairs (b,a)=(1,0) and (d,c)=(3,2): (1,0) < (3,2).
    #[test]
    fn test_tie_break_smallest_pair() {
        let word_counts = counts(&[("ba", 3), ("dc", 3)]);
        let config = TrainerConfig {
            vocab_size: 5,
            ..Default::default()
        };
        let result = train(&word_counts, &config);
        // alphabet: a=0 b=1 c=2 d=3; first merge must be (1,0) -> "ba"
        assert_eq!(result.merges[0], ((1, 0), 4));
    }
}
