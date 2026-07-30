//! CLI: train a BPE vocab and print the serialized merge list (HF
//! tokenizer.json order) to stdout, one merge per line as "left right".
//!
//! Usage:
//!   gigatrain --vocab-size N [options] FILE...
//!   gigatrain --vocab-size N [options] --words-tsv COUNTS.tsv
//!
//! Options:
//!   --min-frequency N       (default 0)
//!   --special TOKEN         (repeatable, in order)
//!   --max-token-length N
//!   --limit-alphabet N
//!   --words-tsv FILE        word<TAB>count table instead of raw text
//!
//! Raw text mode pretokenizes with whitespace splitting, byte-for-byte
//! equivalent to HF's WhitespaceSplit.

use gigatrain::{train, TrainerConfig};
use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(2);
}

fn main() {
    let mut config = TrainerConfig::default();
    let mut vocab_size: Option<usize> = None;
    let mut words_tsv: Option<String> = None;
    let mut inputs: Vec<String> = vec![];

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut val = |name: &str| {
            args.next()
                .unwrap_or_else(|| die(&format!("{name} requires a value")))
        };
        match arg.as_str() {
            "--vocab-size" => vocab_size = Some(val("--vocab-size").parse().unwrap()),
            "--min-frequency" => config.min_frequency = val("--min-frequency").parse().unwrap(),
            "--special" => config.special_tokens.push(val("--special")),
            "--max-token-length" => {
                config.max_token_length = Some(val("--max-token-length").parse().unwrap())
            }
            "--limit-alphabet" => {
                config.limit_alphabet = Some(val("--limit-alphabet").parse().unwrap())
            }
            "--words-tsv" => words_tsv = Some(val("--words-tsv")),
            _ if arg.starts_with("--") => die(&format!("unknown flag {arg}")),
            _ => inputs.push(arg),
        }
    }
    config.vocab_size = vocab_size.unwrap_or_else(|| die("--vocab-size is required"));

    let t0 = Instant::now();

    // Phase 1: build the word-frequency table (first-seen order; order does
    // not affect output).
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut word_counts: Vec<(String, u64)> = vec![];
    let mut add = |word: &str, count: u64| match index.get(word) {
        Some(&i) => word_counts[i].1 += count,
        None => {
            index.insert(word.to_string(), word_counts.len());
            word_counts.push((word.to_string(), count));
        }
    };

    if let Some(path) = &words_tsv {
        let data = std::fs::read_to_string(path)
            .unwrap_or_else(|e| die(&format!("reading {path}: {e}")));
        for line in data.lines() {
            if line.is_empty() {
                continue;
            }
            let (word, count) = line
                .rsplit_once('\t')
                .unwrap_or_else(|| die(&format!("bad TSV line: {line:?}")));
            add(word, count.parse().unwrap());
        }
    } else {
        if inputs.is_empty() {
            die("no input files (or --words-tsv) given");
        }
        for path in &inputs {
            let data = std::fs::read_to_string(path)
                .unwrap_or_else(|e| die(&format!("reading {path}: {e}")));
            // Equivalent to HF WhitespaceSplit (split on char::is_whitespace,
            // delimiters removed).
            for word in data.split_whitespace() {
                add(word, 1);
            }
        }
    }
    let t_phase1 = t0.elapsed();

    let t1 = Instant::now();
    let result = train(&word_counts, &config);
    let t_phase2 = t1.elapsed();

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    for (a, b) in result.serialized_merges() {
        writeln!(out, "{a} {b}").unwrap();
    }
    out.flush().unwrap();

    eprintln!(
        "unique words: {}  vocab: {}  merges: {}  phase1: {:.2?}  phase2: {:.2?}",
        word_counts.len(),
        result.vocab.len(),
        result.merges.len(),
        t_phase1,
        t_phase2,
    );
}
