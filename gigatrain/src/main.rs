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
//!   --threads N             (default: all cores)
//!   --words-tsv FILE        word<TAB>count table instead of raw text
//!
//! Raw text mode pretokenizes with whitespace splitting, byte-for-byte
//! equivalent to HF's WhitespaceSplit. Files are streamed in 32MB chunks
//! (never fully resident) and counted across threads.

use gigatrain::fxhash::FxHashMap;
use gigatrain::{train, TrainerConfig};
use std::io::{Read, Write};
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(2);
}

const CHUNK: usize = 32 << 20;

/// Stream `paths` as whitespace-bounded chunks; count words across
/// `nthreads` workers; merge worker maps in first-seen order.
///
/// Chunks are cut at ASCII whitespace bytes, which are always real word
/// boundaries (UTF-8 continuation bytes are never ASCII, and every ASCII
/// whitespace char satisfies char::is_whitespace).
fn count_words_parallel(paths: &[String], nthreads: usize) -> Vec<(String, u64)> {
    let (tx, rx) = sync_channel::<Vec<u8>>(nthreads * 2);
    let rx = Arc::new(Mutex::new(rx));

    let maps: Vec<FxHashMap<String, u64>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..nthreads)
            .map(|_| {
                let rx = Arc::clone(&rx);
                s.spawn(move || {
                    let mut map: FxHashMap<String, u64> = FxHashMap::default();
                    loop {
                        let chunk = rx.lock().unwrap().recv();
                        let Ok(chunk) = chunk else { break };
                        let text = std::str::from_utf8(&chunk)
                            .unwrap_or_else(|e| die(&format!("input is not UTF-8: {e}")));
                        for w in text.split_whitespace() {
                            match map.get_mut(w) {
                                Some(c) => *c += 1,
                                None => {
                                    map.insert(w.to_owned(), 1);
                                }
                            }
                        }
                    }
                    map
                })
            })
            .collect();

        for path in paths {
            let mut file = std::fs::File::open(path)
                .unwrap_or_else(|e| die(&format!("opening {path}: {e}")));
            let mut carry: Vec<u8> = Vec::new();
            loop {
                let mut buf = std::mem::take(&mut carry);
                let start = buf.len();
                buf.resize(start + CHUNK, 0);
                let mut filled = start;
                while filled < buf.len() {
                    match file.read(&mut buf[filled..]) {
                        Ok(0) => break,
                        Ok(n) => filled += n,
                        Err(e) => die(&format!("reading {path}: {e}")),
                    }
                }
                buf.truncate(filled);
                let eof = filled < start + CHUNK;
                if eof {
                    if !buf.is_empty() {
                        tx.send(buf).unwrap();
                    }
                    break;
                }
                // Cut at the last ASCII whitespace; carry the tail over.
                match buf
                    .iter()
                    .rposition(|&b| matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c))
                {
                    Some(cut) => {
                        carry.extend_from_slice(&buf[cut + 1..]);
                        buf.truncate(cut + 1);
                        tx.send(buf).unwrap();
                    }
                    None => {
                        // No boundary in the whole chunk (pathological token):
                        // keep accumulating.
                        carry = buf;
                    }
                }
            }
        }
        drop(tx);
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut index: FxHashMap<String, usize> = FxHashMap::default();
    let mut word_counts: Vec<(String, u64)> = vec![];
    for map in maps {
        for (word, count) in map {
            match index.get(&word) {
                Some(&i) => word_counts[i].1 += count,
                None => {
                    index.insert(word.clone(), word_counts.len());
                    word_counts.push((word, count));
                }
            }
        }
    }
    word_counts
}

fn main() {
    let mut config = TrainerConfig::default();
    let mut vocab_size: Option<usize> = None;
    let mut words_tsv: Option<String> = None;
    let mut threads: Option<usize> = None;
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
            "--threads" => threads = Some(val("--threads").parse().unwrap()),
            "--words-tsv" => words_tsv = Some(val("--words-tsv")),
            _ if arg.starts_with("--") => die(&format!("unknown flag {arg}")),
            _ => inputs.push(arg),
        }
    }
    config.vocab_size = vocab_size.unwrap_or_else(|| die("--vocab-size is required"));
    let nthreads = threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });

    let t0 = Instant::now();
    let word_counts: Vec<(String, u64)> = if let Some(path) = &words_tsv {
        let data = std::fs::read_to_string(path)
            .unwrap_or_else(|e| die(&format!("reading {path}: {e}")));
        let mut index: FxHashMap<String, usize> = FxHashMap::default();
        let mut word_counts: Vec<(String, u64)> = vec![];
        for line in data.lines() {
            if line.is_empty() {
                continue;
            }
            let (word, count) = line
                .rsplit_once('\t')
                .unwrap_or_else(|| die(&format!("bad TSV line: {line:?}")));
            let count: u64 = count.parse().unwrap();
            match index.get(word) {
                Some(&i) => word_counts[i].1 += count,
                None => {
                    index.insert(word.to_string(), word_counts.len());
                    word_counts.push((word.to_string(), count));
                }
            }
        }
        word_counts
    } else {
        if inputs.is_empty() {
            die("no input files (or --words-tsv) given");
        }
        count_words_parallel(&inputs, nthreads)
    };
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
        "unique words: {}  vocab: {}  merges: {}  threads: {}  phase1: {:.2?}  phase2: {:.2?}",
        word_counts.len(),
        result.vocab.len(),
        result.merges.len(),
        nthreads,
        t_phase1,
        t_phase2,
    );
}
