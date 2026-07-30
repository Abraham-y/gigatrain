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

use gigatrain::batch::WordBatch;
use gigatrain::{train, TrainerConfig, WordCounter, WordTable};
use std::io::{Read, Write};
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(2);
}

/// Chunk size for the reader. Small enough that in-flight chunks stay a
/// negligible share of peak RSS (queue depth x this x ~2 for chunks held by
/// scanners), large enough to amortize the read syscall and boundary scan.
const CHUNK: usize = 4 << 20;

/// Stream `paths` as whitespace-bounded chunks and count words in parallel.
///
/// Three stages, so that no stage's work is replicated and all of it divides
/// across cores:
///
///   reader -> scanners (split + hash + route) -> shard owners (count)
///
/// Words are routed to shard `hash % nshards`, so shards are disjoint: each
/// unique word is stored exactly once machine-wide, and the final combine is
/// a concatenation with no lookups. Two designs were measured and rejected on
/// a 1 GB FineWeb corpus (10 cores):
///
///   - per-worker maps merged at the end: 2.4 s but 1.3 GB peak, since every
///     worker stores its own copy of every frequent word and the union is
///     then merged single-threaded;
///   - broadcasting each chunk to every shard owner: 422 MB but 4.8 s, since
///     splitting and hashing the whole corpus is then replicated per worker.
///     It also barely scales (9.8 s on 1 thread to 4.8 s on 10) because the
///     replicated part is a fixed floor — worse on machines with more cores.
///
/// Scanners and owners are separate thread pools: scanners only send and
/// owners only receive, so bounded channels cannot deadlock.
///
/// Chunks are cut at ASCII whitespace bytes, which are always real word
/// boundaries (UTF-8 continuation bytes are never ASCII, and every ASCII
/// whitespace char satisfies char::is_whitespace).
fn count_words_parallel(paths: &[String], nthreads: usize) -> WordTable {
    // Ship a batch once it reaches this many bytes of packed words.
    const BATCH_BYTES: usize = 64 << 10;

    let (chunk_tx, chunk_rx) = sync_channel::<Arc<Vec<u8>>>(nthreads);
    let chunk_rx = Arc::new(Mutex::new(chunk_rx));

    let mut batch_senders = Vec::with_capacity(nthreads);
    let mut batch_receivers = Vec::with_capacity(nthreads);
    for _ in 0..nthreads {
        let (tx, rx) = sync_channel::<WordBatch>(4);
        batch_senders.push(tx);
        batch_receivers.push(rx);
    }

    let maps: Vec<WordCounter> = std::thread::scope(|s| {
        let owners: Vec<_> = batch_receivers
            .into_iter()
            .map(|rx| {
                s.spawn(move || {
                    let mut map = WordCounter::new();
                    while let Ok(batch) = rx.recv() {
                        for (word, hash) in batch.iter() {
                            map.add_hashed(word, hash, 1);
                        }
                    }
                    map
                })
            })
            .collect();

        let scanners: Vec<_> = (0..nthreads)
            .map(|_| {
                let chunk_rx = Arc::clone(&chunk_rx);
                let batch_senders = batch_senders.clone();
                s.spawn(move || {
                    let nshards = nthreads as u64;
                    let mut batches: Vec<WordBatch> =
                        (0..nthreads).map(|_| WordBatch::new()).collect();
                    loop {
                        let chunk = chunk_rx.lock().unwrap().recv();
                        let Ok(chunk) = chunk else { break };
                        let text = std::str::from_utf8(&chunk)
                            .unwrap_or_else(|e| die(&format!("input is not UTF-8: {e}")));
                        gigatrain::split::for_each_word(text, |w| {
                            let h = gigatrain::counter::hash_word(w);
                            // Route on the high bits: the low bits pick the
                            // hash-map bucket, so reusing them here would
                            // leave each shard's index sparsely populated.
                            let shard = ((h >> 32) % nshards) as usize;
                            batches[shard].push(w, h);
                            if batches[shard].bytes() >= BATCH_BYTES {
                                let full = std::mem::take(&mut batches[shard]);
                                let _ = batch_senders[shard].send(full);
                            }
                        });
                    }
                    for (shard, batch) in batches.into_iter().enumerate() {
                        if !batch.is_empty() {
                            let _ = batch_senders[shard].send(batch);
                        }
                    }
                })
            })
            .collect();
        drop(batch_senders);

        let tx = chunk_tx;
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
                        tx.send(Arc::new(buf)).unwrap();
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
                        tx.send(Arc::new(buf)).unwrap();
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
        for scanner in scanners {
            scanner.join().unwrap();
        }
        owners.into_iter().map(|h| h.join().unwrap()).collect()
    });
    gigatrain::rss::report("phase 1 shard counters");

    // Shards are disjoint, so combining is concatenation: no lookups, no
    // dedup. Each shard's index and arena are freed as it is consumed.
    let mut maps = maps;
    let total_words: usize = maps.iter().map(|m| m.len()).sum();
    let total_bytes: usize = maps.iter().map(|m| m.total_bytes()).sum();
    let mut table = WordTable::with_capacity(total_words, total_bytes);
    for map in maps.drain(..) {
        let shard = map.into_table();
        for i in 0..shard.len() {
            table.push(shard.word(i), shard.count(i));
        }
    }
    gigatrain::rss::report("combining shards");
    table
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
    let word_table: WordTable = if let Some(path) = &words_tsv {
        let data = std::fs::read_to_string(path)
            .unwrap_or_else(|e| die(&format!("reading {path}: {e}")));
        let mut acc = WordCounter::new();
        for line in data.lines() {
            if line.is_empty() {
                continue;
            }
            let (word, count) = line
                .rsplit_once('\t')
                .unwrap_or_else(|| die(&format!("bad TSV line: {line:?}")));
            acc.add(word, count.parse().unwrap());
        }
        acc.into_table()
    } else {
        if inputs.is_empty() {
            die("no input files (or --words-tsv) given");
        }
        count_words_parallel(&inputs, nthreads)
    };
    let t_phase1 = t0.elapsed();
    let word_count = word_table.len();
    gigatrain::rss::report("phase 1 total");

    let t1 = Instant::now();
    let result = train(word_table, &config);
    let t_phase2 = t1.elapsed();

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    for (a, b) in result.serialized_merges() {
        writeln!(out, "{a} {b}").unwrap();
    }
    out.flush().unwrap();

    eprintln!(
        "unique words: {}  vocab: {}  merges: {}  threads: {}  phase1: {:.2?}  phase2: {:.2?}",
        word_count,
        result.vocab.len(),
        result.merges.len(),
        nthreads,
        t_phase1,
        t_phase2,
    );
}
