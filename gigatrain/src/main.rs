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
//!   --pretokenizer MODE     whitespace (default) or bytelevel
//!   --continuing-subword-prefix S   e.g. "##" (WordPiece)
//!   --end-of-word-suffix S          e.g. "</w>"
//!   --wordpiece             shorthand for --continuing-subword-prefix "##"
//!   --words-tsv FILE        word<TAB>count table instead of raw text
//!
//! Raw text mode pretokenizes with whitespace splitting, byte-for-byte
//! equivalent to HF's WhitespaceSplit. Files are streamed in 32MB chunks
//! (never fully resident) and counted across threads.

use gigatrain::{train, TrainerConfig, WordCounter, WordTable};
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
    let mut threads: Option<usize> = None;
    let mut bytelevel = false;
    let mut inputs: Vec<String> = vec![];

    let mut args = std::env::args().skip(1);
    // Parse helpers report like every other error path rather than panicking.
    fn num<T: std::str::FromStr>(flag: &str, raw: &str) -> T {
        raw.parse()
            .unwrap_or_else(|_| die(&format!("{flag} expects a number, got {raw:?}")))
    }
    while let Some(arg) = args.next() {
        let mut val = |name: &str| {
            args.next()
                .unwrap_or_else(|| die(&format!("{name} requires a value")))
        };
        match arg.as_str() {
            "--vocab-size" => vocab_size = Some(num("--vocab-size", &val("--vocab-size"))),
            "--min-frequency" => {
                config.min_frequency = num("--min-frequency", &val("--min-frequency"))
            }
            "--special" => config.special_tokens.push(val("--special")),
            "--max-token-length" => {
                config.max_token_length =
                    Some(num("--max-token-length", &val("--max-token-length")))
            }
            "--limit-alphabet" => {
                config.limit_alphabet = Some(num("--limit-alphabet", &val("--limit-alphabet")))
            }
            "--threads" => threads = Some(num("--threads", &val("--threads"))),
            "--pretokenizer" => {
                bytelevel = match val("--pretokenizer").as_str() {
                    "bytelevel" => true,
                    "whitespace" => false,
                    other => die(&format!(
                        "unknown --pretokenizer {other} (want whitespace or bytelevel)"
                    )),
                }
            }
            "--continuing-subword-prefix" => {
                config.continuing_subword_prefix = Some(val("--continuing-subword-prefix"))
            }
            "--end-of-word-suffix" => config.end_of_word_suffix = Some(val("--end-of-word-suffix")),
            // HF's WordPieceTrainer is exactly BpeTrainer with this prefix.
            "--wordpiece" => config.continuing_subword_prefix = Some("##".to_string()),
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
        let data =
            std::fs::read_to_string(path).unwrap_or_else(|e| die(&format!("reading {path}: {e}")));
        let mut acc = WordCounter::new();
        for line in data.lines() {
            if line.is_empty() {
                continue;
            }
            let (word, count) = line
                .rsplit_once('\t')
                .unwrap_or_else(|| die(&format!("bad TSV line: {line:?}")));
            acc.add(word, num("--words-tsv count", count));
        }
        acc.into_table()
    } else {
        if inputs.is_empty() {
            die("no input files (or --words-tsv) given");
        }
        gigatrain::pipeline::count_words(&inputs, nthreads, bytelevel).unwrap_or_else(|e| die(&e))
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
