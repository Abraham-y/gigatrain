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
//!   --vocab-out FILE        write the vocabulary as a JSON array in id order
//!
//! Raw text mode pretokenizes with whitespace splitting, byte-for-byte
//! equivalent to HF's WhitespaceSplit. Files are read in chunks sized from
//! the input and counted across threads. Chunks are cut at token boundaries,
//! so a stretch of input containing none — no whitespace at all, or no
//! newline under --pretokenizer bytelevel — is buffered whole.
//!
//! The cost of that is mostly *time*, not memory: the buffer grows to the
//! longest boundary-free run (2 GB one-liner peaked at 1.1x the file), but
//! every reader range except the first finds no boundary and retires, so
//! phase 1 collapses to a single thread. Measured 2.0x slower than the same
//! bytes with newlines under ByteLevel; under whitespace the whole file is one
//! word and a 2 GB case did not finish in an hour. See BENCHMARKS.md,
//! "Boundary-free input".

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
    let mut saw_prefix = false;
    let mut saw_wordpiece = false;
    let mut vocab_out: Option<String> = None;

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
                if saw_wordpiece {
                    die("--wordpiece and --continuing-subword-prefix both set the \
                         continuing-subword prefix; pass only one");
                }
                saw_prefix = true;
                config.continuing_subword_prefix = Some(val("--continuing-subword-prefix"))
            }
            "--end-of-word-suffix" => config.end_of_word_suffix = Some(val("--end-of-word-suffix")),
            // HF's WordPieceTrainer is exactly BpeTrainer with this prefix.
            "--wordpiece" => {
                if saw_prefix {
                    die("--wordpiece and --continuing-subword-prefix both set the \
                         continuing-subword prefix; pass only one");
                }
                saw_wordpiece = true;
                config.continuing_subword_prefix = Some("##".to_string())
            }
            "--words-tsv" => words_tsv = Some(val("--words-tsv")),
            "--vocab-out" => vocab_out = Some(val("--vocab-out")),
            _ if arg.starts_with("--") => die(&format!("unknown flag {arg}")),
            _ => inputs.push(arg),
        }
    }
    config.vocab_size = vocab_size.unwrap_or_else(|| die("--vocab-size is required"));

    // Merges are printed as "left<space>right", so any token containing a space
    // yields a line no parser can split back into a pair. `--words-tsv` already
    // rejects such words below; a decoration is prepended/appended to *every*
    // token, so a space there corrupts the whole merge list rather than one
    // line. The Python API returns pairs directly and is unaffected.
    for (flag, value) in [
        (
            "--continuing-subword-prefix",
            &config.continuing_subword_prefix,
        ),
        ("--end-of-word-suffix", &config.end_of_word_suffix),
    ] {
        if let Some(v) = value {
            if v.contains(' ') {
                die(&format!(
                    "{flag} {v:?} contains a space, which the space-separated \
                     merge output cannot represent; use the Python API, which \
                     returns merges as pairs"
                ));
            }
        }
    }
    // `--threads` was the one numeric flag with no validation: large values
    // panicked (thread-spawn failure, capacity overflow, or an arithmetic
    // overflow inside the sizing) instead of reporting like every other flag.
    // 0 means "use every core", matching the Python binding, which treats it
    // as the auto sentinel.
    if let Some(t) = threads {
        if t > gigatrain::pipeline::MAX_WORKERS {
            die(&format!(
                "--threads {t} exceeds the maximum supported ({}); \
                 omit the flag to use every core",
                gigatrain::pipeline::MAX_WORKERS
            ));
        }
    }
    let nthreads = threads.filter(|&t| t > 0).unwrap_or_else(|| {
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
            let count: u64 = num("--words-tsv count", count);
            // Counts are summed as i64 internally; a value past i64::MAX would
            // go negative and the word would silently never be queued for
            // merging rather than being counted.
            if count > i64::MAX as u64 {
                die(&format!(
                    "--words-tsv count {count} exceeds the maximum supported ({})",
                    i64::MAX
                ));
            }
            // Merges are printed space-separated, so a token containing a
            // space would produce a line no parser can split correctly.
            if word.contains(' ') {
                die(&format!(
                    "--words-tsv word {word:?} contains a space, which the \
                     space-separated merge output cannot represent; use the \
                     Python API, which returns merges as pairs"
                ));
            }
            acc.add(word, count);
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

    // The merge list alone does not pin the vocabulary: ids, specials and the
    // alphabet can drift while every merge stays identical. Written as a JSON
    // array in id order so the parity harness can diff it directly.
    if let Some(path) = &vocab_out {
        let mut buf = String::from("[");
        for (i, token) in result.vocab.iter().enumerate() {
            if i > 0 {
                buf.push(',');
            }
            buf.push_str(&gigatrain::tokenizer_json::json_string(token));
        }
        buf.push(']');
        std::fs::write(path, buf).unwrap_or_else(|e| die(&format!("writing {path}: {e}")));
    }

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
