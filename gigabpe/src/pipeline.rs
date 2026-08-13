//! Phase 1: corpus files to a word-frequency table.
//!
//! Three stages connected by bounded channels — reader, scanners, shard
//! owners — so no stage's work is replicated and all of it divides across
//! cores. See ARCHITECTURE.md for the designs that were measured and
//! rejected.

use crate::batch::WordBatch;
use crate::{WordCounter, WordTable};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Mutex};

/// Per-stage nanosecond counters, summed across threads. Only meaningful with
/// GIGABPE_STATS=1; the clock reads cost a few ns each and are skipped
/// otherwise.
#[derive(Default)]
struct StageTimers {
    read: AtomicU64,
    scan: AtomicU64,
    send: AtomicU64,
    insert: AtomicU64,
    recv: AtomicU64,
}

impl StageTimers {
    fn report(&self, scanners: usize, owners: usize, readers: usize) {
        let ms = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e6;
        eprintln!(
            "phase1 cpu-ms (summed across threads):\n  \
             read {:.0} over {} readers\n  \
             scan+hash {:.0} over {} scanners\n  \
             send-blocked {:.0}\n  \
             recv-blocked {:.0}\n  \
             insert {:.0} over {} owners",
            ms(&self.read),
            readers,
            ms(&self.scan),
            scanners,
            ms(&self.send),
            ms(&self.recv),
            ms(&self.insert),
            owners,
        );
    }
}

/// First error seen by any pipeline thread. Phase 1 runs across three thread
/// pools, so a failure is recorded here and surfaced by `count_words` rather
/// than terminating the process: this is library code, and calling
/// `process::exit` from it killed the caller's Python interpreter outright,
/// skipping `finally` blocks and discarding buffered output.
#[derive(Default)]
struct ErrorSlot(Mutex<Option<String>>);

impl ErrorSlot {
    fn set(&self, msg: String) {
        let mut slot = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            *slot = Some(msg);
        }
    }

    fn take(&self) -> Option<String> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).take()
    }
}

/// Upper bound on the worker budget, applied to every entry point.
///
/// Sizing does unchecked arithmetic on the thread count and then spawns what it
/// computes, so an unbounded value is both an overflow hazard and a request to
/// the OS for more threads than it can create. Well above any real core count.
pub const MAX_WORKERS: usize = 4096;

/// Phase-1 pipeline sizing, derived from thread count and input size rather
/// than fixed, so the same binary behaves sensibly from a 1-core container to
/// a 128-core server and from a 1 MB file to a 100 GB one.
struct Sizing {
    /// Bytes per chunk handed to a scanner.
    chunk: usize,
    /// Depth of the reader -> scanner queue.
    chunk_queue: usize,
    /// Bytes a scanner accumulates for one shard before shipping.
    batch: usize,
    /// Number of reader threads (each takes a byte range).
    readers: usize,
    /// Threads splitting and routing (CPU bound).
    scanners: usize,
    /// Threads counting their shard (memory bound). Also the shard count.
    owners: usize,
}

impl Sizing {
    fn plan(total_bytes: u64, nthreads: usize) -> Self {
        // Clamped before any arithmetic: this is a library entry point reachable
        // from the Python bindings as well as the CLI, and the sizing below
        // multiplies `nthreads` several times. Unclamped, `nthreads` near
        // `usize::MAX` overflowed `4 * nthreads` to 0 and tripped `clamp`'s
        // `min <= max` assert, and merely large values asked the OS for
        // billions of threads. No real machine exceeds this.
        let nthreads = nthreads.clamp(1, MAX_WORKERS);

        // Chunks must be small enough that every scanner gets work on a small
        // corpus, and big enough to amortize syscalls on a large one.
        let target_chunks = (nthreads * 8).max(1) as u64;
        let chunk = (total_bytes / target_chunks).clamp(64 << 10, 8 << 20) as usize;

        // Cap bytes in flight rather than fixing the queue depth: depth *
        // chunk is what actually shows up in RSS.
        let chunk_queue = ((64 << 20) / chunk).clamp(2, 4 * nthreads.max(1));

        // Each scanner holds one batch per shard, so total batch memory is
        // O(nthreads^2). Fixing this at 64 KB costs 256 MB at 64 threads and
        // 1 GB at 128. Give each scanner a fixed budget and split it instead.
        let batch = ((4 << 20) / nthreads.max(1)).clamp(4 << 10, 64 << 10);

        // One reader saturates at roughly 700 MB/s here, which becomes the
        // phase-1 bottleneck once enough scanners are consuming. Scale readers
        // with cores, but keep ranges large enough to stay sequential-ish.
        let readers = nthreads
            .clamp(1, 8)
            .min((total_bytes / crate::reader::min_range_bytes()).max(1) as usize);

        // Scanners and owners run concurrently, so sizing both at `nthreads`
        // puts ~2x the core count on the CPU (plus readers). Measured on a
        // 64-core Linux box, 1 GB ByteLevel: 5.51 s at 32 threads against
        // 7.16 s at 64 and 8.25 s at 96, with peak RSS climbing 729 MB ->
        // 1384 MB -> 1804 MB. Splitting the budget puts the default at the
        // measured optimum rather than past it.
        //
        // `nthreads` is the worker budget and is split between the two pools,
        // so `--threads 64` really does run 64 workers rather than 128. A
        // pipeline needs at least one of each, so `--threads 1` still spawns
        // two.
        // Instrumented on 1 GB ByteLevel, scan+hash costs ~7.6 s of CPU
        // against ~5.5 s for the owners' inserts, which suggests a 60/40
        // split. Measured on 64 cores it made no difference (4.7 s either
        // way), so the even split stays.
        let scanners = (nthreads / 2).max(1);
        let owners = nthreads.saturating_sub(scanners).max(1);

        Sizing {
            chunk,
            chunk_queue,
            batch,
            readers,
            scanners,
            owners,
        }
    }
}

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
pub fn count_words(
    paths: &[String],
    nthreads: usize,
    bytelevel: bool,
) -> Result<WordTable, String> {
    let total_bytes: u64 = paths
        .iter()
        .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum();
    let sizing = Sizing::plan(total_bytes, nthreads);
    let nshards_u64 = sizing.owners as u64;
    let timers = Arc::new(StageTimers::default());
    let errors = Arc::new(ErrorSlot::default());
    // HF's trainer pretokenizes files one line at a time, and under ByteLevel
    // a line's trailing newline is content that belongs to that line. Chunks
    // must therefore hold whole lines (see reader::CutRule).
    let cut_rule = if bytelevel {
        crate::reader::CutRule::AfterNewline
    } else {
        crate::reader::CutRule::AfterWhitespace
    };
    let batch_bytes = sizing.batch;
    if crate::rss::enabled() {
        eprintln!(
            "sizing: chunk={}KB queue={} batch={}KB readers={} scanners={} owners={}",
            sizing.chunk >> 10,
            sizing.chunk_queue,
            sizing.batch >> 10,
            sizing.readers,
            sizing.scanners,
            sizing.owners,
        );
    }

    let mut batch_senders = Vec::with_capacity(sizing.owners);
    let mut batch_receivers = Vec::with_capacity(sizing.owners);
    for _ in 0..sizing.owners {
        let (tx, rx) = sync_channel::<WordBatch>(4);
        batch_senders.push(tx);
        batch_receivers.push(rx);
    }

    let maps: Vec<WordCounter> = std::thread::scope(|s| {
        // Created inside the scope so the parent's handle can be dropped once
        // the scanners hold their clones. If it lived in the enclosing frame it
        // would keep the receiver alive even after every scanner had died, and
        // the readers below would block forever in `send` on the bounded
        // channel while the main thread blocked in `join` — a permanent hang
        // rather than a crash. See the scanner-panic note after the spawns.
        let (chunk_tx, chunk_rx) = sync_channel::<Arc<Vec<u8>>>(sizing.chunk_queue);
        let chunk_rx = Arc::new(Mutex::new(chunk_rx));

        let owners: Vec<_> = batch_receivers
            .into_iter()
            .map(|rx| {
                let timers = Arc::clone(&timers);
                s.spawn(move || {
                    let mut map = WordCounter::new();
                    let timed = crate::rss::enabled();
                    loop {
                        let t0 = timed.then(std::time::Instant::now);
                        let batch = rx.recv();
                        if let Some(t0) = t0 {
                            timers
                                .recv
                                .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        }
                        let Ok(batch) = batch else { break };
                        let t1 = timed.then(std::time::Instant::now);
                        for (word, hash) in batch.iter() {
                            map.add_hashed(word, hash, 1);
                        }
                        if let Some(t1) = t1 {
                            timers
                                .insert
                                .fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        }
                    }
                    map
                })
            })
            .collect();

        let scanners: Vec<_> = (0..sizing.scanners)
            .map(|_| {
                let chunk_rx = Arc::clone(&chunk_rx);
                let batch_senders = batch_senders.clone();
                let timers = Arc::clone(&timers);
                let errors = Arc::clone(&errors);
                s.spawn(move || {
                    let timed = crate::rss::enabled();
                    let nshards = nshards_u64;
                    let mut batches: Vec<WordBatch> =
                        (0..nshards as usize).map(|_| WordBatch::new()).collect();
                    loop {
                        let chunk = chunk_rx.lock().unwrap().recv();
                        let Ok(chunk) = chunk else { break };
                        let t_scan = timed.then(std::time::Instant::now);
                        let text = std::str::from_utf8(&chunk).unwrap_or_else(|e| {
                            errors.set(format!("input is not valid UTF-8: {e}"));
                            ""
                        });
                        // Route on the high bits: the low bits pick the
                        // hash-map bucket, so reusing them here would leave
                        // each shard's index sparsely populated.
                        let mut route = |w: &str| {
                            let h = crate::counter::hash_word(w);
                            let shard = ((h >> 32) % nshards) as usize;
                            batches[shard].push(w, h);
                            if batches[shard].bytes() >= batch_bytes {
                                let full = std::mem::take(&mut batches[shard]);
                                let _ = batch_senders[shard].send(full);
                            }
                        };
                        if bytelevel {
                            // Per line, matching HF's line-at-a-time feed: a
                            // trailing newline is terminal within its line, so
                            // "x\r\n" ends in one `čĊ` token rather than two.
                            //
                            // Pieces are counted unmapped; the byte-to-unicode
                            // map is a bijection, so equality is unaffected and
                            // the mapping is deferred to the unique words when
                            // shards are combined.
                            for line in text.split_inclusive('\n') {
                                crate::bytelevel::for_each_piece(line, &mut route);
                            }
                        } else {
                            crate::split::for_each_word(text, route);
                        }
                        if let Some(t) = t_scan {
                            timers
                                .scan
                                .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        }
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
        // The scanners now hold every remaining reference to the receiver.
        // Dropping the parent's handle is what makes a scanner *panic*
        // survivable: with all scanners gone the channel is closed, so the
        // readers' `send` returns Err and they exit their loop, letting
        // `join` below re-raise the panic. Holding this handle instead turns
        // any scanner panic into a deadlock (readers blocked in `send`, main
        // blocked in `join`) with no output and no exit code. The only
        // reachable scanner panic today is `WordBatch::push`'s 4 GiB assert,
        // hit by a single pretoken larger than u32 can address.
        drop(chunk_rx);

        // Parallel range readers: one reader saturates well below what the
        // scanner pool can consume. Ranges are split by byte offset and made
        // word-safe by reader::read_range's skip/overshoot rule.
        let tx = chunk_tx;
        let mut jobs: Vec<(String, u64, u64)> = Vec::new();
        for path in paths {
            let meta = match std::fs::metadata(path) {
                Ok(m) => m,
                Err(e) => {
                    errors.set(format!("reading {path}: {e}"));
                    break;
                }
            };
            // Ranges come from the stat size and the readers seek, so a
            // non-regular file cannot be read this way. A FIFO reports size 0,
            // which previously produced an empty vocabulary and exit 0 — so
            // `gigabpe <(zcat corpus.gz)` silently emitted nothing.
            if !meta.is_file() {
                errors.set(format!(
                    "{path} is not a regular file; pipes and process \
                     substitution are not supported because the reader seeks. \
                     Write the stream to a file first."
                ));
                break;
            }
            let len = meta.len();
            for (start, end) in
                crate::reader::split_ranges(len, sizing.readers, crate::reader::min_range_bytes())
            {
                jobs.push((path.clone(), start, end));
            }
        }

        let next_job = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let jobs = Arc::new(jobs);
        let readers: Vec<_> = (0..sizing.readers.min(jobs.len().max(1)))
            .map(|_| {
                let tx = tx.clone();
                let jobs = Arc::clone(&jobs);
                let next_job = Arc::clone(&next_job);
                let chunk = sizing.chunk;
                let rule = cut_rule;
                let errors = Arc::clone(&errors);
                s.spawn(move || loop {
                    let i = next_job.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((path, start, end)) = jobs.get(i) else {
                        break;
                    };
                    if let Err(e) = crate::reader::read_range(path, *start, *end, chunk, rule, &tx)
                    {
                        errors.set(e);
                        break;
                    }
                })
            })
            .collect();
        for r in readers {
            r.join().unwrap();
        }
        drop(tx);
        for scanner in scanners {
            scanner.join().unwrap();
        }
        owners.into_iter().map(|h| h.join().unwrap()).collect()
    });
    if crate::rss::enabled() {
        timers.report(sizing.scanners, sizing.owners, sizing.readers);
    }
    crate::rss::report("phase 1 shard counters");

    // Shards are disjoint, so combining is concatenation: no lookups, no
    // dedup. Each shard's index and arena are freed as it is consumed.
    let mut maps = maps;
    let total_words: usize = maps.iter().map(|m| m.len()).sum();
    let total_bytes: usize = maps.iter().map(|m| m.total_bytes()).sum();
    let mut table = WordTable::with_capacity(total_words, total_bytes);
    let byte_table = crate::bytelevel::byte_to_char();
    let mut mapped = String::new();
    for map in maps.drain(..) {
        let shard = map.into_table();
        for i in 0..shard.len() {
            if bytelevel {
                crate::bytelevel::map_bytes(shard.word(i), &byte_table, &mut mapped);
                table.push(&mapped, shard.count(i));
            } else {
                table.push(shard.word(i), shard.count(i));
            }
        }
    }
    crate::rss::report("combining shards");
    match errors.take() {
        Some(e) => Err(e),
        None => Ok(table),
    }
}
