//! Parallel range reader for phase 1.
//!
//! A single reader thread tops out around 700 MB/s, which becomes the phase-1
//! bottleneck once enough scanner threads are consuming. Input is therefore
//! split into byte ranges read concurrently.
//!
//! Splitting a file at arbitrary byte offsets would cut words in half, so
//! ranges use the standard input-split rule:
//!
//! - A range emits exactly the words whose **first byte** lies in
//!   `[start, end)`.
//! - A range starting mid-file skips forward past the first whitespace: that
//!   partial word began before `start` and belongs to the previous range.
//! - A range reads *past* `end` until the first whitespace at or after `end`,
//!   to finish the word that started inside it.
//!
//! Every byte is therefore counted exactly once, whatever the range layout —
//! including words longer than a whole range.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::mpsc::SyncSender;
use std::sync::Arc;

/// ASCII whitespace is always a real word boundary: UTF-8 continuation bytes
/// are never ASCII, and every ASCII whitespace char satisfies
/// `char::is_whitespace`.
#[inline]
fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

/// Split `total` bytes into at most `n` ranges of at least `min_range` bytes.
pub fn split_ranges(total: u64, n: usize, min_range: u64) -> Vec<(u64, u64)> {
    if total == 0 {
        return vec![];
    }
    let n = n.max(1) as u64;
    let count = (total / min_range.max(1)).clamp(1, n);
    let step = total / count;
    (0..count)
        .map(|i| {
            let start = i * step;
            let end = if i + 1 == count {
                total
            } else {
                (i + 1) * step
            };
            (start, end)
        })
        .collect()
}

/// Where a chunk may be cut so that pretokenization is unaffected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CutRule {
    /// Whitespace is a discarded delimiter, so a chunk may end just *after*
    /// any whitespace byte.
    AfterWhitespace,
    /// Whitespace is content (ByteLevel maps it to `Ġ`, `Ċ`, ...) and HF's
    /// trainer pretokenizes a file one line at a time, so a line's trailing
    /// newline is terminal within that line. Chunks must therefore hold whole
    /// lines: cut immediately after a `\n`.
    AfterNewline,
}

/// Is `i` a legal cut point in `buf` under `rule`? Both rules cut just after
/// the matched byte, so a chunk always ends on a boundary the pretokenizer
/// would have produced anyway.
#[inline]
fn is_cut_point(b: u8, rule: CutRule) -> bool {
    match rule {
        CutRule::AfterWhitespace => is_ws(b),
        CutRule::AfterNewline => b == b'\n',
    }
}

/// Read `[start, end)` of `path`, sending chunks cut per `rule`.
///
/// Returns Err with a message on I/O failure.
pub fn read_range(
    path: &str,
    start: u64,
    end: u64,
    chunk: usize,
    rule: CutRule,
    tx: &SyncSender<Arc<Vec<u8>>>,
) -> Result<(), String> {
    // `eof` is `n < chunk`, so a zero chunk would never terminate the loop.
    assert!(chunk > 0, "read_range requires a non-zero chunk size");
    let mut file = File::open(path).map_err(|e| format!("opening {path}: {e}"))?;

    // A range starting mid-file usually begins inside a word, which belongs to
    // the previous range and must be skipped. But if the byte just before
    // `start` is whitespace, a word begins exactly at `start` and is ours —
    // skipping then would lose it entirely. So look at that byte first.
    let mut skipping = false;
    if start > 0 {
        file.seek(SeekFrom::Start(start - 1))
            .map_err(|e| format!("seeking {path}: {e}"))?;
        let mut prev = [0u8; 1];
        match file.read(&mut prev) {
            // A range opens cleanly when the byte before it was itself a cut
            // point; otherwise we are mid-word (or mid-line) and that piece
            // belongs to the previous range.
            Ok(1) => skipping = !is_cut_point(prev[0], rule),
            Ok(_) => return Ok(()), // start is at/past EOF
            Err(e) => return Err(format!("reading {path}: {e}")),
        }
    } else {
        file.seek(SeekFrom::Start(0))
            .map_err(|e| format!("seeking {path}: {e}"))?;
    }

    let mut buf: Vec<u8> = Vec::new();
    // Absolute offset of buf[0]. Outside the skip phase, buf[0] is always the
    // first byte of a word.
    let mut buf_start = start;

    loop {
        // Every word from here on starts at or past `end`, so it belongs to
        // the next range.
        if !skipping && buf_start >= end {
            return Ok(());
        }
        let before = buf.len();
        buf.reserve(chunk);
        // read_to_end appends without zero-filling; resizing and reading into
        // the result would memset the entire corpus on the reader threads.
        let n = Read::by_ref(&mut file)
            .take(chunk as u64)
            .read_to_end(&mut buf)
            .map_err(|e| format!("reading {path}: {e}"))?;
        let eof = n < chunk;

        if skipping {
            match buf.iter().position(|&b| is_cut_point(b, rule)) {
                Some(p) => {
                    buf.drain(..=p);
                    buf_start += (p + 1) as u64;
                    skipping = false;
                    // The word after the skipped whitespace starts at or past
                    // `end`, so it belongs to the next range, not this one.
                    if buf_start >= end {
                        return Ok(());
                    }
                }
                None => {
                    // Still inside one very long word; discard and continue.
                    buf_start += (before + n) as u64;
                    buf.clear();
                    if eof {
                        return Ok(());
                    }
                    // No boundary anywhere in this range, so every token
                    // overlapping it began before `start` and belongs to an
                    // earlier range: this one owns nothing. Without this,
                    // each range scans to EOF hunting for a boundary, so an
                    // N-reader run reads a boundary-free file N times and
                    // gets slower as threads are added.
                    if buf_start >= end {
                        return Ok(());
                    }
                    continue;
                }
            }
        }

        let buf_end = buf_start + buf.len() as u64;
        if buf_end > end {
            // We hold bytes past `end`: finish the word that started before
            // `end` and stop, so the next range does not double-count it.
            //
            // Cut at the first boundary at or after `end - 1`, so the token
            // that begins at `end` belongs to the next range rather than being
            // counted by both. Both cut rules consume the matched byte, so the
            // next token starts at `i + 1`; requiring `i + 1 >= end` gives the
            // `end - 1` threshold.
            let jmin = (end - 1).saturating_sub(buf_start) as usize;
            match buf[jmin..].iter().position(|&b| is_cut_point(b, rule)) {
                Some(rel) => {
                    let cut = jmin + rel;
                    buf.truncate(cut + 1);
                    if !buf.is_empty() {
                        tx.send(Arc::new(buf)).map_err(|_| "scanners hung up")?;
                    }
                    return Ok(());
                }
                None if eof => {
                    if !buf.is_empty() {
                        tx.send(Arc::new(buf)).map_err(|_| "scanners hung up")?;
                    }
                    return Ok(());
                }
                None => continue, // word still unterminated; read more
            }
        }

        if eof {
            if !buf.is_empty() {
                tx.send(Arc::new(buf)).map_err(|_| "scanners hung up")?;
            }
            return Ok(());
        }

        // Normal case: cut at the last whitespace, carry the tail forward.
        match buf.iter().rposition(|&b| is_cut_point(b, rule)) {
            Some(cut) => {
                let tail = buf.split_off(cut + 1);
                buf_start += buf.len() as u64;
                tx.send(Arc::new(buf)).map_err(|_| "scanners hung up")?;
                buf = tail;
            }
            None => {
                // No boundary in the whole buffer (a pathological token, or a
                // line longer than the chunk): keep accumulating.
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::Write;
    use std::sync::mpsc::sync_channel;

    fn count_via_ranges(
        data: &[u8],
        nranges: usize,
        chunk: usize,
        rule: CutRule,
    ) -> HashMap<String, u64> {
        let dir = std::env::temp_dir().join(format!(
            "gigatrain_reader_test_{}_{}_{}_{:?}",
            data.len(),
            nranges,
            chunk,
            rule
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("input.txt");
        File::create(&path).unwrap().write_all(data).unwrap();
        let path_str = path.to_str().unwrap().to_string();

        let ranges = split_ranges(data.len() as u64, nranges, 1);
        let (tx, rx) = sync_channel::<Arc<Vec<u8>>>(64);
        std::thread::scope(|s| {
            let collector = s.spawn(move || {
                let mut counts: HashMap<String, u64> = HashMap::new();
                let table = crate::bytelevel::byte_to_char();
                let mut buf = String::new();
                while let Ok(c) = rx.recv() {
                    let text = std::str::from_utf8(&c).unwrap();
                    match rule {
                        CutRule::AfterWhitespace => {
                            for w in text.split_whitespace() {
                                *counts.entry(w.to_string()).or_default() += 1;
                            }
                        }
                        CutRule::AfterNewline => {
                            for line in text.split_inclusive('\n') {
                                crate::bytelevel::for_each_token(line, &table, &mut buf, |t| {
                                    *counts.entry(t.to_string()).or_default() += 1;
                                });
                            }
                        }
                    }
                }
                counts
            });
            for (start, end) in ranges {
                read_range(&path_str, start, end, chunk, rule, &tx).unwrap();
            }
            drop(tx);
            collector.join().unwrap()
        })
    }

    fn expected(data: &[u8], rule: CutRule) -> HashMap<String, u64> {
        let mut counts: HashMap<String, u64> = HashMap::new();
        let text = std::str::from_utf8(data).unwrap();
        match rule {
            CutRule::AfterWhitespace => {
                for w in text.split_whitespace() {
                    *counts.entry(w.to_string()).or_default() += 1;
                }
            }
            CutRule::AfterNewline => {
                let table = crate::bytelevel::byte_to_char();
                let mut buf = String::new();
                for line in text.split_inclusive('\n') {
                    crate::bytelevel::for_each_token(line, &table, &mut buf, |t| {
                        *counts.entry(t.to_string()).or_default() += 1;
                    });
                }
            }
        }
        counts
    }

    // The invariant that matters: word counts are identical no matter how the
    // input is cut into ranges or chunks.
    #[test]
    fn range_splits_never_change_counts() {
        let corpora: Vec<Vec<u8>> = vec![
            b"".to_vec(),
            b"single".to_vec(),
            b"a b c d e f g".to_vec(),
            b"   leading and trailing   ".to_vec(),
            b"repeated repeated repeated words words here".to_vec(),
            b"tabs\tand\nnewlines\r\nmixed together now".to_vec(),
            "unicode \u{00a0}nbsp \u{3000}ideo caf\u{e9} \u{4e2d}\u{6587}"
                .as_bytes()
                .to_vec(),
            // A word longer than several ranges/chunks.
            {
                let mut v = b"short ".to_vec();
                v.extend(std::iter::repeat_n(b'x', 500));
                v.extend(b" tail");
                v
            },
            // Many short words, exercising boundaries densely.
            (0..300)
                .flat_map(|i| format!("w{} ", i % 37).into_bytes())
                .collect(),
        ];

        for rule in [CutRule::AfterWhitespace, CutRule::AfterNewline] {
            for data in &corpora {
                let want = expected(data, rule);
                for nranges in [1, 2, 3, 5, 8, 17] {
                    for chunk in [1, 2, 3, 7, 16, 64, 4096] {
                        let got = count_via_ranges(data, nranges, chunk, rule);
                        assert_eq!(
                            got,
                            want,
                            "mismatch: len={} nranges={nranges} chunk={chunk} rule={rule:?}",
                            data.len()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn split_ranges_covers_everything_without_overlap() {
        for total in [0u64, 1, 7, 100, 1_000_000] {
            for n in [1usize, 2, 3, 9] {
                let ranges = split_ranges(total, n, 1);
                if total == 0 {
                    assert!(ranges.is_empty());
                    continue;
                }
                assert_eq!(ranges[0].0, 0);
                assert_eq!(ranges.last().unwrap().1, total);
                for w in ranges.windows(2) {
                    assert_eq!(w[0].1, w[1].0, "gap or overlap in {ranges:?}");
                }
            }
        }
    }
}
