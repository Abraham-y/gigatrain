//! Debug helper: read stdin, print one pretoken per line.
//!
//! Exists so scripts/check_bytelevel_parity.py can diff our pretokenization
//! against HuggingFace's directly.
//!
//! With --lines, each input line is treated as an independent document and
//! one output line is produced per input line, tokens separated by tabs. This
//! lets a differential test check tens of thousands of cases in one process.
//!
//! Usage: pretok [--bytelevel | --whitespace] [--lines] < input

use std::io::{Read, Write};

fn main() {
    let bytelevel = std::env::args().any(|a| a == "--bytelevel");
    let lines = std::env::args().any(|a| a == "--lines");
    let mut text = String::new();
    std::io::stdin()
        .read_to_string(&mut text)
        .expect("stdin is not UTF-8");

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    if lines {
        let table = gigabpe::bytelevel::byte_to_char();
        let mut buf = String::new();
        for line in text.split('\n') {
            let mut first = true;
            let emit = |t: &str| {
                if !first {
                    write!(out, "\t").unwrap();
                }
                first = false;
                write!(out, "{t}").unwrap();
            };
            if bytelevel {
                gigabpe::bytelevel::for_each_token(line, &table, &mut buf, emit);
            } else {
                gigabpe::split::for_each_word(line, emit);
            }
            writeln!(out).unwrap();
        }
        out.flush().unwrap();
        return;
    }

    if bytelevel {
        let table = gigabpe::bytelevel::byte_to_char();
        let mut buf = String::new();
        gigabpe::bytelevel::for_each_token(&text, &table, &mut buf, |t| {
            writeln!(out, "{t}").unwrap();
        });
    } else {
        gigabpe::split::for_each_word(&text, |w| {
            writeln!(out, "{w}").unwrap();
        });
    }
    out.flush().unwrap();
}
