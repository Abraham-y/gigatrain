//! Python bindings.
//!
//! The point of these is to be a drop-in: `train_tokenizer` writes a
//! `tokenizer.json` that `tokenizers.Tokenizer.from_file()` loads directly,
//! so an existing HuggingFace pipeline can swap the trainer without changing
//! anything downstream.
//!
//! Built behind the `python` feature so the core crate stays dependency-free.

use crate::trainer::{train, TrainerConfig};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn resolve_threads(threads: Option<usize>) -> PyResult<usize> {
    // Same bound the CLI enforces: reject rather than silently clamp, so the
    // two frontends behave identically.
    if let Some(t) = threads {
        if t > crate::pipeline::MAX_WORKERS {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "threads {t} exceeds the maximum supported ({})",
                crate::pipeline::MAX_WORKERS
            )));
        }
    }
    Ok(threads.filter(|&t| t > 0).unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }))
}

#[allow(clippy::too_many_arguments)]
fn build_config(
    vocab_size: usize,
    special_tokens: Vec<String>,
    min_frequency: u64,
    max_token_length: Option<usize>,
    limit_alphabet: Option<usize>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
) -> TrainerConfig {
    TrainerConfig {
        vocab_size,
        min_frequency,
        special_tokens,
        limit_alphabet,
        initial_alphabet: vec![],
        max_token_length,
        continuing_subword_prefix,
        end_of_word_suffix,
    }
}

fn parse_pretokenizer(name: &str) -> PyResult<bool> {
    match name {
        "bytelevel" => Ok(true),
        "whitespace" => Ok(false),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown pretokenizer {other:?} (want \"whitespace\" or \"bytelevel\")"
        ))),
    }
}

/// What `train_bpe` hands back to Python: the vocabulary as a `{token: id}`
/// dict, and the ordered merge list as `(left, right)` pairs.
type VocabAndMerges = (Py<PyDict>, Vec<(String, String)>);

/// Train on `files`, returning `(vocab, merges)`.
///
/// `vocab` maps token string to id; `merges` is the ordered list of
/// `(left, right)` pairs, exactly as HuggingFace serializes them.
#[pyfunction]
#[pyo3(signature = (
    files,
    vocab_size,
    *,
    special_tokens = vec![],
    min_frequency = 0,
    max_token_length = None,
    limit_alphabet = None,
    pretokenizer = "whitespace".to_string(),
    threads = None,
    continuing_subword_prefix = None,
    end_of_word_suffix = None,
))]
#[allow(clippy::too_many_arguments)]
fn train_bpe(
    py: Python<'_>,
    files: Vec<String>,
    vocab_size: usize,
    special_tokens: Vec<String>,
    min_frequency: u64,
    max_token_length: Option<usize>,
    limit_alphabet: Option<usize>,
    pretokenizer: String,
    threads: Option<usize>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
) -> PyResult<VocabAndMerges> {
    let bytelevel = parse_pretokenizer(&pretokenizer)?;
    let config = build_config(
        vocab_size,
        special_tokens,
        min_frequency,
        max_token_length,
        limit_alphabet,
        continuing_subword_prefix.clone(),
        end_of_word_suffix.clone(),
    );
    let nthreads = resolve_threads(threads)?;

    // Training is pure Rust and can take minutes; release the GIL so other
    // Python threads keep running.
    let result = py.detach(move || {
        crate::pipeline::count_words(&files, nthreads, bytelevel).map(|table| train(table, &config))
    });
    let result = result.map_err(pyo3::exceptions::PyValueError::new_err)?;

    let vocab = PyDict::new(py);
    for (id, token) in result.vocab.iter().enumerate() {
        vocab.set_item(token, id as u32)?;
    }
    Ok((vocab.into(), result.serialized_merges()))
}

/// Train and write a HuggingFace `tokenizer.json` to `output`.
///
/// The result is loadable with `tokenizers.Tokenizer.from_file(output)`.
#[pyfunction]
#[pyo3(signature = (
    files,
    vocab_size,
    output,
    *,
    special_tokens = vec![],
    min_frequency = 0,
    max_token_length = None,
    limit_alphabet = None,
    pretokenizer = "whitespace".to_string(),
    threads = None,
    continuing_subword_prefix = None,
    end_of_word_suffix = None,
))]
#[allow(clippy::too_many_arguments)]
fn train_tokenizer(
    py: Python<'_>,
    files: Vec<String>,
    vocab_size: usize,
    output: String,
    special_tokens: Vec<String>,
    min_frequency: u64,
    max_token_length: Option<usize>,
    limit_alphabet: Option<usize>,
    pretokenizer: String,
    threads: Option<usize>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
) -> PyResult<()> {
    let bytelevel = parse_pretokenizer(&pretokenizer)?;
    let specials = special_tokens.clone();
    let prefix_for_json = continuing_subword_prefix.clone();
    let suffix_for_json = end_of_word_suffix.clone();
    let config = build_config(
        vocab_size,
        special_tokens,
        min_frequency,
        max_token_length,
        limit_alphabet,
        continuing_subword_prefix.clone(),
        end_of_word_suffix.clone(),
    );
    let nthreads = resolve_threads(threads)?;

    let json = py.detach(move || {
        let table = crate::pipeline::count_words(&files, nthreads, bytelevel)?;
        let result = train(table, &config);
        Ok::<String, String>(crate::tokenizer_json::render(
            &result,
            &specials,
            bytelevel,
            prefix_for_json.as_deref(),
            suffix_for_json.as_deref(),
        ))
    });
    let json = json.map_err(pyo3::exceptions::PyValueError::new_err)?;

    std::fs::write(&output, json)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("writing {output}: {e}")))
}

#[pymodule]
fn gigatrain(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    m.add_function(wrap_pyfunction!(train_tokenizer, m)?)?;
    Ok(())
}
