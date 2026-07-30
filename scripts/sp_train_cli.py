#!/usr/bin/env python3
"""Train SentencePiece BPE on a text file, for speed/memory comparison only.

SentencePiece BPE does NOT produce a HuggingFace-compatible merge list: it
normalizes (NFKC by default), builds its alphabet from character coverage
rather than every observed char, and emits pieces rather than merge pairs.
So this is not a parity comparison — only wall time and peak RSS are
comparable, and even then only loosely.

Two flags matter for a fair large-corpus run:
  --max_sentence_length  defaults to 4192 BYTES and silently drops longer
                         lines, which would quietly shrink the corpus.
  --train_extremely_large_corpus  required past a few million sentences.
"""
import argparse
import sys
import time

import sentencepiece as spm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--model-prefix", required=True)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--max-sentence-length", type=int, default=1 << 20)
    p.add_argument("input")
    args = p.parse_args()

    kwargs = dict(
        input=args.input,
        model_prefix=args.model_prefix,
        model_type="bpe",
        vocab_size=args.vocab_size,
        max_sentence_length=args.max_sentence_length,
        input_sentence_size=0,
        train_extremely_large_corpus=True,
        normalization_rule_name="identity",
        character_coverage=1.0,
    )
    if args.threads:
        kwargs["num_threads"] = args.threads

    t0 = time.perf_counter()
    spm.SentencePieceTrainer.train(**kwargs)
    print(f"train_seconds: {time.perf_counter() - t0:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
