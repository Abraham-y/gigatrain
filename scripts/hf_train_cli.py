#!/usr/bin/env python3
"""Train HF tokenizers' BpeTrainer on text files; print serialized merges to
stdout (one "left right" per line) and timing to stderr. Subprocess target
for benchmark.py so peak RSS can be measured with /usr/bin/time."""
import argparse
import json
import sys
import time

from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--min-frequency", type=int, default=0)
    p.add_argument("--special", action="append", default=[])
    p.add_argument("--max-token-length", type=int, default=None)
    p.add_argument("--pretokenizer", choices=["whitespace", "bytelevel"],
                   default="whitespace")
    p.add_argument("files", nargs="+")
    args = p.parse_args()

    tok = Tokenizer(models.BPE())
    if args.pretokenizer == "bytelevel":
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,
                                                    use_regex=True)
    else:
        tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    kwargs = dict(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=args.special,
        show_progress=False,
    )
    if args.max_token_length is not None:
        kwargs["max_token_length"] = args.max_token_length
    trainer = trainers.BpeTrainer(**kwargs)

    t0 = time.perf_counter()
    tok.train(args.files, trainer)
    elapsed = time.perf_counter() - t0

    model = json.loads(tok.to_str())["model"]
    out = sys.stdout
    for a, b in model["merges"]:
        out.write(f"{a} {b}\n")
    print(f"train_seconds: {elapsed:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
