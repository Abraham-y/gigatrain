#!/usr/bin/env python3
"""Verify the Python bindings, including that the emitted tokenizer.json is a
genuine drop-in.

Three checks:

1. `train_bpe` returns the same merges HF's BpeTrainer produces.
2. `train_tokenizer` writes a tokenizer.json that
   `tokenizers.Tokenizer.from_file()` loads, whose model matches, and which
   encodes identically to a tokenizer HF trained itself.
3. Both pretokenizers behave.

Usage: python3 scripts/check_bindings.py CORPUS.txt
"""
import json
import sys
import tempfile
from pathlib import Path

import gigatrain
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def hf_train(files, vocab_size, pretokenizer, specials):
    tok = Tokenizer(models.BPE())
    if pretokenizer == "bytelevel":
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,
                                                     use_regex=True)
    else:
        tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.train(files, trainers.BpeTrainer(vocab_size=vocab_size,
                                         special_tokens=specials,
                                         show_progress=False))
    return tok


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else None
    if not corpus:
        print("usage: check_bindings.py CORPUS.txt")
        sys.exit(2)

    failures = 0
    print(f"gigatrain {gigatrain.__version__}")

    for pretok in ("whitespace", "bytelevel"):
        vocab_size = 3000
        specials = ["<|endoftext|>"]
        print(f"\n=== {pretok} ===")

        hf = hf_train([corpus], vocab_size, pretok, specials)
        hf_model = json.loads(hf.to_str())["model"]
        hf_merges = [tuple(m) for m in hf_model["merges"]]

        # 1. train_bpe returns matching merges and vocab
        vocab, merges = gigatrain.train_bpe(
            [corpus], vocab_size, special_tokens=specials, pretokenizer=pretok
        )
        if merges != hf_merges:
            print(f"FAIL train_bpe merges differ ({len(merges)} vs {len(hf_merges)})")
            for i, (a, b) in enumerate(zip(merges, hf_merges)):
                if a != b:
                    print(f"  first diff at {i}: ours={a} hf={b}")
                    break
            failures += 1
        else:
            print(f"  train_bpe: {len(merges)} merges identical")

        if vocab != hf_model["vocab"]:
            print(f"FAIL vocab differs ({len(vocab)} vs {len(hf_model['vocab'])})")
            failures += 1
        else:
            print(f"  train_bpe: vocab of {len(vocab)} identical")

        # 2. train_tokenizer writes a loadable, matching tokenizer.json
        with tempfile.TemporaryDirectory() as d:
            out = str(Path(d) / "tokenizer.json")
            gigatrain.train_tokenizer(
                [corpus], vocab_size, out,
                special_tokens=specials, pretokenizer=pretok,
            )
            loaded = Tokenizer.from_file(out)
            loaded_model = json.loads(loaded.to_str())["model"]

            if [tuple(m) for m in loaded_model["merges"]] != hf_merges:
                print("FAIL round-tripped merges differ")
                failures += 1
            elif loaded_model["vocab"] != hf_model["vocab"]:
                print("FAIL round-tripped vocab differs")
                failures += 1
            else:
                print("  tokenizer.json: loads in tokenizers, model matches")

            # 3. and it encodes the same as HF's own tokenizer
            samples = [
                "Hello world, this is a test.",
                "don't stop  believing\n",
                "中文 text 123 mixed",
                "   leading and trailing   ",
            ]
            bad = 0
            for s in samples:
                if loaded.encode(s).ids != hf.encode(s).ids:
                    bad += 1
                    if bad == 1:
                        print(f"FAIL encoding differs on {s!r}")
                        print(f"  ours={loaded.encode(s).ids}")
                        print(f"  hf  ={hf.encode(s).ids}")
            if bad:
                failures += bad
            else:
                print(f"  encoding: identical ids on {len(samples)} samples")

    if failures:
        print(f"\nBINDINGS CHECK FAILED: {failures} problems")
        sys.exit(1)
    print("\nBINDINGS OK")


if __name__ == "__main__":
    main()
