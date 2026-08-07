#!/usr/bin/env python3
"""Train a YouTokenToMe BPE model, for benchmark comparison.

YouTokenToMe is archived (2024) and publishes no wheels past cp38, but the
sdist still builds given Cython — see the Modal image in modal_benchmark.py.
It is included because its `benchmark.md` is the most-cited trainer benchmark
in this area and PRIOR_ART.md audits it; comparing against a tool while never
running it would be exactly the sort of thing that audit criticises.

This is a speed/memory comparison only. YouTokenToMe produces its own
vocabulary format and applies its own pretokenization, so there is no
merge-for-merge diff against HuggingFace.

Usage:
  python scripts/yttm_train_cli.py --vocab-size 32000 --model-prefix /tmp/y corpus.txt
"""
import argparse
import sys
import time

import youtokentome as yttm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--model-prefix", required=True)
    p.add_argument("--threads", type=int, default=-1,
                   help="-1 lets YouTokenToMe choose (it caps training at 8)")
    p.add_argument("--coverage", type=float, default=1.0)
    p.add_argument("input")
    args = p.parse_args()

    t0 = time.perf_counter()
    yttm.BPE.train(
        data=args.input,
        model=args.model_prefix + ".model",
        vocab_size=args.vocab_size,
        coverage=args.coverage,
        n_threads=args.threads,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )
    print(f"train_seconds: {time.perf_counter() - t0:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
