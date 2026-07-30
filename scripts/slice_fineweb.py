#!/usr/bin/env python3
"""Slice FineWeb parquet file(s) into plain-text corpora of target sizes.

Streams record batches so the parquet never fully materializes in RAM.
Documents are separated by blank lines (irrelevant to whitespace
pretokenization, readable for humans).

Usage:
  python scripts/slice_fineweb.py data/fineweb_000.parquet --sizes-mb 100 1000
"""
import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("parquets", nargs="+")
    p.add_argument("--sizes-mb", type=int, nargs="+", required=True)
    p.add_argument("--out-dir", default="data")
    args = p.parse_args()

    targets = sorted(args.sizes_mb)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    outs = []  # (target_bytes, path, file_handle, done)
    for mb in targets:
        path = out_dir / f"fineweb_{mb}mb.txt"
        outs.append([mb * 1_000_000, path, open(path, "w"), False])

    written = 0
    for parquet in args.parquets:
        pf = pq.ParquetFile(parquet)
        for batch in pf.iter_batches(columns=["text"], batch_size=1024):
            for text in batch.column("text").to_pylist():
                doc = text + "\n\n"
                written += len(doc.encode("utf-8", "ignore"))
                for out in outs:
                    if not out[3]:
                        out[2].write(doc)
                        if written >= out[0]:
                            out[3] = True
                            out[2].close()
                            print(f"wrote {out[1]} ({out[0] / 1e6:.0f} MB)", file=sys.stderr)
            if all(o[3] for o in outs):
                return
    for out in outs:
        if not out[3]:
            out[2].close()
            print(f"WARNING: {out[1]} only reached {written / 1e6:.0f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
