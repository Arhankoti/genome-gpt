"""Encode a (multi-record) FASTA genome into train.bin / val.bin (uint8) + meta.

Two correctness points:
  1. Records (genomes/contigs) are joined with the SEP boundary token, so the
     model never learns transitions across genome boundaries.
  2. Validation is a CONTIGUOUS TAIL of the concatenated stream. With a
     multi-genome corpus that tail is whole held-out genome(s) the model never
     trains on, so val bits/bp measures cross-genome generalization, not recall.

    python data/prepare.py                       # uses cfg.fasta_path
    python data/prepare.py --fasta path.fasta    # override
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ITOS, SEP_ID, STOI, Config


def read_records(path):
    """Yield one uppercase sequence string per FASTA record."""
    recs, cur = [], []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    recs.append("".join(cur))
                    cur = []
            else:
                cur.append(line.strip().upper())
    if cur:
        recs.append("".join(cur))
    return recs


def encode(seq):
    return np.fromiter((STOI.get(c, STOI["N"]) for c in seq), dtype=np.uint8, count=len(seq))


def main():
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default=cfg.fasta_path)
    args = ap.parse_args()

    recs = read_records(args.fasta)
    sep = np.array([SEP_ID], dtype=np.uint8)
    parts = []
    for i, r in enumerate(recs):
        parts.append(encode(r))
        if i < len(recs) - 1:
            parts.append(sep)  # boundary between genomes
    data = np.concatenate(parts) if parts else np.array([], dtype=np.uint8)

    n = len(data)
    split = int(n * (1.0 - cfg.val_tail_frac))
    train, val = data[:split], data[split:]

    os.makedirs(os.path.dirname(cfg.train_bin), exist_ok=True)
    train.tofile(cfg.train_bin)
    val.tofile(cfg.val_bin)
    with open(cfg.meta_path, "wb") as f:
        pickle.dump({"stoi": STOI, "itos": ITOS}, f)

    acgt = np.isin(data, [STOI["A"], STOI["C"], STOI["G"], STOI["T"]])
    gc = float(np.isin(data, [STOI["G"], STOI["C"]]).sum() / max(1, acgt.sum()))
    print(
        f"records: {len(recs)} | total {n:,} tokens "
        f"({int(acgt.sum()):,} bp + {len(recs) - 1} boundaries)"
    )
    print(f"train {len(train):,} | val {len(val):,} (contiguous tail = held-out genomes)")
    print(f"GC content: {gc:.4f}")
    print(f"wrote {cfg.train_bin}, {cfg.val_bin}, {cfg.meta_path}")


if __name__ == "__main__":
    main()
