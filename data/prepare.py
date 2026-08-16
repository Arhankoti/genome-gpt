"""Encode a (multi-record) FASTA genome into train.bin / val.bin (uint8) + meta.

Two correctness points:
  1. Records (genomes/contigs) are joined with the SEP boundary token, so the
     model never learns transitions across genome boundaries. Each split gets its
     own internal boundaries; the two splits never touch.
  2. Validation is WHOLE HELD-OUT GENOMES (Part 5), chosen deliberately — not a
     contiguous tail that accidentally holds out whichever record landed last.
     Held-out records go entirely into val, so val bits/bp measures cross-genome
     generalization, not recall. meta.pkl records which genomes went where.

    python data/prepare.py                              # cfg holdout settings
    python data/prepare.py --fasta path.fasta           # override input
    python data/prepare.py --holdout e_coli_k12,h_pylori_26695   # by name
    python data/prepare.py --holdout_k 3 --holdout_seed 7        # random k
    python data/prepare.py --holdout_k 0                # legacy contiguous tail
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ITOS, SEP_ID, STOI, Config


def read_records(path):
    """Yield one uppercase sequence string per FASTA record (sequence only)."""
    return [seq for _, seq in read_named_records(path)]


def read_named_records(path):
    """Return [(name, sequence)] per FASTA record.

    name is the first whitespace-delimited token of the `>` header (how a human
    refers to a genome, e.g. `e_coli_k12`). Sequence is uppercased.
    """
    recs, name, cur = [], None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    recs.append((name, "".join(cur)))
                    cur = []
                name = line[1:].strip().split()[0] if line[1:].strip() else ""
            else:
                cur.append(line.strip().upper())
    if cur:
        recs.append((name, "".join(cur)))
    return recs


def encode(seq):
    return np.fromiter((STOI.get(c, STOI["N"]) for c in seq), dtype=np.uint8, count=len(seq))


def _join_with_boundaries(seqs):
    """Concatenate encoded records, inserting a SEP boundary token between them."""
    sep = np.array([SEP_ID], dtype=np.uint8)
    parts = []
    for i, s in enumerate(seqs):
        parts.append(encode(s))
        if i < len(seqs) - 1:
            parts.append(sep)
    return np.concatenate(parts) if parts else np.array([], dtype=np.uint8)


def select_holdout(names, holdout_genomes="", holdout_k=2, holdout_seed=1337):
    """Return the set of record names to route into val.

    Named holdout wins; otherwise pick holdout_k names with a seeded RNG. The
    holdout is never empty and never all records (val must be held-out, train
    must be non-empty). Raises ValueError on an impossible request.
    """
    if len(names) < 2:
        raise ValueError("need at least 2 records to hold one out")
    if holdout_genomes:
        want = [g.strip() for g in holdout_genomes.split(",") if g.strip()]
        missing = [g for g in want if g not in names]
        if missing:
            raise ValueError(f"holdout genome(s) not found: {missing}; have {names}")
        chosen = set(want)
    else:
        if holdout_k <= 0:
            raise ValueError("holdout_k must be >= 1 for named/random holdout")
        if holdout_k >= len(names):
            raise ValueError(f"holdout_k={holdout_k} leaves no training records ({len(names)})")
        rng = np.random.default_rng(holdout_seed)
        chosen = set(rng.choice(names, size=holdout_k, replace=False).tolist())
    if not chosen or len(chosen) >= len(names):
        raise ValueError("holdout must be a non-empty proper subset of records")
    return chosen


def _split_stats(data):
    acgt = np.isin(data, [STOI["A"], STOI["C"], STOI["G"], STOI["T"]])
    gc = float(np.isin(data, [STOI["G"], STOI["C"]]).sum() / max(1, acgt.sum()))
    return int(acgt.sum()), gc


def main():
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default=cfg.fasta_path)
    ap.add_argument("--holdout", default=cfg.holdout_genomes, help="comma-separated record names")
    ap.add_argument(
        "--holdout_k", type=int, default=cfg.holdout_k, help="0 = legacy contiguous tail"
    )
    ap.add_argument("--holdout_seed", type=int, default=cfg.holdout_seed)
    args = ap.parse_args()

    named = read_named_records(args.fasta)
    names = [n for n, _ in named]
    os.makedirs(os.path.dirname(cfg.train_bin), exist_ok=True)

    # legacy contiguous-tail split (reproduces Parts 2-4) when holdout_k == 0
    if args.holdout_k == 0 and not args.holdout:
        data = _join_with_boundaries([s for _, s in named])
        split = int(len(data) * (1.0 - cfg.val_tail_frac))
        train, val = data[:split], data[split:]
        train_genomes, val_genomes, mode = names, [], "legacy_contiguous_tail"
    else:
        holdout = select_holdout(names, args.holdout, args.holdout_k, args.holdout_seed)
        train_seqs = [s for n, s in named if n not in holdout]
        val_seqs = [s for n, s in named if n in holdout]
        train = _join_with_boundaries(train_seqs)
        val = _join_with_boundaries(val_seqs)
        train_genomes = [n for n in names if n not in holdout]
        val_genomes = [n for n in names if n in holdout]
        mode = "whole_genome_holdout"

    train.tofile(cfg.train_bin)
    val.tofile(cfg.val_bin)
    with open(cfg.meta_path, "wb") as f:
        pickle.dump(
            {
                "stoi": STOI,
                "itos": ITOS,
                "train_genomes": train_genomes,
                "val_genomes": val_genomes,
                "holdout_seed": args.holdout_seed,
                "split_mode": mode,
            },
            f,
        )

    tr_bp, tr_gc = _split_stats(train)
    va_bp, va_gc = _split_stats(val)
    print(f"records: {len(named)} | split mode: {mode}")
    print(f"  train: {len(train_genomes)} genomes, {tr_bp:,} bp, GC {tr_gc:.4f}")
    print(f"  val (held-out): {val_genomes or '[tail]'}, {va_bp:,} bp, GC {va_gc:.4f}")
    print(f"wrote {cfg.train_bin}, {cfg.val_bin}, {cfg.meta_path}")


if __name__ == "__main__":
    main()
