"""Real-data benchmark: does the neural net beat the best Markov k-gram on whole
held-out genomes it never trained on — overall AND per genome?

This is the honest headline of Part 5. It reuses the existing honest tools:
Markov baseline (evaluate.markov_bits) and k-mer / GC stats (generation.py). A
per-genome report keeps one weird organism from hiding inside a flattering
average.

    python benchmark.py --ckpt checkpoints/real.pt
    python benchmark.py --ckpt checkpoints/synth.pt --out benchmark.json --fig benchmark.png

Reads the whole-genome split written by data/prepare.py (meta.pkl records which
genomes are train vs val). A negative gap_vs_best_markov means the neural net
wins.
"""

import argparse
import pickle

import numpy as np

from config import ITOS, SEP_ID, Config
from evaluate import markov_bits
from generation import gc_content, js_divergence, kmer_spectrum
from inference import GenomeModel

_NOTE = (
    "gap < 0 means the neural net beats the best k-gram; measured on whole "
    "held-out genomes the model never trained on."
)


def _decode(ids):
    """uint8 id array -> DNA string (used for neural scoring and k-mer stats)."""
    return "".join(ITOS[int(i)] for i in ids)


def split_by_boundary(ids):
    """Split a token array on the SEP boundary into per-record segments (drops
    empty segments). The whole-genome val split joins held-out records with SEP,
    so this recovers each genome's own tokens in file order."""
    segs, cur = [], []
    for t in ids:
        if int(t) == SEP_ID:
            if cur:
                segs.append(np.array(cur, dtype=np.uint8))
                cur = []
        else:
            cur.append(int(t))
    if cur:
        segs.append(np.array(cur, dtype=np.uint8))
    return segs


def _score_segment(gm, train_ids, seg_ids, markov_orders):
    """Neural + Markov bits/bp for one held-out genome segment."""
    seq = _decode(seg_ids)
    neural = gm.score(seq)["bits_per_bp"]
    markov = {k: markov_bits(train_ids, seg_ids, k) for k in markov_orders}
    best_k = min(markov, key=markov.get)
    return {
        "neural_bits_per_bp": neural,
        "markov": {int(k): float(v) for k, v in markov.items()},
        "best_markov_k": int(best_k),
        "best_markov_bits_per_bp": float(markov[best_k]),
        "gap_vs_best_markov": float(neural - markov[best_k]),
        "gc": gc_content(seq),
        "n_tokens": int(len(seg_ids)),
    }


def benchmark(
    ckpt=None,
    *,
    train_bin=None,
    val_bin=None,
    meta_path=None,
    max_eval_tokens=500_000,
    markov_orders=(0, 2, 4, 6, 8),
):
    """Neural vs Markov on the held-out genomes, overall and per genome.

    Returns a JSON-serializable dict; see module docstring / README for the
    contract. `max_eval_tokens` caps the overall neural pass on huge corpora;
    per-genome segments are scored in full (bacterial genomes fit comfortably).
    The bin/meta paths default to Config's; overrides exist for tests.
    """
    cfg = Config()
    train_ids = np.fromfile(train_bin or cfg.train_bin, dtype=np.uint8)
    val_ids = np.fromfile(val_bin or cfg.val_bin, dtype=np.uint8)
    with open(meta_path or cfg.meta_path, "rb") as f:
        meta = pickle.load(f)
    val_genomes = meta.get("val_genomes", [])
    train_genomes = meta.get("train_genomes", [])

    gm = GenomeModel(ckpt)

    # overall (capped) — the whole held-out set at a glance
    val_capped = val_ids[:max_eval_tokens]
    val_str = _decode(val_capped[val_capped != SEP_ID])
    train_str = _decode(train_ids[:max_eval_tokens][train_ids[:max_eval_tokens] != SEP_ID])
    overall_markov = {k: markov_bits(train_ids, val_capped, k) for k in markov_orders}
    best_k = min(overall_markov, key=overall_markov.get)
    overall_neural = gm.score(val_str)["bits_per_bp"]
    overall = {
        "neural_bits_per_bp": overall_neural,
        "markov": {int(k): float(v) for k, v in overall_markov.items()},
        "best_markov_k": int(best_k),
        "gap_vs_best_markov": float(overall_neural - overall_markov[best_k]),
        "gc": gc_content(val_str),
        "kmer_js_bits_vs_train": js_divergence(kmer_spectrum(val_str), kmer_spectrum(train_str)),
    }

    # per held-out genome — split val on the boundary token, map to names in order
    segments = split_by_boundary(val_ids)
    per_genome = []
    if val_genomes and len(segments) == len(val_genomes):
        for name, seg in zip(val_genomes, segments):
            row = _score_segment(gm, train_ids, seg, markov_orders)
            per_genome.append({"name": name, **row})

    return {
        "overall": overall,
        "per_genome": per_genome,
        "val_genomes": val_genomes,
        "train_genomes": train_genomes,
        "markov_orders": list(markov_orders),
        "note": _NOTE,
    }


def _print_report(report):
    o = report["overall"]
    print(f"\n  held-out genomes: {report['val_genomes'] or '[legacy tail]'}")
    print(
        f"  overall: neural {o['neural_bits_per_bp']:.4f} | "
        f"best markov k={o['best_markov_k']} {o['markov'][o['best_markov_k']]:.4f} | "
        f"gap {o['gap_vs_best_markov']:+.4f} | GC {o['gc']:.4f}\n"
    )
    if report["per_genome"]:
        hdr = f"  {'genome':<24} {'n_tok':>9} {'neural':>8} {'bestMk':>8} {'k':>2} {'gap':>8} {'GC':>6}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in report["per_genome"]:
            print(
                f"  {r['name']:<24} {r['n_tokens']:>9,} {r['neural_bits_per_bp']:>8.4f} "
                f"{r['best_markov_bits_per_bp']:>8.4f} {r['best_markov_k']:>2} "
                f"{r['gap_vs_best_markov']:>+8.4f} {r['gc']:>6.3f}"
            )
    print(f"\n  {report['note']}")


def main():
    cfg = Config()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", default=cfg.ckpt_path)
    ap.add_argument("--max_eval_tokens", type=int, default=500_000)
    ap.add_argument("--out", default=None, help="Write the report JSON here")
    ap.add_argument("--fig", default=None, help="Write the benchmark PNG here")
    args = ap.parse_args()

    report = benchmark(args.ckpt, max_eval_tokens=args.max_eval_tokens)
    _print_report(report)

    if args.out:
        import json

        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  wrote {args.out}")
    if args.fig:
        from viz import render_benchmark

        render_benchmark(report, out_path=args.fig)
        print(f"  wrote {args.fig}")


if __name__ == "__main__":
    main()
