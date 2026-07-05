"""Evaluate a trained checkpoint — and answer the only question that matters
first: does the neural model actually beat a Markov (k-gram) baseline, or did
it just memorize? If the Transformer only ties a 5-mer chain, that is the real
finding.

    python evaluate.py
    python evaluate.py --ckpt checkpoints/synth.pt --max_eval_tokens 50000

Metrics are in bits/token (cross-entropy / ln2). Random over 4 bases = 2.0;
over the full 6-token vocab = 2.585. Lower is better.
"""
import os
import sys
import argparse
import numpy as np
import torch

from config import Config, STOI, ITOS, VOCAB_SIZE, LN2
from inference import GenomeModel


# ----------------------------- Markov baseline -----------------------------
def _contexts(s, k):
    """Encode every length-k context and its next symbol. Returns (ctx, nxt)."""
    m = len(s) - k
    ctx = np.zeros(m, dtype=np.int64)
    for j in range(k):
        ctx = ctx * VOCAB_SIZE + s[j:j + m].astype(np.int64)
    nxt = s[k:k + m].astype(np.int64)
    return ctx, nxt


def markov_bits(train_ids, val_ids, k, fit_cap=5_000_000):
    """Fit an order-k Markov model (Laplace-smoothed) on train, score val."""
    tr = train_ids[:fit_cap]
    ctx_t, nxt_t = _contexts(tr, k)
    counts = np.ones((VOCAB_SIZE ** k, VOCAB_SIZE), dtype=np.float64)  # +1 Laplace
    np.add.at(counts, (ctx_t, nxt_t), 1.0)
    probs = counts / counts.sum(axis=1, keepdims=True)

    ctx_v, nxt_v = _contexts(val_ids, k)
    p = probs[ctx_v, nxt_v]
    return float(-np.log2(p).mean())


# ----------------------------- helpers -----------------------------
def kmer_freqs(seq, k=6):
    c = {}
    for i in range(len(seq) - k + 1):
        c[seq[i:i + k]] = c.get(seq[i:i + k], 0) + 1
    t = sum(c.values())
    return {km: v / t for km, v in c.items()}


def kl(p, q, eps=1e-9):
    return sum(p.get(x, eps) * np.log(p.get(x, eps) / q.get(x, eps)) for x in set(p) | set(q))


def main():
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=cfg.ckpt_path)
    ap.add_argument("--max_eval_tokens", type=int, default=200_000)
    args = ap.parse_args()

    train_ids = np.fromfile(cfg.train_bin, dtype=np.uint8)
    val_ids = np.fromfile(cfg.val_bin, dtype=np.uint8)[: args.max_eval_tokens]
    val_str = "".join(ITOS[int(i)] for i in val_ids)

    gm = GenomeModel(args.ckpt)
    neural = gm.score(val_str)["bits_per_bp"]

    print(f"\n  evaluating on {len(val_ids):,} val tokens "
          f"(held-out genome region)\n")
    print(f"  {'model':<22}{'bits/token':>12}")
    print(f"  {'-'*34}")
    print(f"  {'random (4 bases)':<22}{2.000:>12.4f}")
    for k in (0, 2, 4, 6, 8):
        b = markov_bits(train_ids, val_ids, k)
        print(f"  {'markov k='+str(k):<22}{b:>12.4f}")
    print(f"  {'neural GPT':<22}{neural:>12.4f}   <-- must beat the best k-gram")

    # qualitative checks
    gen = gm.generate(prompt="A", n_bases=min(len(val_str), 4000), temperature=1.0,
                      top_k=4, seed=0)
    gc = lambda s: (s.count("G") + s.count("C")) / max(1, s.count("G") + s.count("C") +
                                                       s.count("A") + s.count("T"))
    print(f"\n  GC   real={gc(val_str):.4f}  generated={gc(gen):.4f}")
    print(f"  6-mer KL(real||gen) = "
          f"{kl(kmer_freqs(val_str.replace('|','')), kmer_freqs(gen)):.4f}")


if __name__ == "__main__":
    main()
