"""Pure helpers for reading the reading frame out of a landscape (Part 11).

numpy-only: no torch, no model import (the generation.py / variants.py pattern).

Genes are read in triplets, and each genome writes the three codon positions in a
different "style" (Part 10: M. tuberculosis is ~68/50/80% G+C at codon positions
1/2/3). So any per-position signal that tracks that style should oscillate with
period 3 inside genes — and the *phase* of the oscillation gives away the reading
frame. This module turns that into a test with a right answer:

  * take a window of a gene at a random offset (so its first base can sit at any
    of the three codon positions),
  * summarize a per-position signal by phase (three means, `phase_means`),
  * call the frame by matching those three numbers against a genome-specific
    template learned on OTHER genes (`frame_template` / `call_frame`),
  * score against the annotation. Chance is 1/3.

The same caller is applied unchanged to every signal — the neural model's
anticipation (expected G+C / entropy, computed from left context only, before the
base is seen), its saturation landscape, a k-gram's anticipation, and the classic
"GC frame plot" (just the observed bases) — so they're compared on equal terms.
"""

import numpy as np


def orient(x, strand):
    """A per-position array in + strand order -> gene (5'->3') order."""
    x = np.asarray(x, dtype=np.float64)
    return x if strand == "+" else x[::-1]


def window_shift(lo, hi, cds):
    """Codon position (0,1,2) of the first gene-oriented base of + window [lo, hi).

    For a + gene the first gene-oriented base is `lo`; for a - gene it is `hi - 1`.
    """
    if cds["strand"] == "+":
        g = lo - cds["start"]
    else:
        g = cds["end"] - hi
    if g < 0 or (cds["end"] - cds["start"]) - g < hi - lo:
        raise ValueError("window is not inside the CDS")
    return g % 3


def phase_means(x):
    """Mean of x at indices 0,3,6.. / 1,4,7.. / 2,5,8.. (NaNs ignored)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(3, np.nan)
    for o in range(3):
        v = x[o::3]
        v = v[np.isfinite(v)]
        if v.size:
            out[o] = v.mean()
    return out


def _center(v):
    v = np.asarray(v, dtype=np.float64)
    return v - v.mean()


def align_to_codon(v, shift):
    """Rotate phase means so index c holds codon position c (given the true shift)."""
    return np.roll(np.asarray(v, dtype=np.float64), shift)


def frame_template(phase_vectors, shifts):
    """Genome-specific signature: the average centered, unit-scaled phase-means
    vector, aligned to codon positions 1,2,3 (index 0,1,2) using the TRUE shifts."""
    rows = []
    for v, s in zip(phase_vectors, shifts):
        a = _center(align_to_codon(v, s))
        n = np.linalg.norm(a)
        if np.all(np.isfinite(a)) and n > 0:
            rows.append(a / n)
    if not rows:
        raise ValueError("no usable training windows for the template")
    return np.mean(rows, axis=0)


def call_frame(v, template):
    """Predicted shift: the rotation of v's phase means that best matches the template.

    Returns (shift, margin) where margin = best score - second-best score (a
    confidence-like gap; 0 = a coin toss between two frames).
    """
    v = np.asarray(v, dtype=np.float64)
    if not np.all(np.isfinite(v)):
        return 0, 0.0
    scores = np.array([float(_center(align_to_codon(v, r)) @ template) for r in range(3)])
    order = np.argsort(scores)[::-1]
    return int(order[0]), float(scores[order[0]] - scores[order[1]])


def period3_snr(x):
    """Spectral power at period 3 relative to the mean power of the other non-zero
    frequencies (the classic gene-finding signal). ~1 = no period-3 structure."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = (x.size // 3) * 3
    if n < 12:
        return float("nan")
    x = x[:n] - x[:n].mean()
    p = np.abs(np.fft.rfft(x)) ** 2
    k3 = n // 3
    others = np.delete(p[1:], k3 - 1)
    denom = others.mean() if others.size else 0.0
    return float(p[k3] / denom) if denom > 0 else float("nan")


def frame_accuracy(train, test):
    """Fit a template on `train` [(phase_means, true_shift)] and call `test`.

    Returns {"accuracy", "n", "template"}; accuracy is the fraction of test windows
    whose called shift equals the annotated one (chance = 1/3).
    """
    t = frame_template([v for v, _ in train], [s for _, s in train])
    hits = [call_frame(v, t)[0] == s for v, s in test]
    return {
        "accuracy": float(np.mean(hits)) if hits else float("nan"),
        "n": len(hits),
        "template": [float(x) for x in t],
    }
