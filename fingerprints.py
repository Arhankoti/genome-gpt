"""Pure helpers for judging DNA embeddings — Fingerprints for DNA (Part 14).

numpy/stdlib only: no torch, no model import. The model is passed in as a closure
(an `embed(seq) -> vector` callable), the generation.py pattern.

The trap this module exists to avoid: "these two sequences look alike to the model"
can simply mean "they have the same G+C content". Every claim is therefore measured
next to controls that carry only composition: a GC-only fingerprint, a k-mer
spectrum, the embedding with its linear GC component removed, and queries whose
letters were shuffled (same composition, no order).
"""

import numpy as np

from generation import composition_shuffle, gc_content

_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def kmer_vector(seq: str, k: int = 4) -> np.ndarray:
    """Frequency vector over all 4**k k-mers (ACGT-only windows; zeros if none)."""
    s = seq.upper()
    v = np.zeros(4**k)
    for i in range(len(s) - k + 1):
        w = s[i : i + k]
        if all(c in _IDX for c in w):
            j = 0
            for c in w:
                j = j * 4 + _IDX[c]
            v[j] += 1
    t = v.sum()
    return v / t if t else v


def unit(X) -> np.ndarray:
    """Row-normalize to unit length (zero rows stay zero)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n == 0, 1.0, n)


def cosine(a, b) -> float:
    """Cosine similarity of two vectors (0.0 if either is all zeros)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0


def center(Xref, X) -> np.ndarray:
    """Subtract the reference mean. Mean-pooled hidden states share a large common
    component, so raw cosine is dominated by "all of these are DNA"."""
    return np.asarray(X, float) - np.asarray(Xref, float).mean(axis=0)


def nn_labels(Xref, Xq) -> np.ndarray:
    """Index of the most cosine-similar reference row for every query row."""
    return np.argmax(unit(Xq) @ unit(Xref).T, axis=1)


def nn_accuracy(Xref, yref, Xq, yq) -> float:
    """1-nearest-neighbour label accuracy of queries against a reference set."""
    yref = np.asarray(yref)
    return float(np.mean(yref[nn_labels(Xref, Xq)] == np.asarray(yq)))


def linear_fit(Xref, yref, lam: float = 1.0) -> np.ndarray:
    """Ridge weights (intercept is the last row) for y ~ X."""
    X = np.asarray(Xref, float)
    A = np.hstack([X, np.ones((len(X), 1))])
    reg = lam * np.eye(A.shape[1])
    reg[-1, -1] = 0.0
    return np.linalg.solve(A.T @ A + reg, A.T @ np.asarray(yref, float))


def linear_predict(w, X) -> np.ndarray:
    X = np.asarray(X, float)
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def r2(y, pred) -> float:
    """Coefficient of determination (1 = perfect, 0 = no better than the mean)."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - pred) ** 2).sum()) / tot if tot else 0.0


def regress_r2(Xref, yref, Xq, yq, lam: float = 1.0) -> float:
    """Held-out R^2 of a ridge probe fit on the reference set."""
    return r2(yq, linear_predict(linear_fit(Xref, yref, lam), Xq))


def remove_linear(Xref, gref, X, g, lam: float = 1e-3):
    """Remove from every column the part linearly explained by one covariate
    (e.g. GC). The fit is on the reference set only; applied to both."""
    Xref, X = np.asarray(Xref, float), np.asarray(X, float)
    gref, g = np.asarray(gref, float), np.asarray(g, float)
    W = linear_fit(gref[:, None], Xref, lam)
    return Xref - linear_predict(W, gref[:, None]), X - linear_predict(W, g[:, None])


def probe_accuracy(Xref, yref, Xq, yq, lam: float = 1.0) -> float:
    """Binary linear-probe accuracy (labels 0/1; ridge regression onto ±1)."""
    w = linear_fit(Xref, 2.0 * np.asarray(yref, float) - 1.0, lam)
    return float(np.mean((linear_predict(w, Xq) > 0) == (np.asarray(yq) == 1)))


def standardize(Xref, X):
    """z-score columns using reference statistics. Returns (Xref_z, X_z)."""
    Xref, X = np.asarray(Xref, float), np.asarray(X, float)
    mu, sd = Xref.mean(axis=0), Xref.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (Xref - mu) / sd, (X - mu) / sd


def sample_windows(seq: str, n: int, length: int, lo: int, hi: int, seed: int = 0, max_n=0.02):
    """Up to n windows of `length` starting in [lo, hi-length], <= max_n fraction N.

    Returns [(start, window)]; deterministic given the seed.
    """
    rng = np.random.default_rng(seed)
    out, tries = [], 0
    while len(out) < n and tries < n * 50 and hi - length > lo:
        tries += 1
        s = int(rng.integers(lo, hi - length))
        w = seq[s : s + length]
        if w.count("N") <= max_n * length and "|" not in w:
            out.append((s, w))
    return out


def similarity_report(embed, a: str, b: str, k: int = 4, seed: int = 0) -> dict:
    """Bounded comparison of two sequences, with composition-only controls.

    `cosine_ab` is the raw embedding cosine; `cosine_a_shuffled_b` is the same
    cosine after b's letters are shuffled (b's composition, no order), so
    `order_gain` = cosine_ab - cosine_a_shuffled_b is the part attributable to
    sequence order. kmer_cosine / gc_difference are the classic comparisons.

    `verdict` is a gloss on order_gain alone: "order_beyond_composition" when the
    real b is closer to a than b's own shuffle, else "no_evidence_beyond_composition".
    """
    ea, eb = np.asarray(embed(a)), np.asarray(embed(b))
    es = np.asarray(embed(composition_shuffle(b, seed=seed)))
    c_ab, c_as = cosine(ea, eb), cosine(ea, es)
    return {
        "verdict": "order_beyond_composition" if c_ab > c_as else "no_evidence_beyond_composition",
        "cosine_ab": round(c_ab, 4),
        "cosine_a_shuffled_b": round(c_as, 4),
        "order_gain": round(c_ab - c_as, 4),
        "kmer_cosine": round(cosine(kmer_vector(a, k), kmer_vector(b, k)), 4),
        "gc_a": round(gc_content(a), 4),
        "gc_b": round(gc_content(b), 4),
        "gc_difference": round(abs(gc_content(a) - gc_content(b)), 4),
    }
