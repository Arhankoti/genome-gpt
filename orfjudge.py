"""Pure helpers for judging ORFs: is this a gene, or a gene's shadow? (Part 13)

numpy-only: no torch, no model import (the frames.py / variants.py pattern). The
neural model's per-position anticipation is passed in as an array, so this module
treats neural and classical signals identically.

Part 12 left a residue no composition null can remove: long *antisense shadow*
ORFs, which run backwards through real genes on the opposite strand. Both a gene
and its shadow are open reading frames — neither contains an in-frame stop — so
the ORF finder can't tell them apart. What differs is *how* each is written,
codon position by codon position. Every signal here is reduced to a profile in the
ORF's own reading frame and orientation:

  * `codon_usage`     — 64 codon frequencies (classic; the core of GeneMark-era
                        gene finders),
  * `phase_profile`   — the four base frequencies at codon positions 1/2/3
                        (12 numbers), from observed bases (classic) OR from a
                        model's anticipation, P(base | left context) (neural / k-gram),
  * `gc_frame_rule`   — GC3 - max(GC1, GC2): the old GC-frame-plot rule that a
                        gene's third codon position is its most GC-rich (label-free).

and scored two ways:

  * `trained_scores`      — nearest-centroid with templates from annotated genes AND
                            shadows of the same genome, 2-fold (the frames.py
                            discipline: never score a row with a template it built);
  * `self_trained_scores` — no annotation at all: similarity to the centroid of the
                            sequence's own long ORFs, refined by repeatedly dropping
                            the less gene-like member of each overlapping antisense
                            pair (GeneMark-style self-training in miniature).
"""

import numpy as np

from classic import reverse_complement

_IDX = {b: i for i, b in enumerate("ACGT")}


def oriented(seq: str, orf: dict, ctx: int = 0) -> tuple[str, int]:
    """The ORF in gene orientation (5'->3' on its strand) with up to `ctx` upstream
    bases in front. Returns (s, off): s[off:] starts at the ORF's start codon."""
    L = len(seq)
    if orf["strand"] == "+":
        lo = max(0, orf["start"] - ctx)
        return seq[lo : orf["end"]], orf["start"] - lo
    hi = min(L, orf["end"] + ctx)
    return reverse_complement(seq[orf["start"] : hi]), hi - orf["end"]


def onehot(s: str) -> np.ndarray:
    """(len(s), 4) indicator of A/C/G/T; a non-ACGT row is NaN."""
    out = np.full((len(s), 4), np.nan)
    for j, ch in enumerate(s.upper()):
        if ch in _IDX:
            out[j] = 0.0
            out[j, _IDX[ch]] = 1.0
    return out


def phase_profile(p, trim: int = 0) -> np.ndarray:
    """12 numbers: mean of each base column at codon positions 1, 2, 3.

    `p` is (n, 4) with row 0 = codon position 1 (the start codon's first base);
    the first `trim` rows are skipped (little left context there). NaN rows ignored.
    """
    p = np.asarray(p, dtype=np.float64)
    ph = np.arange(len(p)) % 3
    keep = np.arange(len(p)) >= trim
    out = np.full(12, np.nan)
    for k in range(3):
        rows = p[keep & (ph == k)]
        rows = rows[np.all(np.isfinite(rows), axis=1)]
        if len(rows):
            out[4 * k : 4 * k + 4] = rows.mean(axis=0)
    return out


def codon_usage(s: str) -> np.ndarray:
    """64 codon frequencies of s read from its first base (ACGT codons only)."""
    s = s.upper()
    out = np.zeros(64)
    for j in range(0, len(s) - 2, 3):
        c = s[j : j + 3]
        if all(ch in _IDX for ch in c):
            out[16 * _IDX[c[0]] + 4 * _IDX[c[1]] + _IDX[c[2]]] += 1
    n = out.sum()
    return out / n if n else out


def gc_frame_rule(profile) -> float:
    """GC3 - max(GC1, GC2) from a 12-number phase profile (higher = more gene-like
    in a GC-rich genome; meaningless in an AT-rich one — that's the point)."""
    v = np.asarray(profile, dtype=np.float64)
    gc = [v[4 * k + 1] + v[4 * k + 2] for k in range(3)]
    return float(gc[2] - max(gc[0], gc[1]))


def centered_cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = a - a.mean(), b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def _cos_rows(X, c):
    return np.array([centered_cosine(x, c) for x in X])


def trained_scores(X, y, seed: int = 0) -> np.ndarray:
    """2-fold nearest-centroid score: cos(x, gene centroid) - cos(x, shadow centroid),
    with both centroids built on the OTHER half. y: 1 = gene, 0 = shadow."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    half = np.zeros(len(y), dtype=bool)
    half[np.random.default_rng(seed).permutation(len(y))[: len(y) // 2]] = True
    out = np.zeros(len(y))
    for tr in (half, ~half):
        g, s = X[tr & (y == 1)], X[tr & (y == 0)]
        if not len(g) or not len(s):
            raise ValueError("each fold needs at least one gene and one shadow")
        out[~tr] = _cos_rows(X[~tr], g.mean(0)) - _cos_rows(X[~tr], s.mean(0))
    return out


def self_trained_scores(X, pairs, n_iter: int = 5) -> np.ndarray:
    """Label-free score: cos(x, centroid of the sequence's own ORFs).

    Starts from the centroid of every row, then `n_iter` times drops the lower-
    scoring member of each overlapping antisense pair (i, j) and rebuilds the
    centroid from the rest. No annotation is used anywhere.
    """
    X = np.asarray(X, dtype=np.float64)
    keep = np.ones(len(X), dtype=bool)
    sc = _cos_rows(X, X.mean(0))
    for _ in range(n_iter):
        keep[:] = True
        for i, j in pairs:
            keep[j if sc[i] >= sc[j] else i] = False
        if not keep.any():
            break
        new = _cos_rows(X, X[keep].mean(0))
        if np.allclose(new, sc):
            break
        sc = new
    return sc


def antisense_pairs(orfs, min_overlap: float = 0.5) -> list[tuple[int, int]]:
    """Index pairs (i, j) of opposite-strand ORFs overlapping by >= min_overlap of
    the shorter one."""
    order = sorted(range(len(orfs)), key=lambda i: orfs[i]["start"])
    out = []
    for a_pos, i in enumerate(order):
        oi = orfs[i]
        for j in order[a_pos + 1 :]:
            oj = orfs[j]
            if oj["start"] >= oi["end"]:
                break
            if oj["strand"] == oi["strand"]:
                continue
            ov = min(oi["end"], oj["end"]) - oj["start"]
            short = min(oi["end"] - oi["start"], oj["end"] - oj["start"])
            if short > 0 and ov >= min_overlap * short:
                out.append((i, j) if i < j else (j, i))
    return sorted(out)


def auc_higher(pos, neg) -> float:
    """P(a random `pos` score > a random `neg` score), ties counted half. 0.5 = chance."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if not pos.size or not neg.size:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):  # average ranks over ties
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    r = ranks[: pos.size].sum()
    return float((r - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def pair_accuracy(scores, pairs, labels) -> float:
    """Over (i, j) pairs holding exactly one gene (label 1) and one shadow (label 0):
    the fraction where the gene scores higher. Ties count half. 0.5 = chance."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    hits = []
    for i, j in pairs:
        if {labels[i], labels[j]} != {0, 1}:
            continue
        g, s = (i, j) if labels[i] == 1 else (j, i)
        hits.append(1.0 if scores[g] > scores[s] else 0.5 if scores[g] == scores[s] else 0.0)
    return float(np.mean(hits)) if hits else float("nan")


def zscore(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v - v.mean()


_JUDGE_NOTE = (
    "Self-trained on this sequence's own long ORFs (no annotation): each ORF's first "
    "`window` bp are summarized by codon usage (classic) and by the genome model's "
    "anticipation P(base | left context) at codon positions 1/2/3 (neural), scored by "
    "similarity to the centroid of the ORFs, refined over overlapping antisense pairs. "
    "For each pair, `classic_pick` / `neural_pick` name the strand that looks more "
    "gene-like. On held-out genomes (Part 13) classic codon usage was the stronger "
    "signal in GC-rich DNA and the neural one on short windows in AT-rich DNA; when "
    "they disagree, neither is reliable. Needs many ORFs (a contig or genome)."
)


def judge_pairs(
    seq: str,
    base_probs_fn,
    *,
    min_codons: int = 100,
    window: int = 480,
    ctx: int = 60,
    max_pool: int = 600,
    min_pool: int = 30,
    top_k: int = 10,
) -> dict:
    """The ORF finder proposes, both judges vote: for every overlapping antisense ORF
    pair in `seq`, which strand looks like the gene? (the `dna_orf_judge` tool)

    `base_probs_fn(s) -> (len(s), 4)` is the neural anticipation (a closure over
    GenomeModel.base_probs). The pool is the `max_pool` longest ORFs of at least
    `min_codons`; fewer than `min_pool` ORFs -> no call (too little to self-train).
    Bounded: counts + at most `top_k` pairs (longest overlap first), never sequence.
    """
    from classic import find_orfs

    seq = seq.upper()
    orfs = find_orfs(seq, min_codons=min_codons)
    if len(orfs) > max_pool:
        orfs = sorted(orfs, key=lambda o: -o["n_codons"])[:max_pool]
        orfs.sort(key=lambda o: (o["start"], o["strand"]))
    base = {"length": len(seq), "min_codons": min_codons, "n_pool": len(orfs)}
    if len(orfs) < min_pool:
        return {
            **base,
            "verdict": "too_few_orfs",
            "plain_english": f"Only {len(orfs)} ORFs of at least {min_codons} codons — too "
            f"few to self-train (need {min_pool}). Give a longer sequence (a contig or "
            "genome) or lower min_codons.",
            "note": _JUDGE_NOTE,
        }
    codon, neural = [], []
    for o in orfs:
        s, off = oriented(seq, o, ctx)
        n = min(window, len(s) - off - 3)
        n -= n % 3
        s = s[: off + n]
        codon.append(codon_usage(s[off:]))
        neural.append(phase_profile(np.asarray(base_probs_fn(s))[off:]))
    pairs = antisense_pairs(orfs)
    sc_c = self_trained_scores(np.array(codon), pairs)
    sc_n = self_trained_scores(np.array(neural), pairs)
    rows = []
    for i, j in pairs:
        a, b = orfs[i], orfs[j]
        plus, minus = (a, b) if a["strand"] == "+" else (b, a)
        ip, im = (i, j) if a["strand"] == "+" else (j, i)
        cp = "+" if sc_c[ip] >= sc_c[im] else "-"
        npk = "+" if sc_n[ip] >= sc_n[im] else "-"
        rows.append(
            {
                "plus_orf": {k: plus[k] for k in ("start", "end", "n_codons")},
                "minus_orf": {k: minus[k] for k in ("start", "end", "n_codons")},
                "overlap_bp": min(a["end"], b["end"]) - max(a["start"], b["start"]),
                "classic_pick": cp,
                "neural_pick": npk,
                "agree": cp == npk,
                "classic_margin": round(float(abs(sc_c[ip] - sc_c[im])), 4),
                "neural_margin": round(float(abs(sc_n[ip] - sc_n[im])), 4),
            }
        )
    rows.sort(key=lambda r: -r["overlap_bp"])
    agree = float(np.mean([r["agree"] for r in rows])) if rows else float("nan")
    return {
        **base,
        "verdict": "judged",
        "n_antisense_pairs": len(rows),
        "agreement_rate": agree,
        "pairs": rows[:top_k],
        "plain_english": f"{len(rows)} overlapping antisense ORF pairs among {len(orfs)} "
        f"long ORFs; the classic and neural judges pick the same strand in "
        f"{agree:.0%} of them."
        if rows
        else f"No overlapping antisense ORF pairs among {len(orfs)} long ORFs.",
        "note": _JUDGE_NOTE,
    }
