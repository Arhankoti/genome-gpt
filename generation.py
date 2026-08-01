"""Pure statistics for judging generated DNA — the referee for generation.

numpy-only: no torch, no model import. Anything that needs the model (generate
/ score) is passed in as a closure, so this module stays trivially testable and
importable from evaluate.py, the tests, and the bridge without pulling torch or
matplotlib.

The honesty discipline (carried from Parts 2-3): a "good" dream must be good on
TWO axes at once —
  * fidelity  — do its k-mer statistics match real DNA? (js_divergence, gc)
  * novelty   — is it *producing* sequence, not *copying* it? (copy_stats)
A perfect fidelity score with high copying is memorization, not generation.
"""

import numpy as np

_ACGT = frozenset("ACGT")


def gc_content(seq: str) -> float:
    """Fraction of G+C among A/C/G/T. N and | boundaries are ignored."""
    s = seq.upper()
    a, c, g, t = s.count("A"), s.count("C"), s.count("G"), s.count("T")
    denom = a + c + g + t
    return (g + c) / denom if denom else 0.0


def kmer_spectrum(seq: str, k: int = 4) -> dict:
    """Normalized k-mer frequencies over ACGT-only k-mers.

    Any window containing N or a | boundary is skipped, so boundaries never
    create spurious k-mers. Returns {kmer: probability} summing to 1.0 (or an
    empty dict if no valid window exists).
    """
    s = seq.upper()
    counts: dict[str, int] = {}
    for i in range(len(s) - k + 1):
        km = s[i : i + k]
        if _ACGT.issuperset(km):
            counts[km] = counts.get(km, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {km: v / total for km, v in counts.items()}


def js_divergence(p: dict, q: dict, base: float = 2.0) -> float:
    """Jensen-Shannon divergence between two spectra, in bits by default.

    0 = identical; with base 2 it is bounded in [0, 1]. Symmetric — unlike the
    raw KL evaluate.py used to print, which is unbounded and asymmetric.
    """
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    logb = np.log(base)

    def _kl_to_mixture(a: dict) -> float:
        s = 0.0
        for x in keys:
            ax = a.get(x, 0.0)
            if ax > 0.0:
                m = 0.5 * (p.get(x, 0.0) + q.get(x, 0.0))  # always > 0 when ax > 0
                s += ax * (np.log(ax) - np.log(m)) / logb
        return s

    return float(0.5 * _kl_to_mixture(p) + 0.5 * _kl_to_mixture(q))


def _longest_exact_match(gen: str, ref: str) -> int:
    """Length of the longest substring of `gen` that also occurs in `ref`.

    Binary search on the match length: for a candidate length L, ask whether any
    length-L window of gen appears among ref's length-L windows. O((|gen|+|ref|))
    set work per probe, O(log min(|gen|,|ref|)) probes. Cost scales with |ref|,
    so callers should pass a bounded reference window.
    """

    def _shares_window(length: int) -> bool:
        if length == 0:
            return True
        ref_windows = {ref[i : i + length] for i in range(len(ref) - length + 1)}
        return any(gen[i : i + length] in ref_windows for i in range(len(gen) - length + 1))

    lo, hi = 0, min(len(gen), len(ref))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _shares_window(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def copy_stats(gen: str, ref: str, k: int = 20) -> dict:
    """Plagiarism check — how much of `gen` is copied verbatim from `ref`.

    Returns:
        copied_kmer_fraction: fraction of length-k gen k-mers found verbatim in
            ref (1.0 = every k-mer is lifted from the reference).
        longest_exact_match: length of the longest substring of gen occurring in
            ref.
    High values mean the model is regurgitating, not generating.
    """
    g, r = gen.upper(), ref.upper()
    if len(g) < k or len(r) < k:
        copied_fraction = 0.0
    else:
        ref_kmers = {r[i : i + k] for i in range(len(r) - k + 1)}
        gen_kmers = [g[i : i + k] for i in range(len(g) - k + 1)]
        copied_fraction = sum(km in ref_kmers for km in gen_kmers) / len(gen_kmers)
    return {
        "copied_kmer_fraction": float(copied_fraction),
        "longest_exact_match": int(_longest_exact_match(g, r)),
    }


_REPORT_NOTE = (
    "fidelity (kmer_js_bits) and novelty (copied_kmer_fraction) must BOTH be "
    "good; a low js with high copying means the model is memorizing, not "
    "generating. self_bits_per_bp is a weak sanity check only."
)


def dream_report(
    generate_fn,
    score_fn,
    ref_seq,
    *,
    prompt="A",
    n_bases=2000,
    temperatures=(0.5, 0.7, 0.9, 1.1),
    k_fidelity=4,
    k_copy=20,
    seed=0,
):
    """Sweep sampling temperature and report fidelity + novelty at each step.

    The stats module never imports the model; the caller passes thin closures:
        generate_fn(prompt, n_bases, temperature, seed) -> str
        score_fn(seq) -> bits_per_bp   (the model's own surprise; weak sanity)

    `ref_seq` should be HELD-OUT DNA (val region / a genome the model never
    trained on) so fidelity isn't measured against memorized text. Every value
    returned is a plain Python float/int, so the result is JSON-serializable and
    the bridge can reuse it.
    """
    ref_spectrum = kmer_spectrum(ref_seq, k_fidelity)
    ref_gc = gc_content(ref_seq)
    sweep = []
    for t in temperatures:
        gen = generate_fn(prompt, n_bases, float(t), seed)
        g_gc = gc_content(gen)
        cs = copy_stats(gen, ref_seq, k_copy)
        sweep.append(
            {
                "temperature": float(t),
                "gc": g_gc,
                "gc_abs_error": abs(g_gc - ref_gc),
                "kmer_js_bits": js_divergence(kmer_spectrum(gen, k_fidelity), ref_spectrum),
                "copied_kmer_fraction": cs["copied_kmer_fraction"],
                "longest_exact_match": cs["longest_exact_match"],
                "self_bits_per_bp": float(score_fn(gen)),
            }
        )
    return {
        "reference": {"gc": ref_gc, "self_bits_per_bp": float(score_fn(ref_seq))},
        "sweep": sweep,
        "k_fidelity": k_fidelity,
        "k_copy": k_copy,
        "n_bases": n_bases,
        "note": _REPORT_NOTE,
    }
