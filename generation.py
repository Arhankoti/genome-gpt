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


def composition_shuffle(seq: str, seed: int = 0) -> str:
    """A random permutation of the sequence — identical base composition, zero order.

    This is the honest null for "does the model read grammar, or just composition?":
    scoring `seq` against its own shuffle controls for base/GC frequency exactly, so
    any bits the model saves on the real sequence come from *sequential structure*,
    not from the letters it happens to contain. (Mononucleotide shuffle; it does not
    hold dinucleotide frequency fixed — a documented limitation, not a bug.)
    """
    arr = np.array(list(seq.upper()))
    np.random.default_rng(seed).shuffle(arr)
    return "".join(arr.tolist())


def sequence_complexity(seq: str, k: int = 3) -> float:
    """Distinct-k-mer richness in [0, 1] — a fast repetitiveness proxy.

    len(distinct ACGT k-mers) / min(4**k, #positions). ≈1.0 when the sequence uses
    the full k-mer alphabet (random-like); near 0 when a few repeats dominate (a
    homopolymer or short tandem repeat). Model-free, so a low neural bits/bp can be
    checked against it: low bits with low complexity is *repetition*, not grammar.
    """
    s = seq.upper()
    kmers: set[str] = set()
    positions = 0
    for i in range(len(s) - k + 1):
        km = s[i : i + k]
        if _ACGT.issuperset(km):
            kmers.add(km)
            positions += 1
    if positions == 0:
        return 0.0
    return len(kmers) / min(4**k, positions)


# Verdict tiers are a human gloss over the numbers this function ALWAYS returns —
# below-random margin, grammar gain vs an in-sample composition counter, and
# complexity. Thresholds are conservative and documented; nothing hides behind a word.
_LOW_COMPLEXITY_MAX = 0.30  # below this distinct-k-mer richness => repetitive
_RANDOM_TOL = 0.05  # within this many bits of 2.0 => no usable signal
_GRAMMAR_MIN = 0.02  # neural must beat the counter by this to claim "reads grammar"
_BELOW_RANDOM_MIN = 0.05  # ...and sit at least this far below random

_VERDICT_NOTE = (
    "verdict is a gloss over the returned numbers: margin_vs_random_bits (distance "
    "below the 2.0 random line), grammar_gain_bits (bits the model saves on the real "
    "sequence vs a composition-preserving shuffle of it — i.e. sequential structure "
    "beyond base composition), and complexity (repetitiveness)."
)


def score_verdict(
    seq: str,
    neural_bits: float,
    shuffle_bits: float,
    *,
    random_bits: float = 2.0,
    random_tol: float = _RANDOM_TOL,
) -> dict:
    """Turn a raw bits/bp into a calibrated, plain-English "is this real DNA?".

    The two ways a low/so-so number lies, guarded with computed signals:
      * repetition — a homopolymer scores ~0 bits but isn't grammar (complexity).
      * beating nothing — random DNA sits at ~2.0 (margin_vs_random).
    And the positive signal, composition-controlled: does the model score the real
    sequence below a shuffled copy with the SAME base composition? If so, it's
    reading sequential *order*, not just which letters are present (grammar_gain).

    Pure (numpy/stdlib): the caller passes the model-derived `neural_bits` and
    `shuffle_bits` (the model's bits/bp on composition_shuffle(seq)); this function
    adds the model-free stats and the verdict. Bounded, JSON-serializable.
    """
    neural_bits, shuffle_bits = float(neural_bits), float(shuffle_bits)
    n = sum(1 for c in seq.upper() if c in _ACGT)
    complexity = sequence_complexity(seq)
    gc = gc_content(seq)
    grammar_gain = shuffle_bits - neural_bits  # > 0: model prefers the real order
    margin_vs_random = random_bits - neural_bits  # > 0: below random

    cautions: list[str] = []
    if complexity < _LOW_COMPLEXITY_MAX:
        verdict = "low_complexity"
        plain = (
            "Highly repetitive — the low surprise score reflects repetition, not "
            "natural genome grammar."
        )
        cautions.append("low-complexity: a low bits/bp here is repetition, not grammar")
    elif margin_vs_random < random_tol:
        verdict = "random_like"
        plain = (
            "Indistinguishable from random DNA — the model finds essentially no "
            "structure it can predict."
        )
    elif grammar_gain >= _GRAMMAR_MIN and margin_vs_random >= _BELOW_RANDOM_MIN:
        verdict = "dna_like"
        plain = (
            "Looks like real DNA — the model scores it well below random AND below a "
            "shuffle of the same bases, so it is reading sequential order, not just "
            "base composition."
        )
    else:
        verdict = "plausible_composition"
        plain = (
            "Plausible but unremarkable — below random, but a shuffle of the same "
            "bases scores about as well, so the signal is base composition, not order."
        )

    # Confidence tracks how reliable the bits estimate is (i.e. length); a verdict
    # sitting right on a tier boundary is flagged rather than silently trusted.
    confidence = "high" if n >= 600 else "moderate" if n >= 120 else "low"
    if n < 120:
        cautions.append("short sequence: bits/bp estimate is noisy, verdict low-confidence")
    # only meaningful for signal-bearing verdicts: flag when the below-random margin
    # sits right on the threshold (the random_like/low_complexity calls are decisive).
    if verdict in ("dna_like", "plausible_composition") and margin_vs_random - random_tol < 0.02:
        cautions.append("borderline: below-random margin is small")

    return {
        "verdict": verdict,
        "plain_english": plain,
        "confidence": confidence,
        "neural_bits_per_bp": round(neural_bits, 4),
        "shuffle_bits_per_bp": round(shuffle_bits, 4),
        "grammar_gain_bits": round(grammar_gain, 4),
        "margin_vs_random_bits": round(margin_vs_random, 4),
        "complexity": round(complexity, 4),
        "gc_content": round(gc, 4),
        "n_bases": int(n),
        "cautions": cautions,
        "note": _VERDICT_NOTE,
    }


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
