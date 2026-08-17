"""Pure data-quality statistics — the honest rulers for the *corpus*.

numpy + stdlib only: no torch, no matplotlib, no model. These measure what a
real genome file actually contains (ambiguous bases, low-complexity dust) and,
more importantly, how similar two genomes are — so near-duplicate strains can be
deduped and, critically, kept from leaking across the train/held-out boundary
(which would fake the Part 5 generalization result).

Similarity uses MinHash (the Mash approach): each genome is compressed to a small
fixed-size "fingerprint" of its k-mer set, and Jaccard similarity is estimated
from the fingerprints without ever aligning sequences. k-mers are CANONICAL
(a k-mer and its reverse complement hash the same) so strand orientation never
fakes a difference.
"""

import math

import numpy as np

_MASK64 = (1 << 64) - 1
_BASE_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}


def gc_content(seq):
    """Fraction G+C among A/C/G/T (ignores N / ambiguity). Local copy so this
    module stays free of any non-stdlib/numpy import."""
    s = seq.upper()
    a, c, g, t = s.count("A"), s.count("C"), s.count("G"), s.count("T")
    denom = a + c + g + t
    return (g + c) / denom if denom else 0.0


def n_fraction(seq):
    """Fraction of bases that are NOT A/C/G/T (N and every IUPAC ambiguity code)."""
    if not seq:
        return 0.0
    s = seq.upper()
    acgt = s.count("A") + s.count("C") + s.count("G") + s.count("T")
    return 1.0 - acgt / len(s)


def low_complexity_fraction(seq, window=32, min_entropy_bits=1.5):
    """Fraction of sliding windows whose base-composition Shannon entropy is below
    min_entropy_bits (of a possible 2.0) — homopolymer / simple-repeat 'dust'.

    A pure homopolymer scores 0 bits (all one base); random ACGT approaches 2.0.
    Sequences shorter than `window` are judged as a single window.
    """
    s = seq.upper()
    if len(s) < window:
        return 1.0 if len(s) and _window_entropy(s) < min_entropy_bits else 0.0
    low = 0
    total = len(s) - window + 1
    for i in range(total):
        if _window_entropy(s[i : i + window]) < min_entropy_bits:
            low += 1
    return low / total


def _window_entropy(w):
    counts = [w.count(b) for b in "ACGT"]
    n = sum(counts)
    if n == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c:
            p = c / n
            h -= p * math.log2(p)
    return h


def clean_record(seq, *, min_len=5000, max_n_fraction=0.05, trim_terminal_n=True):
    """Uppercase, map non-ACGTN to N, optionally trim terminal N runs.

    Returns the cleaned sequence, or None if the record should be DROPPED (too
    short, or too many Ns after trimming). Never silently mangles a kept record;
    callers log what was dropped and why.
    """
    s = "".join(c if c in "ACGTN" else "N" for c in seq.upper())
    if trim_terminal_n:
        s = s.strip("N")
    if len(s) < min_len:
        return None
    if n_fraction(s) > max_n_fraction:
        return None
    return s


def clean_action(seq, **clean_kw):
    """Dry-run label for a record: 'keep' or 'drop:<reason>' (does not mutate)."""
    s = "".join(c if c in "ACGTN" else "N" for c in seq.upper())
    if clean_kw.get("trim_terminal_n", True):
        s = s.strip("N")
    if len(s) < clean_kw.get("min_len", 5000):
        return "drop:too_short"
    if n_fraction(s) > clean_kw.get("max_n_fraction", 0.05):
        return "drop:too_many_N"
    return "keep"


# ----------------------------- MinHash fingerprints -----------------------------
def _splitmix64(x):
    """A fast, well-distributed 64-bit mixer — turns a packed k-mer code into a
    uniform hash without per-k-mer hashlib overhead."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def _canonical_hashes(seq, k):
    """Yield the splitmix64 hash of each CANONICAL k-mer (min of a k-mer and its
    reverse complement, in 2-bit encoding). Windows containing a non-ACGT base
    reset the roll, so N/boundaries never create spurious k-mers."""
    s = seq.upper()
    kmask = (1 << (2 * k)) - 1
    shift = 2 * (k - 1)
    fwd = rc = 0
    valid = 0
    for ch in s:
        code = _BASE_CODE.get(ch)
        if code is None:
            valid = 0  # reset on N / ambiguity
            fwd = rc = 0
            continue
        fwd = ((fwd << 2) | code) & kmask
        rc = (rc >> 2) | ((3 - code) << shift)  # complement = 3 - code
        valid += 1
        if valid >= k:
            yield _splitmix64(fwd if fwd < rc else rc)


def minhash_sketch(seq, k=14, sketch_size=400):
    """Bottom-k MinHash fingerprint: the `sketch_size` smallest canonical-k-mer
    hashes, sorted and unique. A whole genome's k-mer set compressed to a fixed,
    comparable signature (returned as a uint64 numpy array)."""
    smallest = set()
    cap = None  # current largest kept hash once the sketch is full
    for h in _canonical_hashes(seq, k):
        if h in smallest:
            continue
        if len(smallest) < sketch_size:
            smallest.add(h)
            if len(smallest) == sketch_size:
                cap = max(smallest)
        elif h < cap:
            smallest.discard(cap)
            smallest.add(h)
            cap = max(smallest)
    return np.array(sorted(smallest), dtype=np.uint64)


def sketch_jaccard(a, b, sketch_size=400):
    """Estimated Jaccard similarity of two bottom-k sketches (0=disjoint,
    1=identical). Takes the smallest k hashes of the union and reports the
    fraction present in BOTH sketches — the standard bottom-k estimator, valid
    even when the two sketches differ in size."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    sa, sb = set(a.tolist()), set(b.tolist())
    k = min(sketch_size, len(a), len(b))
    bottom = np.union1d(a, b)[:k]
    inter = sum(1 for h in bottom.tolist() if h in sa and h in sb)
    return inter / k if k else 0.0


# ----------------------------- corpus-level tools -----------------------------
def _sketches(named, k, sketch_size):
    return [(name, minhash_sketch(seq, k, sketch_size)) for name, seq in named]


def dedup_records(named, *, k=14, sketch_size=400, threshold=0.9):
    """Greedy near-duplicate clustering. Records are considered longest-first;
    a record is dropped if its Jaccard to an already-kept representative is
    >= threshold. Never empties a cluster (the representative always survives).

    Returns (kept_names, dropped) where dropped is
    [{"name", "duplicate_of", "jaccard"}].
    """
    order = sorted(named, key=lambda ns: len(ns[1]), reverse=True)
    kept, dropped = [], []  # kept: [(name, sketch)]
    for name, seq in order:
        sig = minhash_sketch(seq, k, sketch_size)
        dup_of, dup_j = None, 0.0
        for kname, ksig in kept:
            j = sketch_jaccard(sig, ksig, sketch_size)
            if j >= threshold and j > dup_j:
                dup_of, dup_j = kname, j
        if dup_of is not None:
            dropped.append({"name": name, "duplicate_of": dup_of, "jaccard": float(dup_j)})
        else:
            kept.append((name, sig))
    kept_names = [n for n, _ in kept]
    # preserve original input order among kept names
    original = [n for n, _ in named]
    kept_names.sort(key=original.index)
    return kept_names, dropped


def leakage_check(train_named, val_named, *, k=14, sketch_size=400, threshold=0.7):
    """For each held-out (val) genome, its max Jaccard to ANY training genome.

    Returns [{"val", "nearest_train", "jaccard", "leak"}]; leak=True means a
    near-twin of a held-out genome sits in training, so the generalization claim
    for that genome is compromised.
    """
    train_sigs = _sketches(train_named, k, sketch_size)
    out = []
    for vname, vseq in val_named:
        vsig = minhash_sketch(vseq, k, sketch_size)
        nearest, best = None, 0.0
        for tname, tsig in train_sigs:
            j = sketch_jaccard(vsig, tsig, sketch_size)
            if j > best:
                nearest, best = tname, j
        out.append(
            {
                "val": vname,
                "nearest_train": nearest,
                "jaccard": float(best),
                "leak": best >= threshold,
            }
        )
    return out


def qc_report(named, *, k=14, sketch_size=400, dedup_threshold=0.9, **clean_kw):
    """Aggregate corpus report: per-genome cleanliness stats, dedup clusters, and
    the full pairwise MinHash-Jaccard matrix. All plain floats/ints/str — JSON-
    serializable.
    """
    per_genome = []
    for name, seq in named:
        per_genome.append(
            {
                "name": name,
                "length": len(seq),
                "gc": gc_content(seq),
                "n_fraction": n_fraction(seq),
                "low_complexity_fraction": low_complexity_fraction(seq),
                "clean_action": clean_action(seq, **clean_kw),
            }
        )

    sigs = _sketches(named, k, sketch_size)
    names = [n for n, _ in sigs]
    n = len(sigs)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            val = sketch_jaccard(sigs[i][1], sigs[j][1], sketch_size)
            matrix[i][j] = matrix[j][i] = float(val)

    _, dropped = dedup_records(named, k=k, sketch_size=sketch_size, threshold=dedup_threshold)
    return {
        "per_genome": per_genome,
        "similarity": {"names": names, "matrix": matrix},
        "dedup": {"threshold": dedup_threshold, "dropped": dropped},
        "params": {"k": k, "sketch_size": sketch_size},
        "note": (
            "similarity is MinHash-Jaccard (0=distinct, 1=identical); near-duplicate "
            "genomes waste capacity AND can leak across the train/held-out split, "
            "faking generalization — dedup and run a leakage check before trusting a "
            "held-out number."
        ),
    }
