"""Old-school, deterministic biology tools — Part 12.

numpy/stdlib only: no torch, no model import (the generation.py / variants.py /
frames.py pattern). These are the tools a biologist reached for long before anyone
trained a language model on DNA, and they now sit behind the same bridge schema as
the neural tools so the frontier model can call both and compare:

  * `find_orfs` — open reading frames: a start codon, then codons until the first
    in-frame stop, on all six frames (three per strand).
  * `gc_scan` / `gc_skew_extrema` — sliding-window G+C and GC skew, (G-C)/(G+C).
    The cumulative skew's minimum and maximum are the classic predictors of a
    bacterial chromosome's replication origin and terminus.
  * `restriction_sites` / `digest` — where common restriction enzymes cut, and the
    fragment sizes a digest would leave.

The lesson Parts 9 and 11 kept teaching applies to the dumb tools too: a
deterministic answer is not automatically a meaningful one. Stop codons (TAA, TAG,
TGA) are A/T-rich, so GC-rich DNA has long ORFs *by chance* — the textbook
"≥100 codons" rule that works at 50% GC calls thousands of fake genes in a 65% GC
genome. `orf_shuffle_null` measures that with the same composition-preserving
shuffle the neural tools use, so every ORF can carry an honest "how often does
random DNA of this composition and length contain an ORF this long?".
"""

import re

import numpy as np

from generation import composition_shuffle

_COMP = str.maketrans("ACGTN", "TGCAN")
_IDX = np.full(256, 4, dtype=np.int64)
for _i, _b in enumerate("ACGT"):
    _IDX[ord(_b)] = _i


def _codon_id(c: str) -> int:
    return 16 * "ACGT".index(c[0]) + 4 * "ACGT".index(c[1]) + "ACGT".index(c[2])


# NCBI translation table 11 (bacteria): stops are the standard ones; ATG is the main
# start, GTG and TTG the common alternatives (the rarer ones are left out on purpose).
STOP_CODONS = ("TAA", "TAG", "TGA")
START_CODONS = ("ATG", "GTG", "TTG")

_ORF_NOTE = (
    "ORF = start codon (ATG/GTG/TTG) through the first in-frame stop (TAA/TAG/TGA), "
    "longest form per stop (earliest start), on all six frames; coordinates are "
    "0-based half-open on the + strand and include the stop codon. ORFs running off "
    "either end of the sequence are not reported. A long ORF is not proof of a gene: "
    "stop codons are A/T-rich, so GC-rich DNA has long ORFs by chance — "
    "shuffle_p is how often a composition-matched shuffle of the same length "
    "contains an ORF at least this long."
)


def reverse_complement(seq: str) -> str:
    return seq.upper().translate(_COMP)[::-1]


def _codons(seq: str, frame: int) -> np.ndarray:
    """Codon ids (0..63) of `seq` read from `frame`; -1 for a codon containing a non-ACGT."""
    b = _IDX[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    n = (len(b) - frame) // 3
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    c = b[frame : frame + 3 * n].reshape(n, 3)
    ids = 16 * c[:, 0] + 4 * c[:, 1] + c[:, 2]
    ids[(c == 4).any(axis=1)] = -1
    return ids


def _strand_orfs(s: str, min_codons: int, starts: tuple) -> list[tuple[int, int, int]]:
    """(start_bp, end_bp_incl_stop, frame) for ORFs on one strand, in that strand's coords."""
    stop_ids = np.array([_codon_id(c) for c in STOP_CODONS])
    start_ids = np.array([_codon_id(c) for c in starts])
    out = []
    for frame in range(3):
        ids = _codons(s, frame)
        if ids.size == 0:
            continue
        is_stop = np.isin(ids, stop_ids)
        # an N-containing codon breaks an ORF like a stop does, but ends nothing real
        breaks = np.flatnonzero(is_stop | (ids < 0))
        start_idx = np.flatnonzero(np.isin(ids, start_ids))
        if breaks.size == 0 or start_idx.size == 0:
            continue
        prev = np.concatenate(([-1], breaks[:-1]))
        j = np.searchsorted(start_idx, prev + 1)
        ok = j < start_idx.size
        first = np.where(ok, start_idx[np.minimum(j, start_idx.size - 1)], -1)
        keep = ok & (first < breaks) & is_stop[breaks] & (breaks - first >= min_codons)
        for st, br in zip(first[keep], breaks[keep]):
            out.append((frame + 3 * int(st), frame + 3 * int(br) + 3, frame))
    return out


def find_orfs(seq: str, min_codons: int = 100, starts: tuple = START_CODONS) -> list[dict]:
    """All ORFs of at least `min_codons` sense codons (stop not counted), both strands.

    Returns [{"start", "end", "strand", "frame", "n_codons", "start_codon"}] sorted by
    + strand start; [start, end) is 0-based half-open on the + strand and includes the
    stop. `frame` is the reading frame on that ORF's own strand (0/1/2 from that
    strand's first base). The longest form per stop is reported (earliest start).
    """
    seq = seq.upper()
    L = len(seq)
    out = []
    for strand, s in (("+", seq), ("-", reverse_complement(seq))):
        for a, b, frame in _strand_orfs(s, min_codons, starts):
            lo, hi = (a, b) if strand == "+" else (L - b, L - a)
            out.append(
                {
                    "start": lo,
                    "end": hi,
                    "strand": strand,
                    "frame": frame,
                    "n_codons": (b - a) // 3 - 1,
                    "start_codon": s[a : a + 3],
                }
            )
    out.sort(key=lambda o: (o["start"], o["strand"]))
    return out


def orf_shuffle_null(seq: str, n_shuffles: int = 20, seed: int = 0) -> np.ndarray:
    """Longest ORF (codons) in each of `n_shuffles` composition-preserving shuffles.

    The null for "is this ORF longer than chance?": same length, same base
    composition, order destroyed (generation.composition_shuffle, as in Parts 9/11).
    """
    longest = []
    for i in range(n_shuffles):
        orfs = find_orfs(composition_shuffle(seq, seed=seed + i), min_codons=1)
        longest.append(max((o["n_codons"] for o in orfs), default=0))
    return np.asarray(longest, dtype=np.int64)


def shuffle_p(n_codons: int, null_longest) -> float:
    """Empirical p: P(a shuffle's longest ORF >= n_codons), with the +1 correction."""
    null_longest = np.asarray(null_longest)
    return float((1 + np.sum(null_longest >= n_codons)) / (1 + null_longest.size))


def orf_report(seq: str, min_codons: int = 100, top_k: int = 10, n_shuffles: int = 20) -> dict:
    """Bounded ORF summary for the bridge: counts + the top_k longest, each with a shuffle p.

    Never returns the sequence (or protein) itself — coordinates and scalars only.
    """
    seq = seq.upper()
    orfs = find_orfs(seq, min_codons=min_codons)
    null = orf_shuffle_null(seq, n_shuffles=n_shuffles) if n_shuffles > 0 else np.zeros(0)
    top = sorted(orfs, key=lambda o: -o["n_codons"])[:top_k]
    if null.size:
        for o in top:
            o["shuffle_p"] = shuffle_p(o["n_codons"], null)
    return {
        "length": len(seq),
        "gc": gc_fraction(seq),
        "n_orfs": len(orfs),
        "n_plus": sum(o["strand"] == "+" for o in orfs),
        "n_minus": sum(o["strand"] == "-" for o in orfs),
        "min_codons": min_codons,
        "longest_orfs": top,
        "null_longest_median": float(np.median(null)) if null.size else None,
        "null_longest_max": int(null.max()) if null.size else None,
        "n_shuffles": int(null.size),
        "note": _ORF_NOTE,
    }


# --- G+C and GC skew ---


def gc_fraction(seq: str) -> float:
    """(G+C) / (A+C+G+T); N and other symbols are ignored. 0.0 if no ACGT."""
    seq = seq.upper()
    acgt = sum(seq.count(b) for b in "ACGT")
    return (seq.count("G") + seq.count("C")) / acgt if acgt else 0.0


def gc_scan(seq: str, window: int = 1000, step: int | None = None) -> dict:
    """Sliding-window G+C fraction and GC skew (G-C)/(G+C) over `seq`.

    Windows are [i, i+window) for i = 0, step, 2*step, ... while they fit (one
    window covering everything if `seq` is shorter than `window`). Also returns the
    cumulative skew sum(G-C) over the whole sequence, sampled at each window start.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    seq = seq.upper()
    step = step or window
    b = _IDX[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    g, c, acgt = (b == 2).astype(np.int64), (b == 1).astype(np.int64), (b < 4).astype(np.int64)
    cg, cc, ca = (np.concatenate(([0], np.cumsum(x))) for x in (g, c, acgt))
    w = min(window, len(seq))
    starts = np.arange(0, max(len(seq) - w, 0) + 1, step) if len(seq) else np.zeros(0, int)
    G, C, N = cg[starts + w] - cg[starts], cc[starts + w] - cc[starts], ca[starts + w] - ca[starts]
    with np.errstate(divide="ignore", invalid="ignore"):
        gcf = np.where(N > 0, (G + C) / np.maximum(N, 1), np.nan)
        skew = np.where(G + C > 0, (G - C) / np.maximum(G + C, 1), np.nan)
    return {
        "starts": starts.tolist(),
        "window": int(w),
        "gc": gcf.tolist(),
        "skew": skew.tolist(),
        "cumulative_skew": (cg[starts] - cc[starts]).tolist(),
    }


def gc_skew_extrema(seq: str) -> dict:
    """Positions of the minimum and maximum of the cumulative GC skew sum(G-C).

    On a circular bacterial chromosome the leading strand is G-rich, so cumulative
    skew bottoms out near the replication origin and peaks near the terminus — a
    textbook, model-free way to locate oriC. Also returns `separation`, the
    circular distance between the two as a fraction of the length (~0.5 when the
    signal is clean, since origin and terminus sit roughly opposite each other).
    """
    seq = seq.upper()
    if not seq:
        raise ValueError("empty sequence")
    b = _IDX[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    cum = np.cumsum((b == 2).astype(np.int64) - (b == 1).astype(np.int64))
    lo, hi = int(np.argmin(cum)), int(np.argmax(cum))
    d = abs(hi - lo)
    return {
        "length": len(seq),
        "origin_guess": lo,
        "terminus_guess": hi,
        "skew_range": int(cum[hi] - cum[lo]),
        "separation": min(d, len(seq) - d) / len(seq),
    }


def gc_report(seq: str, window: int = 1000, max_points: int = 40) -> dict:
    """Bounded G+C / skew summary for the bridge: scalars + a downsampled profile."""
    seq = seq.upper()
    scan = gc_scan(seq, window=window)
    gcs = np.asarray(scan["gc"], dtype=float)
    starts = scan["starts"]
    stride = max(1, -(-len(starts) // max_points))
    i_lo, i_hi = int(np.nanargmin(gcs)), int(np.nanargmax(gcs))
    ext = gc_skew_extrema(seq)
    return {
        "length": len(seq),
        "gc": gc_fraction(seq),
        "window": scan["window"],
        "n_windows": len(starts),
        "gc_min": {"start": starts[i_lo], "gc": float(gcs[i_lo])},
        "gc_max": {"start": starts[i_hi], "gc": float(gcs[i_hi])},
        "gc_std": float(np.nanstd(gcs)),
        "profile": [
            {"start": starts[i], "gc": round(float(gcs[i]), 4)}
            for i in range(0, len(starts), stride)
        ],
        "cumulative_skew_min_at": ext["origin_guess"],
        "cumulative_skew_max_at": ext["terminus_guess"],
        "note": "gc = (G+C)/(A+C+G+T) per window. Cumulative GC skew sum(G-C) bottoms out "
        "near the replication origin and peaks near the terminus on a whole circular "
        "bacterial chromosome; on a short fragment those positions mean little.",
    }


# --- restriction enzymes ---

# name -> (recognition site, top-strand cut offset from the site's first base).
# Classic Type II enzymes; IUPAC N = any base.
ENZYMES = {
    "EcoRI": ("GAATTC", 1),
    "BamHI": ("GGATCC", 1),
    "HindIII": ("AAGCTT", 1),
    "XhoI": ("CTCGAG", 1),
    "SalI": ("GTCGAC", 1),
    "PstI": ("CTGCAG", 5),
    "SacI": ("GAGCTC", 5),
    "KpnI": ("GGTACC", 5),
    "SmaI": ("CCCGGG", 3),
    "XbaI": ("TCTAGA", 1),
    "NcoI": ("CCATGG", 1),
    "NdeI": ("CATATG", 2),
    "EcoRV": ("GATATC", 3),
    "NotI": ("GCGGCCGC", 2),
    "DraI": ("TTTAAA", 3),
    "HaeIII": ("GGCC", 2),
    "AluI": ("AGCT", 2),
    "TaqI": ("TCGA", 1),
    "MspI": ("CCGG", 1),
    "HinfI": ("GANTC", 1),
    "BsaI": ("GGTCTC", 7),
}

_IUPAC = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "N": "[ACGT]",
    "R": "[AG]",
    "Y": "[CT]",
    "W": "[AT]",
    "S": "[CG]",
}


def _site_regex(site: str) -> re.Pattern:
    return re.compile("(?=(" + "".join(_IUPAC[ch] for ch in site.upper()) + "))")


def restriction_sites(seq: str, enzymes=None) -> dict:
    """Recognition sites per enzyme: {name: [{"pos", "strand", "cut"}]}.

    `pos` is the site's leftmost base on the + strand. Palindromic sites are found
    once (strand "+"); a non-palindromic site (e.g. BsaI) is also searched on the -
    strand. `cut` is the + strand coordinate of the cut on the strand the site was
    read from
    (top strand for "+", bottom for "-"; it may fall outside the sequence for a
    Type IIS site near an edge).
    """
    seq = seq.upper()
    names = list(ENZYMES) if enzymes is None else list(enzymes)
    out = {}
    for name in names:
        if name not in ENZYMES:
            raise ValueError(f"unknown enzyme {name!r}; known: {', '.join(ENZYMES)}")
        site, cut = ENZYMES[name]
        hits = [
            {"pos": m.start(), "strand": "+", "cut": m.start() + cut}
            for m in _site_regex(site).finditer(seq)
        ]
        rc_site = reverse_complement(site)
        if rc_site != site:  # non-palindromic: also read the bottom strand
            for m in _site_regex(rc_site).finditer(seq):
                # cut offset counts from the site's first base on the - strand
                hits.append({"pos": m.start(), "strand": "-", "cut": m.start() + len(site) - cut})
        hits.sort(key=lambda h: (h["pos"], h["strand"]))
        out[name] = hits
    return out


def digest(seq_len: int, cuts, circular: bool = False) -> list[int]:
    """Fragment lengths after cutting a sequence of `seq_len` bp at `cuts`.

    Cuts outside (0, seq_len) are ignored. Linear: the two ends make fragments.
    Circular: the last piece wraps around to join the first; no cuts -> one
    uncut circle of seq_len.
    """
    cs = sorted({int(c) for c in cuts if 0 < int(c) < seq_len})
    if not cs:
        return [seq_len]
    if circular:
        return [b - a for a, b in zip(cs, cs[1:])] + [seq_len - cs[-1] + cs[0]]
    edges = [0, *cs, seq_len]
    return [b - a for a, b in zip(edges, edges[1:])]


def sites_report(seq: str, enzymes=None, circular: bool = False, max_positions: int = 20) -> dict:
    """Bounded restriction summary for the bridge: per enzyme the count, first
    `max_positions` cut positions and the digest's fragment sizes (largest first)."""
    seq = seq.upper()
    hits = restriction_sites(seq, enzymes)
    per = {}
    for name, hs in hits.items():
        cuts = [h["cut"] for h in hs]
        frags = sorted(digest(len(seq), cuts, circular=circular), reverse=True)
        per[name] = {
            "site": ENZYMES[name][0],
            "n_sites": len(hs),
            "cut_positions": cuts[:max_positions],
            "fragments": frags[:max_positions],
        }
    return {
        "length": len(seq),
        "circular": circular,
        "enzymes": per,
        "cutters": sorted(n for n, v in per.items() if v["n_sites"]),
        "non_cutters": sorted(n for n, v in per.items() if not v["n_sites"]),
        "note": "cut_positions are + strand coordinates of the top-strand cut; lists "
        f"are capped at {max_positions}. Fragment sizes ignore sticky-end overhangs.",
    }


def site_enrichment(seq: str, enzymes=None, n_shuffles: int = 5, seed: int = 0) -> dict:
    """Observed site count vs the mean over composition-preserving shuffles.

    ratio < 1 = the genome carries fewer copies of that site than its base
    composition predicts (classic restriction-site avoidance by bacteria whose own
    restriction-modification systems would cut them).
    """
    seq = seq.upper()
    obs = {n: len(h) for n, h in restriction_sites(seq, enzymes).items()}
    exp = dict.fromkeys(obs, 0.0)
    for i in range(n_shuffles):
        for n, h in restriction_sites(composition_shuffle(seq, seed=seed + i), enzymes).items():
            exp[n] += len(h) / n_shuffles
    return {
        n: {"observed": obs[n], "expected": exp[n], "ratio": obs[n] / exp[n] if exp[n] else None}
        for n in obs
    }


def gene_location(ft_text: str, name: str) -> dict | None:
    """First `gene` feature named `name` in an NCBI feature table, as
    {"start", "end", "strand"} (0-based half-open, + strand), or None."""
    loc = None
    for line in ft_text.splitlines():
        cols = line.split("\t")
        if cols[0] and len(cols) >= 3:
            loc = None
            if cols[2] == "gene" and cols[0].isdigit() and cols[1].isdigit():
                a, b = int(cols[0]), int(cols[1])
                loc = {"start": min(a, b) - 1, "end": max(a, b), "strand": "+" if a <= b else "-"}
        elif loc is not None and len(cols) >= 5 and cols[3] == "gene" and cols[4] == name:
            return loc
    return None


def orf_stop_key(o: dict) -> tuple[str, int]:
    """(strand, + strand coordinate just past the stop codon's 3' end) — how an ORF
    or an annotated CDS is matched: start codons are ambiguous, stops are not."""
    return (o["strand"], o["end"] if o["strand"] == "+" else o["start"])
