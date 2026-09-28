"""Pure helpers for calling single-base variants — the Variant Detective (Part 10).

numpy/stdlib only: no torch, no model import. Anything that needs the model is
passed in as a closure (the generation.py pattern), so this module stays trivially
testable and importable from the bridge, detective.py, and the tests.

The trap this module exists to avoid: a raw LLR's *sign* says almost nothing.
Nearly every change to real DNA makes it a little less likely under a model that
learned real DNA, so "negative = disruptive" fires for almost everything. The
honest question is relative — *is this change more surprising than the other
changes the model could have seen right here?* So a variant is calibrated against
its own local background: every other single-base substitution within ±flank bp,
scored the same way (the model judges its own control, as in Part 9).

The biology the codon test leans on (model-free labels from gene annotations):
inside a protein-coding gene, a substitution is
  * synonymous — the codon still encodes the same amino acid,
  * missense   — it encodes a different amino acid,
  * nonsense   — it becomes a stop codon (the protein is truncated).
Evolution tolerates these in roughly that order, so a model that learned coding
grammar should rank nonsense < missense < synonymous by LLR.
"""

import numpy as np

from generation import composition_shuffle, score_verdict

_ACGT = "ACGT"
_COMP = str.maketrans("ACGTN", "TGCAN")

# NCBI translation table 11 (bacteria/archaea). Sense and stop codons are identical
# to the standard code; table 11 differs only in which codons may act as *starts*,
# which is irrelevant to substitutions inside a gene body.
_BASES = "TCAG"
_AAS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE = {
    a + b + c: _AAS[16 * i + 4 * j + k]
    for i, a in enumerate(_BASES)
    for j, b in enumerate(_BASES)
    for k, c in enumerate(_BASES)
}

VARIANT_CLASSES = ("synonymous", "missense", "nonsense")

# verdict thresholds on the local disruption percentile (fraction of nearby
# substitutions the variant is MORE disruptive than)
_DISRUPTIVE_PCT = 0.90
_TOLERATED_PCT = 0.50
_MIN_LEFT_CONTEXT = 32  # bases of left context below which the call is flagged

_VERDICT_NOTE = (
    "LLR = logP(alt) - logP(ref) over a window centered on the variant (both flanks "
    "+ downstream ripple), in nats. The verdict is relative: disruption_percentile is "
    "the fraction of other single-base substitutions within ±flank bp that this one "
    "is more disruptive than. A model-plausibility call, not a clinical or "
    "pathogenicity prediction."
)


def reverse_complement(seq: str) -> str:
    return seq.upper().translate(_COMP)[::-1]


def classify_substitution(codon: str, codon_pos: int, alt: str) -> str:
    """Class of substituting codon[codon_pos] -> alt (all on the gene's strand).

    Returns one of synonymous / missense / nonsense / stop_loss / stop_to_stop.
    """
    codon = codon.upper()
    if codon not in CODON_TABLE:
        raise ValueError(f"not an ACGT codon: {codon!r}")
    if alt.upper() == codon[codon_pos]:
        raise ValueError("alt base equals the reference base")
    mut = codon[:codon_pos] + alt.upper() + codon[codon_pos + 1 :]
    ref_aa, alt_aa = CODON_TABLE[codon], CODON_TABLE[mut]
    if ref_aa == "*":
        return "stop_to_stop" if alt_aa == "*" else "stop_loss"
    if alt_aa == "*":
        return "nonsense"
    return "synonymous" if alt_aa == ref_aa else "missense"


def parse_feature_table(text: str) -> list[dict]:
    """Parse an NCBI 5-column feature table into simple, trustworthy CDS intervals.

    Returns [{"start", "end", "strand"}] with 0-based half-open coordinates on the +
    strand. Deliberately conservative — anything that isn't a plain single-interval,
    complete, non-pseudo CDS is dropped (joins/frameshifts, `<`/`>` partial ends,
    `pseudo`), because a wrong reading frame would mislabel every variant in it.
    """
    out, cur, bad = [], None, False

    def flush():
        if cur is not None and not bad:
            out.append(cur)

    for line in text.splitlines():
        if not line or line.startswith(">"):
            continue
        cols = line.split("\t")
        if cols[0]:  # a location line: "start\tend[\tfeature_key]"
            if len(cols) >= 3 and cols[2]:
                flush()
                cur, bad = None, False
                if cols[2] == "CDS":
                    a, b = cols[0], cols[1]
                    if not (a.isdigit() and b.isdigit()):
                        bad = True  # partial (<, >) ends
                        cur = {}
                        continue
                    a, b = int(a), int(b)
                    strand = "+" if a <= b else "-"
                    cur = {"start": min(a, b) - 1, "end": max(a, b), "strand": strand}
            elif cur is not None:
                bad = True  # a continuation interval: a multi-part (joined) CDS
        elif cur is not None and len(cols) >= 4 and cols[3].strip() == "pseudo":
            bad = True
    flush()
    return [c for c in out if c and (c["end"] - c["start"]) % 3 == 0]


def codon_variants(genome: str, cds: dict, codon_idx: int) -> list[dict]:
    """All 9 single-base substitutions of one codon of a CDS, with model-free labels.

    Positions and ref/alt bases are reported on the genome's + strand (what the
    model reads); the class is computed on the gene's own strand. `gc_neutral` marks
    substitutions that don't change G+C content (S<->S or W<->W), for a
    composition-controlled comparison.
    """
    strand = cds["strand"]
    if strand == "+":
        g0 = cds["start"] + 3 * codon_idx
        codon = genome[g0 : g0 + 3].upper()
        plus_pos = [g0, g0 + 1, g0 + 2]
    else:
        g_end = cds["end"] - 3 * codon_idx  # codon occupies [g_end-3, g_end) on +
        codon = reverse_complement(genome[g_end - 3 : g_end])
        plus_pos = [g_end - 1, g_end - 2, g_end - 3]
    if codon not in CODON_TABLE:
        return []
    rows = []
    for cp in range(3):
        for alt in _ACGT:
            if alt == codon[cp]:
                continue
            p = plus_pos[cp]
            plus_ref = genome[p].upper()
            plus_alt = alt if strand == "+" else alt.translate(_COMP)
            rows.append(
                {
                    "pos": p,
                    "ref": plus_ref,
                    "alt": plus_alt,
                    "codon_pos": cp + 1,
                    "klass": classify_substitution(codon, cp, alt),
                    "gc_neutral": (plus_ref in "GC") == (plus_alt in "GC"),
                }
            )
    return rows


def separation_auc(disruptive, benign) -> float:
    """P(a random `disruptive` variant has a LOWER LLR than a random `benign` one).

    The Mann-Whitney AUC with ties counted half: 0.5 = the scores can't tell the
    two classes apart; 1.0 = every disruptive variant scores below every benign one.
    """
    d = np.asarray(disruptive, dtype=np.float64)
    b = np.asarray(benign, dtype=np.float64)
    if d.size == 0 or b.size == 0:
        return float("nan")
    allv = np.concatenate([d, b])
    # average ranks (ties share their mean rank)
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, dtype=np.float64)
    sorted_v = allv[order]
    i = 0
    while i < allv.size:
        j = i
        while j + 1 < allv.size and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    # U for "benign ranks above disruptive" == P(d < b)
    r_b = ranks[d.size :].sum()
    u = r_b - b.size * (b.size + 1) / 2.0
    return float(u / (d.size * b.size))


def disruption_percentile(llr: float, background) -> float:
    """Fraction of `background` LLRs strictly above `llr` (ties count half):
    how many nearby substitutions this one is more disruptive than."""
    bg = np.asarray(background, dtype=np.float64)
    if bg.size == 0:
        return float("nan")
    return float(((bg > llr).sum() + 0.5 * (bg == llr).sum()) / bg.size)


def variant_verdict(llr, background, *, context_verdict=None, left_context=None) -> dict:
    """Turn a raw LLR into a calibrated, plain-English "did this change break something?".

    Pure: the caller passes the model-derived `llr` and the LLRs of every other
    substitution nearby (`background`); this function ranks one against the other.
    `context_verdict` (a generation.score_verdict verdict for the surrounding DNA)
    guards against calling a variant in DNA the model can't read at all.
    Bounded, JSON-serializable.
    """
    llr = float(llr)
    bg = np.asarray(background, dtype=np.float64)
    pct = disruption_percentile(llr, bg)
    med = float(np.median(bg)) if bg.size else float("nan")
    mad = float(np.median(np.abs(bg - med))) if bg.size else float("nan")
    robust_z = (llr - med) / (1.4826 * mad) if bg.size and mad > 0 else 0.0

    cautions: list[str] = []
    if context_verdict in ("random_like", "low_complexity"):
        verdict = "unreliable_context"
        plain = (
            f"Can't judge — the surrounding DNA reads as {context_verdict}, so the "
            "model has no grammar here for a change to disrupt."
        )
    elif llr >= 0:
        verdict = "likely_tolerated"
        plain = (
            "Likely tolerated — the model finds the new base at least as expected as "
            "the original here."
        )
    elif pct >= _DISRUPTIVE_PCT:
        verdict = "likely_disruptive"
        plain = (
            f"Likely disruptive — more surprising to the model than "
            f"{pct:.0%} of the other single-letter changes nearby."
        )
    elif pct >= _TOLERATED_PCT:
        verdict = "uncertain"
        plain = (
            f"Unremarkable — more surprising than {pct:.0%} of nearby changes; "
            "the model doesn't single this one out."
        )
    else:
        verdict = "likely_tolerated"
        plain = (
            f"Likely tolerated — less surprising than most nearby changes "
            f"(only beats {pct:.0%} of them)."
        )

    confidence = "moderate" if bg.size >= 90 else "low"
    if bg.size < 30:
        cautions.append("small background: percentile is coarse")
    if left_context is not None and left_context < _MIN_LEFT_CONTEXT:
        confidence = "low"
        cautions.append("near the sequence start: little left context, LLR is noisy")
    if verdict in ("likely_disruptive", "uncertain", "likely_tolerated") and bg.size:
        edge = min(abs(pct - _DISRUPTIVE_PCT), abs(pct - _TOLERATED_PCT))
        if edge < 0.03:
            cautions.append("borderline: percentile sits on a verdict threshold")

    return {
        "verdict": verdict,
        "plain_english": plain,
        "confidence": confidence,
        "llr": round(llr, 4),
        "disruption_percentile": round(pct, 4),
        "robust_z": round(float(robust_z), 3),
        "background_median_llr": round(med, 4),
        "n_background": int(bg.size),
        "cautions": cautions,
        "note": _VERDICT_NOTE,
    }


def neighborhood_variants(seq: str, center: int, flank: int) -> list[tuple[int, str]]:
    """Every ACGT substitution at positions center±flank (clipped to the sequence)."""
    s = seq.upper()
    out = []
    for p in range(max(0, center - flank), min(len(s), center + flank + 1)):
        if s[p] not in _ACGT:
            continue
        out.extend((p, a) for a in _ACGT if a != s[p])
    return out


def variant_report(seq, pos, alt, *, effects_fn, score_fn, flank=30, window=256) -> dict:
    """Compose one calibrated variant call from model closures.

        effects_fn(seq, [(pos, alt), ...]) -> [llr, ...]   (GenomeModel.variant_effects)
        score_fn(seq) -> bits_per_bp                         (GenomeModel.score)

    Scores the variant together with its ±flank background in one batched call,
    runs the Part 9 context check on the window around it, and returns the verdict
    plus every number behind it (bounded — never the sequence).
    """
    s = seq.upper()
    alt = alt.upper()
    if not 0 <= pos < len(s):
        raise ValueError(f"pos {pos} out of range for a {len(s)}-bp sequence")
    if alt not in _ACGT:
        raise ValueError(f"alt base must be one of A/C/G/T, got {alt!r}")
    if alt == s[pos]:
        raise ValueError(f"alt base {alt} equals the reference base at pos {pos}")

    background = [v for v in neighborhood_variants(s, pos, flank) if v != (pos, alt)]
    llrs = effects_fn(s, [(pos, alt)] + background)
    llr, bg = llrs[0], llrs[1:]

    lo = max(0, min(pos - window // 2, len(s) - window))
    ctx = s[lo : lo + window]
    ctx_v = score_verdict(ctx, score_fn(ctx), score_fn(composition_shuffle(ctx)))

    v = variant_verdict(llr, bg, context_verdict=ctx_v["verdict"], left_context=pos)
    v.update(
        {
            "position": int(pos),
            "ref_base": s[pos],
            "alt_base": alt,
            "flank": int(flank),
            "context_verdict": ctx_v["verdict"],
            "context_bits_per_bp": ctx_v["neural_bits_per_bp"],
        }
    )
    return v
