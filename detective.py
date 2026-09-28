"""CLI: the Variant Detective — did changing one letter break something? (Part 10)

Two modes:

    # one calibrated call: the variant ranked against every substitution within ±flank bp
    python detective.py --ckpt checkpoints/real.pt --seq ACGT... --pos 120 --alt T

    # the codon test: does the model rank nonsense < missense < synonymous on a
    # HELD-OUT genome, using its gene annotations as model-free ground truth?
    python data/download.py --cds m_tuberculosis_h37rv
    python detective.py --ckpt checkpoints/real.pt --codon_test \\
        --genome m_tuberculosis_h37rv --n_codons 600 --out variants.json --fig variant_classes.png

    # the "writing style" view: codon-position G/C<->A/T shift across genomes
    python detective.py --style_compare mtb.json,hp.json --labels "M. tb,H. pylori" \\
        --fig codon_style.png

The codon test scores every one of the 9 single-base substitutions of randomly
sampled codons from annotated genes. Each class gets two scores — the neural
whole-window LLR, and an order-k Markov LLR fit on the training split (the
counter the neural net must beat, Part 2) — and the headline number per class
pair is a separation AUC: P(a nonsense variant scores lower than a synonymous
one). 0.5 = can't tell them apart. The gc_neutral subset (substitutions that
don't change G+C) controls for "the model just dislikes A/T in a GC-rich genome".
"""

import argparse
import json

import numpy as np

from config import STOI, Config
from data.download import cds_path
from data.prepare import read_named_records
from evaluate import markov_fit, markov_variant_llr
from inference import GenomeModel
from variants import (
    VARIANT_CLASSES,
    codon_variants,
    neighborhood_variants,
    parse_feature_table,
    separation_auc,
    variant_report,
    variant_verdict,
)

_PAIRS = (("nonsense", "synonymous"), ("missense", "synonymous"), ("nonsense", "missense"))


def sample_codons(genome, cds_list, n_codons, *, margin, seed=0, min_codons=100):
    """Seeded (cds, codon_idx) picks: a random gene, then a random interior codon.

    Skips the first/last 5 codons (start/stop neighborhoods) and any codon too close
    to the genome ends for a full centered window (`margin` bp each side).
    """
    rng = np.random.default_rng(seed)
    genes = [c for c in cds_list if (c["end"] - c["start"]) // 3 >= min_codons]
    picks, picked = [], set()
    for _ in range(50 * n_codons):
        if len(picks) == n_codons:
            break
        c = genes[int(rng.integers(len(genes)))]
        idx = int(rng.integers(5, (c["end"] - c["start"]) // 3 - 5))
        g0 = c["start"] + 3 * idx if c["strand"] == "+" else c["end"] - 3 * idx - 3
        if (c["start"], idx) in picked or g0 - margin < 0 or g0 + 3 + margin > len(genome):
            continue
        picked.add((c["start"], idx))
        picks.append((c, idx))
    return picks


def codon_test(
    gm,
    genome,
    cds_list,
    *,
    n_codons=600,
    markov=None,
    verdict_codons=0,
    flank=30,
    seed=0,
):
    """Score all 9 substitutions of sampled codons; return rows + class summaries.

    markov: optional (probs, k) from evaluate.markov_fit for the k-gram baseline.
    verdict_codons: for the first N codons, also run the calibrated verdict
        (each variant ranked against every substitution within ±flank bp).
    """
    W = gm.cfg.block_size
    margin = W + flank
    rows = []
    for ci, (cds, idx) in enumerate(
        sample_codons(genome, cds_list, n_codons, margin=margin, seed=seed)
    ):
        vs = codon_variants(genome, cds, idx)
        if not vs:
            continue
        # a local slice with a full window of context on both sides of the codon
        lo = min(v["pos"] for v in vs) - margin
        local = genome[lo : lo + 2 * margin + 3]
        local_vs = [(v["pos"] - lo, v["alt"]) for v in vs]
        with_bg = ci < verdict_codons
        bg_vs = []
        if with_bg:
            # every substitution within ±flank of any codon position (the codon's own
            # variants are already in local_vs, so they're scored once and reused)
            centre = sum(p for p, _ in local_vs) // len(local_vs)
            bg_vs = [
                x for x in neighborhood_variants(local, centre, flank + 1) if x not in local_vs
            ]
        llrs = gm.variant_effects(local, local_vs + bg_vs)
        lookup = dict(zip(local_vs + bg_vs, llrs))
        ids = np.array([STOI.get(c, STOI["N"]) for c in local], dtype=np.int64)
        for v, (lp, la), llr in zip(vs, local_vs, llrs):
            row = dict(v, llr_neural=llr, gene_start=cds["start"], codon_idx=idx)
            if markov is not None:
                probs, k = markov
                row["llr_markov"] = markov_variant_llr(probs, k, ids, lp, STOI[la])
            if with_bg:
                bg = [
                    val
                    for (p, a), val in lookup.items()
                    if abs(p - lp) <= flank and (p, a) != (lp, la)
                ]
                row["verdict"] = variant_verdict(llr, bg)["verdict"]
            rows.append(row)
    return {"rows": rows, "summary": summarize(rows)}


def summarize(rows):
    """Per-class LLR stats, per-codon-position means, GC-direction shifts, and
    separation AUCs.

    `sign_negative_frac` is what the old sign-only rule ("llr < 0 => disruptive")
    would have flagged. `gc_shift` splits each codon position by the direction of
    the change — G/C->A/T (S>W) vs A/T->G/C (W>S) — which exposes a model that has
    learned the genome's codon-position composition (its "writing style").
    """
    scorers = ["llr_neural"] + (["llr_markov"] if rows and "llr_markov" in rows[0] else [])
    out = {
        "n_variants": len(rows),
        "classes": {},
        "codon_position": {},
        "gc_shift": {},
        "auc": {},
    }
    for cls in VARIANT_CLASSES:
        sub = [r for r in rows if r["klass"] == cls]
        out["classes"][cls] = {"n": len(sub)}
        for s in scorers:
            vals = np.array([r[s] for r in sub]) if sub else np.array([np.nan])
            out["classes"][cls][s] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
            }
        out["classes"][cls]["sign_negative_frac"] = (
            float(np.mean([r["llr_neural"] < 0 for r in sub])) if sub else float("nan")
        )
        called = [r for r in sub if "verdict" in r]
        if called:
            out["classes"][cls]["n_verdicts"] = len(called)
            out["classes"][cls]["frac_likely_disruptive"] = float(
                np.mean([r["verdict"] == "likely_disruptive" for r in called])
            )
    for cp in (1, 2, 3):
        sub = [r for r in rows if r["codon_pos"] == cp]
        out["codon_position"][str(cp)] = {
            s: float(np.mean([r[s] for r in sub])) if sub else float("nan") for s in scorers
        }
        out["gc_shift"][str(cp)] = {}
        for label, ref_s, alt_s in (("S>W", True, False), ("W>S", False, True)):
            d = [r for r in sub if (r["ref"] in "GC") == ref_s and (r["alt"] in "GC") == alt_s]
            out["gc_shift"][str(cp)][label] = {"n": len(d)} | {
                s: float(np.mean([r[s] for r in d])) if d else float("nan") for s in scorers
            }
    for subset, keep in (("all", lambda r: True), ("gc_neutral", lambda r: r["gc_neutral"])):
        out["auc"][subset] = {}
        for a, b in _PAIRS:
            key = f"{a}_vs_{b}"
            ra = [r for r in rows if r["klass"] == a and keep(r)]
            rb = [r for r in rows if r["klass"] == b and keep(r)]
            out["auc"][subset][key] = {
                s: separation_auc([r[s] for r in ra], [r[s] for r in rb]) for s in scorers
            }
            out["auc"][subset][key]["n"] = [len(ra), len(rb)]
    return out


def _print_summary(summ, k):
    print(f"\n  {summ['n_variants']} variants scored")
    print(
        f"\n  {'class':<12}{'n':>6}{'neural mean':>14}{'markov mean':>14}"
        f"{'% llr<0':>10}{'% likely_disr':>15}"
    )
    for cls, c in summ["classes"].items():
        mk = c.get("llr_markov", {}).get("mean", float("nan"))
        fd = c.get("frac_likely_disruptive")
        fd_s = f"{fd:>14.0%}" if fd is not None and np.isfinite(fd) else f"{'—':>14}"
        print(
            f"  {cls:<12}{c['n']:>6}{c['llr_neural']['mean']:>14.3f}{mk:>14.3f}"
            f"{c['sign_negative_frac']:>10.0%} {fd_s}"
        )
    print(f"\n  {'codon pos':<12}{'neural mean':>14}{'markov mean':>14}")
    for cp, c in summ["codon_position"].items():
        print(f"  {cp:<12}{c['llr_neural']:>14.3f}{c.get('llr_markov', float('nan')):>14.3f}")
    print(f"\n  {'codon pos':<12}{'change':<8}{'n':>6}{'neural mean':>14}{'markov mean':>14}")
    for cp, d in summ["gc_shift"].items():
        for label, c in d.items():
            mk = c.get("llr_markov", float("nan"))
            print(f"  {cp:<12}{label:<8}{c['n']:>6}{c['llr_neural']:>14.3f}{mk:>14.3f}")
    print(f"\n  separation AUC (0.5 = can't tell apart; markov k={k})")
    for subset, pairs in summ["auc"].items():
        for key, a in pairs.items():
            print(
                f"  {subset:<11}{key:<26}neural {a['llr_neural']:.3f}   "
                f"markov {a.get('llr_markov', float('nan')):.3f}   n={a['n']}"
            )


def _print_report(v):
    print(f"\n  verdict: {v['verdict']}  ({v['confidence']} confidence)")
    print(f"  {v['plain_english']}")
    print(
        f"    {v['ref_base']}->{v['alt_base']} at {v['position']} | llr {v['llr']:+.4f} nats | "
        f"percentile {v['disruption_percentile']:.3f} | robust z {v['robust_z']:+.2f} | "
        f"background n={v['n_background']} (median {v['background_median_llr']:+.4f})"
    )
    print(f"    context: {v['context_verdict']} ({v['context_bits_per_bp']:.4f} b/bp)")
    for c in v["cautions"]:
        print(f"    caution: {c}")


def main():
    cfg = Config()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", default=cfg.ckpt_path, help="Checkpoint path")
    ap.add_argument("--seq", default=None, help="Reference DNA for a single call")
    ap.add_argument("--pos", type=int, default=None, help="0-based variant position (--seq)")
    ap.add_argument("--alt", default=None, help="Alternate base A/C/G/T (--seq)")
    ap.add_argument("--flank", type=int, default=30, help="Background radius in bp")
    ap.add_argument("--codon_test", action="store_true", help="Run the codon-class test")
    ap.add_argument("--fasta", default=cfg.fasta_path, help="FASTA holding --genome")
    ap.add_argument("--genome", default="m_tuberculosis_h37rv", help="Held-out record name")
    ap.add_argument("--ft", default=None, help="Feature table (default data/{genome}.ft)")
    ap.add_argument("--n_codons", type=int, default=600)
    ap.add_argument("--verdict_codons", type=int, default=100, help="Codons given full verdicts")
    ap.add_argument("--markov_k", type=int, default=5, help="Order of the k-gram baseline")
    ap.add_argument("--train_bin", default=cfg.train_bin, help="Markov baseline training data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="variants.json", help="JSON report path (--codon_test)")
    ap.add_argument("--fig", default="variant_classes.png", help="PNG path (--codon_test)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument(
        "--style_compare",
        default=None,
        help="comma-separated codon-test JSONs: plot the codon-position GC shift side by side",
    )
    ap.add_argument("--labels", default=None, help="comma-separated labels for --style_compare")
    args = ap.parse_args()

    if args.style_compare:
        from viz import render_codon_style

        paths = args.style_compare.split(",")
        labels = args.labels.split(",") if args.labels else paths
        results = []
        for path in paths:
            with open(path) as f:
                res = json.load(f)
            res["summary"] = summarize(res["rows"])  # recompute: older JSONs lack gc_shift
            results.append(res)
        print(f"  wrote {render_codon_style(results, labels, out_path=args.fig)}")
        return

    gm = GenomeModel(args.ckpt)

    if args.seq:
        if args.pos is None or args.alt is None:
            ap.error("--seq needs --pos and --alt")
        v = variant_report(
            args.seq,
            args.pos,
            args.alt,
            effects_fn=gm.variant_effects,
            score_fn=lambda s: gm.score(s)["bits_per_bp"],
            flank=args.flank,
            window=gm.cfg.block_size,
        )
        _print_report(v)
        return

    if args.codon_test:
        recs = dict(read_named_records(args.fasta))
        if args.genome not in recs:
            ap.error(f"--genome {args.genome!r} not in {args.fasta}")
        with open(args.ft or cds_path(args.genome)) as f:
            cds_list = parse_feature_table(f.read())
        train_ids = np.fromfile(args.train_bin, dtype=np.uint8)
        markov = (markov_fit(train_ids, args.markov_k), args.markov_k)
        res = codon_test(
            gm,
            recs[args.genome],
            cds_list,
            n_codons=args.n_codons,
            markov=markov,
            verdict_codons=args.verdict_codons,
            flank=args.flank,
            seed=args.seed,
        )
        res["meta"] = {
            "genome": args.genome,
            "ckpt": args.ckpt,
            "n_codons": args.n_codons,
            "verdict_codons": args.verdict_codons,
            "markov_k": args.markov_k,
            "flank": args.flank,
            "window": gm.cfg.block_size,
            "seed": args.seed,
        }
        _print_summary(res["summary"], args.markov_k)
        with open(args.out, "w") as f:
            json.dump(res, f)
        print(f"\n  wrote {args.out}")
        if not args.no_plot:
            from viz import render_variant_classes

            print(f"  wrote {render_variant_classes(res, out_path=args.fig)}")
        return

    ap.error("give --seq/--pos/--alt or --codon_test")


if __name__ == "__main__":
    main()
