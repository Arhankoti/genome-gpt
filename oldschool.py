"""CLI: old-school biology tools, measured against the annotation (Part 12).

    # one sequence through every classic tool (no model, no checkpoint)
    python oldschool.py --seq ATG...TAA

    # the ORF test: how well does the textbook ORF finder recover a held-out genome's
    # annotated genes, at each minimum-length threshold — and how many ORFs that long
    # does a composition-matched shuffle of the genome produce by chance?
    python data/download.py --cds m_tuberculosis_h37rv
    python oldschool.py --orf_test --genome m_tuberculosis_h37rv --out orfs_mtb.json

    # side by side for two genomes (figure only)
    python oldschool.py --compare orfs_mtb.json,orfs_hp.json --labels "M. tb,H. pylori" \\
        --fig orf_calibration.png

`--orf_test` also runs the GC-skew origin test (does the cumulative-skew minimum land
on dnaA, the gene beside the replication origin?) and a restriction-site census
(observed site counts vs a composition shuffle). Nothing here loads the neural
model: these are the deterministic baselines the neural tools sit next to.

An ORF is matched to an annotated CDS by its stop codon (strand + 3' end), the
standard convention: start codons are ambiguous, stops are not. Precision is a
lower bound — `variants.parse_feature_table` keeps only plain, complete CDS, so an
ORF on a dropped (joined / partial / pseudo) gene counts as a false call.
"""

import argparse
import json

import numpy as np

from classic import (
    ENZYMES,
    find_orfs,
    gc_fraction,
    gc_report,
    gc_skew_extrema,
    gene_location,
    orf_report,
    orf_stop_key,
    site_enrichment,
    sites_report,
)
from config import Config
from data.download import cds_path
from data.prepare import read_named_records
from generation import composition_shuffle
from variants import parse_feature_table

THRESHOLDS = (30, 50, 75, 100, 150, 200, 300, 500)
_FINE = tuple(range(30, 1001, 10))  # grid for the calibrated threshold
_MAX_FALSE = 0.05  # calibrated = shortest min_codons whose chance ORFs are <= 5% of calls


def _five_prime(o):
    """+ strand coordinate of the start codon's first base (gene orientation)."""
    return o["start"] if o["strand"] == "+" else o["end"]


def _coverage(cds_list, L, strand):
    cov = np.zeros(L + 1, dtype=np.int64)
    for c in cds_list:
        if c["strand"] == strand:
            cov[c["start"]] += 1
            cov[c["end"]] -= 1
    return np.concatenate(([0], np.cumsum(np.cumsum(cov)[:L] > 0)))


def _classify_false(orfs, cds_list, L):
    """Where do false ORFs live? antisense to a gene / inside a gene on its own
    strand (another frame) / outside genes. By >= 50% overlap, antisense first."""
    cum = {s: _coverage(cds_list, L, s) for s in "+-"}
    out = {"antisense_shadow": 0, "same_strand_other_frame": 0, "intergenic": 0}
    for o in orfs:
        n = o["end"] - o["start"]
        other = "-" if o["strand"] == "+" else "+"
        if (cum[other][o["end"]] - cum[other][o["start"]]) >= 0.5 * n:
            out["antisense_shadow"] += 1
        elif (cum[o["strand"]][o["end"]] - cum[o["strand"]][o["start"]]) >= 0.5 * n:
            out["same_strand_other_frame"] += 1
        else:
            out["intergenic"] += 1
    return out


def orf_test(seq, cds_list, thresholds=THRESHOLDS, n_shuffles=2, seed=0):
    """Precision / recall of find_orfs vs annotated CDS at each min_codons, plus the
    number of ORFs that long in composition-matched shuffles (chance calls)."""
    L = len(seq)
    lo = min(min(thresholds), min(_FINE))
    orfs = find_orfs(seq, min_codons=lo)
    ann = {orf_stop_key(c): c for c in cds_list}
    ann_len = np.array([(c["end"] - c["start"]) // 3 - 1 for c in cds_list])
    hit = np.array([orf_stop_key(o) in ann for o in orfs])
    n_cod = np.array([o["n_codons"] for o in orfs])
    exact = np.array(
        [h and _five_prime(o) == _five_prime(ann[orf_stop_key(o)]) for o, h in zip(orfs, hit)],
        dtype=bool,
    )
    null = np.concatenate(
        [
            [o["n_codons"] for o in find_orfs(composition_shuffle(seq, seed=seed + i), lo)]
            for i in range(n_shuffles)
        ]
    )

    def row(t):
        sel = n_cod >= t
        n_called, n_true = int(sel.sum()), int((sel & hit).sum())
        chance = float((null >= t).sum() / n_shuffles)
        return {
            "min_codons": t,
            "n_called": n_called,
            "n_true": n_true,
            "precision": n_true / n_called if n_called else float("nan"),
            "recall": n_true / len(cds_list),
            "recall_eligible": n_true / max(int((ann_len >= t).sum()), 1),
            "start_exact": float(exact[sel & hit].mean()) if n_true else float("nan"),
            "chance_orfs": chance,
            "chance_fraction": chance / n_called if n_called else float("nan"),
        }

    fine = [row(t) for t in _FINE]
    calibrated = next((r["min_codons"] for r in fine if r["chance_fraction"] <= _MAX_FALSE), None)
    rows = [row(t) for t in thresholds]
    false_at = {}
    for t in sorted({100, calibrated or 100}):
        sel = [o for o, h in zip(orfs, hit) if not h and o["n_codons"] >= t]
        false_at[str(t)] = _classify_false(sel, cds_list, L)
    return {
        "length": L,
        "gc": gc_fraction(seq),
        "n_annotated": len(cds_list),
        "thresholds": rows,
        "curve": fine,
        "calibrated_min_codons": calibrated,
        "max_false_fraction": _MAX_FALSE,
        "false_orfs_by_location": false_at,
        "null_longest": int(null.max()) if null.size else 0,
        "n_shuffles": n_shuffles,
    }


def skew_test(seq, ft_text, n_points=2000):
    """Cumulative GC skew vs the annotated dnaA gene (which sits beside oriC)."""
    ext = gc_skew_extrema(seq)
    dnaa = gene_location(ft_text, "dnaA")
    b = np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)
    cum = np.cumsum((b == ord("G")).astype(np.int64) - (b == ord("C")).astype(np.int64))
    idx = np.linspace(0, len(seq) - 1, n_points).astype(np.int64)
    res = {**ext, "curve_pos": idx.tolist(), "curve": cum[idx].tolist(), "dnaA": dnaa}
    if dnaa is not None:
        L, o = len(seq), ext["origin_guess"]
        inside = dnaa["start"] <= o < dnaa["end"]
        circ = (min(abs(o - p), L - abs(o - p)) for p in (dnaa["start"], dnaa["end"]))
        res["origin_to_dnaA_bp"] = 0 if inside else int(min(circ))
        res["origin_to_dnaA_fraction"] = res["origin_to_dnaA_bp"] / len(seq)
    return res


def _print_orf_test(res):
    print(f"\n  {res['n_annotated']} annotated CDS, G+C {res['gc']:.3f}")
    print("  min_codons  called    true  precision  recall  start_ok  chance ORFs  chance/called")
    for r in res["thresholds"]:
        print(
            f"  {r['min_codons']:>10}  {r['n_called']:>6}  {r['n_true']:>6}  "
            f"{r['precision']:>9.3f}  {r['recall']:>6.3f}  {r['start_exact']:>8.3f}  "
            f"{r['chance_orfs']:>11.0f}"
            f"  {r['chance_fraction']:>13.3f}"
        )
    print(
        f"  calibrated min_codons (chance <= {res['max_false_fraction']:.0%} of calls): "
        f"{res['calibrated_min_codons']}; longest ORF in a shuffle: {res['null_longest']}"
    )
    for t, d in res["false_orfs_by_location"].items():
        print(f"  false ORFs >= {t} codons by location: {d}")


def main():
    cfg = Config()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seq", default=None, help="Literal DNA: run every classic tool on it")
    ap.add_argument("--orf_test", action="store_true", help="ORF / skew / site test on --genome")
    ap.add_argument("--genome", default="m_tuberculosis_h37rv", help="Held-out record name")
    ap.add_argument("--fasta", default=cfg.fasta_path, help="FASTA holding --genome")
    ap.add_argument("--n_shuffles", type=int, default=2, help="Shuffles for the ORF null")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="oldschool.json", help="JSON report path (--orf_test)")
    ap.add_argument("--fig", default="orf_calibration.png", help="ORF figure path")
    ap.add_argument("--skew_fig", default="gc_skew.png", help="GC-skew figure path")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--compare", default=None, help="comma-separated --orf_test JSONs to plot")
    ap.add_argument("--labels", default=None, help="comma-separated labels for --compare")
    args = ap.parse_args()

    if args.compare:
        from viz import render_gc_skew, render_orf_calibration

        paths = args.compare.split(",")
        labels = args.labels.split(",") if args.labels else paths
        results = []
        for path in paths:
            with open(path) as f:
                results.append(json.load(f))
        print(f"wrote {render_orf_calibration(results, labels, out_path=args.fig)}")
        print(f"wrote {render_gc_skew(results, labels, out_path=args.skew_fig)}")
        return

    if args.seq is not None:
        seq = args.seq.strip().upper()
        report = {
            "orfs": orf_report(seq, min_codons=30),
            "gc": gc_report(seq, window=min(1000, max(len(seq) // 10, 50))),
            "sites": sites_report(seq),
        }
        print(json.dumps(report, indent=2))
        return

    if not args.orf_test:
        ap.error("give --seq, --orf_test or --compare")
    recs = dict(read_named_records(args.fasta))
    if args.genome not in recs:
        ap.error(f"--genome {args.genome!r} not in {args.fasta}")
    seq = recs[args.genome].upper()
    with open(cds_path(args.genome)) as f:
        ft_text = f.read()
    cds_list = parse_feature_table(ft_text)

    res = {"genome": args.genome, "seed": args.seed}
    res["orfs"] = orf_test(seq, cds_list, n_shuffles=args.n_shuffles, seed=args.seed)
    _print_orf_test(res["orfs"])
    res["skew"] = skew_test(seq, ft_text)
    sk = res["skew"]
    print(
        f"\n  cumulative GC skew: min at {sk['origin_guess']:,}, max at "
        f"{sk['terminus_guess']:,} (separation {sk['separation']:.2f} of the genome)"
    )
    if sk.get("dnaA"):
        print(
            f"  dnaA at {sk['dnaA']['start']:,}-{sk['dnaA']['end']:,}: origin guess is "
            f"{sk['origin_to_dnaA_bp']:,} bp away ({sk['origin_to_dnaA_fraction']:.2%})"
        )
    res["sites"] = site_enrichment(seq, n_shuffles=3, seed=args.seed)
    print("\n  enzyme    site       observed  expected  obs/exp")
    for name, v in sorted(res["sites"].items(), key=lambda kv: kv[1]["ratio"] or 0):
        ratio = f"{v['ratio']:.2f}" if v["ratio"] is not None else "  -"
        print(
            f"  {name:<8}  {ENZYMES[name][0]:<9}  {v['observed']:>8}  {v['expected']:>8.0f}  "
            f"{ratio:>7}"
        )
    with open(args.out, "w") as f:
        json.dump(res, f)
    print(f"\n  wrote {args.out}")
    if not args.no_plot:
        from viz import render_gc_skew, render_orf_calibration

        print(f"  wrote {render_orf_calibration([res], [args.genome], out_path=args.fig)}")
        print(f"  wrote {render_gc_skew([res], [args.genome], out_path=args.skew_fig)}")


if __name__ == "__main__":
    main()
