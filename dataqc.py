"""CLI: inspect a genome corpus BEFORE committing to a split.

Read-only — writes no train/val bins. Prints per-genome cleanliness (length, GC,
N%, low-complexity%) and the near-duplicate clusters, and optionally saves a JSON
report and a genome x genome MinHash-Jaccard similarity heatmap. Near-duplicate
strains show up as hot blocks: they waste capacity and, worse, can leak across
the train/held-out boundary and fake the Part 5 generalization result.

    python dataqc.py --fasta data/genomes.fasta
    python dataqc.py --fasta data/genomes.fasta --out qc.json --fig qc_similarity.png
    python dataqc.py --fasta data/genomes.fasta --holdout h_pylori_26695   # + leakage note
"""

import argparse

from data.prepare import read_named_records
from data.quality import leakage_check, qc_report


def _print_report(report, holdout=None, named=None):
    print(f"\n  {'genome':<28} {'length':>12} {'GC':>6} {'N%':>7} {'low-cplx%':>9} {'clean':>14}")
    print("  " + "-" * 80)
    for r in report["per_genome"]:
        print(
            f"  {r['name']:<28} {r['length']:>12,} {r['gc']:>6.3f} "
            f"{r['n_fraction'] * 100:>6.2f}% {r['low_complexity_fraction'] * 100:>8.2f}% "
            f"{r['clean_action']:>14}"
        )

    dropped = report["dedup"]["dropped"]
    print(f"\n  near-duplicate clusters (threshold {report['dedup']['threshold']}):")
    if dropped:
        for d in dropped:
            print(f"    {d['name']} ~ {d['duplicate_of']}  (Jaccard {d['jaccard']:.3f}) -> drop")
    else:
        print("    none — every genome is distinct")

    if holdout and named:
        holdset = {h.strip() for h in holdout.split(",") if h.strip()}
        train_named = [(n, s) for n, s in named if n not in holdset]
        val_named = [(n, s) for n, s in named if n in holdset]
        leak = leakage_check(train_named, val_named)
        print("\n  leakage check (held-out vs train):")
        for x in leak:
            flag = "  <-- LEAK" if x["leak"] else ""
            print(
                f"    {x['val']} nearest train {x['nearest_train']} "
                f"(Jaccard {x['jaccard']:.3f}){flag}"
            )
    print(f"\n  {report['note']}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fasta", required=True, help="multi-record FASTA to inspect")
    ap.add_argument(
        "--holdout", default=None, help="comma-separated held-out names for a leakage note"
    )
    ap.add_argument("--k", type=int, default=14, help="k-mer size for MinHash")
    ap.add_argument("--sketch_size", type=int, default=400, help="MinHash sketch size")
    ap.add_argument("--dedup_threshold", type=float, default=0.9)
    ap.add_argument("--out", default=None, help="write the QC report JSON here")
    ap.add_argument("--fig", default=None, help="write the similarity heatmap PNG here")
    args = ap.parse_args()

    named = read_named_records(args.fasta)
    report = qc_report(
        named, k=args.k, sketch_size=args.sketch_size, dedup_threshold=args.dedup_threshold
    )
    _print_report(report, holdout=args.holdout, named=named)

    if args.out:
        import json

        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  wrote {args.out}")
    if args.fig:
        from viz import render_similarity

        render_similarity(report, out_path=args.fig)
        print(f"  wrote {args.fig}")


if __name__ == "__main__":
    main()
