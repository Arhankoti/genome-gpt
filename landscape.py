"""CLI: scan every single-base substitution across a DNA window and save a
saturation-mutagenesis landscape (ranked table to stdout + heatmap PNG).

    # score a literal sequence
    python landscape.py --seq ATG...GCA --out landscape.png

    # score a window of a FASTA record with a specific checkpoint
    python landscape.py --fasta data/genomes.fasta --record 0 \
        --start 0 --end 300 --ckpt checkpoints/synth.pt --out landscape.png

This is the fast single-site surprise measure (left context only). Confirm
individual hits with a centered variant_effect for a whole-window estimate.
"""

import argparse

from data.prepare import read_records
from inference import GenomeModel


def _print_table(scan):
    print(f"scored {scan['n_scored']} positions | {scan['note']}")
    print(f"{'rank':>4}  {'pos':>8}  {'ref':>3}  {'alt':>3}  {'llr':>10}")
    for i, h in enumerate(scan["most_disruptive"], 1):
        print(
            f"{i:>4}  {h['position']:>8}  {h['ref_base']:>3}  {h['alt_base']:>3}  {h['llr']:>10.4f}"
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--seq", help="Literal DNA sequence to scan (ACGTN)")
    src.add_argument("--fasta", help="FASTA file; use with --record")
    ap.add_argument("--record", type=int, default=0, help="Record index within --fasta (default 0)")
    ap.add_argument("--start", type=int, default=0, help="First position to score (default 0)")
    ap.add_argument(
        "--end", type=int, default=None, help="One past last position (default end of seq)"
    )
    ap.add_argument("--top-k", type=int, default=20, help="How many top hits to rank (default 20)")
    ap.add_argument("--ckpt", default=None, help="Checkpoint path (default cfg.ckpt_path)")
    ap.add_argument(
        "--out", default="landscape.png", help="Output PNG path (default landscape.png)"
    )
    ap.add_argument("--title", default=None, help="Optional figure title")
    ap.add_argument("--no-plot", action="store_true", help="Skip the PNG (print the table only)")
    args = ap.parse_args()

    if args.seq is not None:
        seq = args.seq.strip().upper()
    else:
        recs = read_records(args.fasta)
        if not (0 <= args.record < len(recs)):
            ap.error(f"--record {args.record} out of range (file has {len(recs)} records)")
        seq = recs[args.record]

    gm = GenomeModel(args.ckpt)
    scan = gm.saturation_scan(seq, start=args.start, end=args.end, top_k=args.top_k)
    _print_table(scan)

    if not args.no_plot:
        if scan["n_scored"] == 0:
            print("no scored positions — skipping plot")
            return
        from viz import render_landscape

        title = args.title or f"Saturation landscape ({scan['n_scored']} positions)"
        out = render_landscape(scan, out_path=args.out, title=title)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
