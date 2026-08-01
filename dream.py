"""CLI: let the model dream, then judge the dreams honestly.

Sweeps sampling temperature and, at each step, reports how DNA-like the output
is (k-mer fidelity) AND whether the model is producing vs. copying (novelty).
The point is the *tension*: low temperature copies, high temperature turns to
noise, and the sweet spot is where fidelity is good and copying is low.

    python dream.py --ckpt checkpoints/synth.pt                 # sweep on val reference
    python dream.py --ckpt checkpoints/synth.pt --prompt ATG --n_bases 3000 \
        --temperatures 0.5,0.7,0.9,1.1 --out dream_sweep.png

The reference defaults to a held-out slice of the val split (a genome region the
model never trained on), so fidelity isn't measured against memorized text.
"""

import argparse

import numpy as np

from config import ITOS, Config
from generation import dream_report
from inference import GenomeModel


def _read_val_reference(cfg, n_bases):
    """Decode a held-out reference window from the val split, stripping | boundaries."""
    val_ids = np.fromfile(cfg.val_bin, dtype=np.uint8)[:n_bases]
    return "".join(ITOS[int(i)] for i in val_ids).replace("|", "")


def _print_table(report):
    ref = report["reference"]
    print(
        f"\n  reference: GC={ref['gc']:.4f}  self_bits/bp={ref['self_bits_per_bp']:.4f}  "
        f"(fidelity k={report['k_fidelity']}, copy k={report['k_copy']}, "
        f"n_bases={report['n_bases']})"
    )
    print(f"  {report['note']}\n")
    hdr = f"  {'temp':>5} {'GC':>7} {'GC err':>7} {'JS bits':>8} {'copied@k':>9} {'longest':>8} {'self b/bp':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in report["sweep"]:
        print(
            f"  {r['temperature']:>5.2f} {r['gc']:>7.4f} {r['gc_abs_error']:>7.4f} "
            f"{r['kmer_js_bits']:>8.4f} {r['copied_kmer_fraction']:>9.4f} "
            f"{r['longest_exact_match']:>8d} {r['self_bits_per_bp']:>10.4f}"
        )


def main():
    cfg = Config()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", default=cfg.ckpt_path, help="Checkpoint path")
    ap.add_argument("--prompt", default="A", help="Seed DNA (ACGTN); default 'A'")
    ap.add_argument("--n_bases", type=int, default=2000, help="Bases to generate per temperature")
    ap.add_argument(
        "--temperatures",
        default="0.5,0.7,0.9,1.1",
        help="Comma-separated sampling temperatures to sweep",
    )
    ap.add_argument(
        "--ref_bases",
        type=int,
        default=8000,
        help="How many val tokens to use as the held-out reference (bounds copy-stat cost)",
    )
    ap.add_argument("--k_fidelity", type=int, default=4, help="k-mer size for fidelity (JS)")
    ap.add_argument("--k_copy", type=int, default=20, help="k-mer size for the plagiarism check")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed")
    ap.add_argument("--out", default="dream_sweep.png", help="Output PNG path")
    ap.add_argument("--no-plot", action="store_true", help="Print the table only, skip the PNG")
    args = ap.parse_args()

    temperatures = tuple(float(t) for t in args.temperatures.split(","))
    ref_seq = _read_val_reference(cfg, args.ref_bases)

    gm = GenomeModel(args.ckpt)

    def generate_fn(prompt, n_bases, temperature, seed):
        return gm.generate(
            prompt=prompt, n_bases=n_bases, temperature=temperature, top_k=4, seed=seed
        )

    def score_fn(seq):
        return gm.score(seq)["bits_per_bp"]

    report = dream_report(
        generate_fn,
        score_fn,
        ref_seq,
        prompt=args.prompt,
        n_bases=args.n_bases,
        temperatures=temperatures,
        k_fidelity=args.k_fidelity,
        k_copy=args.k_copy,
        seed=args.seed,
    )
    _print_table(report)

    if not args.no_plot:
        from viz import render_dream_sweep

        out = render_dream_sweep(report, out_path=args.out)
        print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
