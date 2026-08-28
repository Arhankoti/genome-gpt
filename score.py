"""CLI: score DNA out loud — a plain-English verdict, with the numbers behind it.

Turns the model's raw bits/bp into "does this look like real DNA?" using three
computed signals (Part 9): distance below the ~2.0 random line, whether the neural
model beats an in-sample base-composition counter on this exact sequence, and
sequence complexity (so a repetitive sequence's low score isn't mistaken for
grammar). Verdict ∈ {dna_like, plausible_composition, random_like, low_complexity}.

    python score.py --ckpt checkpoints/real.pt --seq ATGACCTGA...       # one sequence
    python score.py --ckpt checkpoints/real.pt --demo --out score_verdicts.png

--demo scores four labelled sequences drawn from a HELD-OUT val slice — the real
held-out DNA, a shuffled copy of it, a homopolymer, and random DNA — so you can see
the verdict separate real grammar from every way a low score can lie.
"""

import argparse

import numpy as np

from config import ITOS, Config
from generation import composition_shuffle, score_verdict
from inference import GenomeModel


def score_sequence(gm, seq, label=None):
    """Compose model score + composition-shuffle null + verdict into one dict."""
    neural_bits = gm.score(seq)["bits_per_bp"]
    shuffle_bits = gm.score(composition_shuffle(seq))["bits_per_bp"]
    v = score_verdict(seq, neural_bits, shuffle_bits)
    if label is not None:
        v["label"] = label
    return v


def _read_val_slice(cfg, n_bases):
    """Decode a held-out reference window from the val split, stripping | boundaries."""
    val_ids = np.fromfile(cfg.val_bin, dtype=np.uint8)[:n_bases]
    return "".join(ITOS[int(i)] for i in val_ids).replace("|", "")


def _demo_sequences(real, seed=0):
    """The real held-out slice plus three controls that each score low for the
    WRONG reason (shuffle destroys grammar; homopolymer/random have none)."""
    rng = np.random.default_rng(seed)
    n = len(real)
    shuffled = "".join(rng.permutation(list(real)))  # same composition, no grammar
    homopolymer = "A" * n
    random_dna = "".join(rng.choice(list("ACGT"), n))
    return [
        ("real held-out DNA", real),
        ("same DNA, shuffled", shuffled),
        ("homopolymer (AAAA…)", homopolymer),
        ("random DNA", random_dna),
    ]


def _print_verdict(v):
    print(f"\n  verdict: {v['verdict']}  ({v['confidence']} confidence)")
    print(f"  {v['plain_english']}")
    print(
        f"    neural {v['neural_bits_per_bp']:.4f} b/bp | "
        f"shuffle {v['shuffle_bits_per_bp']:.4f} | "
        f"grammar gain {v['grammar_gain_bits']:+.4f} | "
        f"below random {v['margin_vs_random_bits']:+.4f}"
    )
    print(
        f"    complexity {v['complexity']:.3f} | GC {v['gc_content']:.3f} | n_bases {v['n_bases']}"
    )
    for c in v["cautions"]:
        print(f"    caution: {c}")


def main():
    cfg = Config()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", default=cfg.ckpt_path, help="Checkpoint path")
    ap.add_argument("--seq", default=None, help="A single DNA sequence (ACGTN) to score")
    ap.add_argument(
        "--demo", action="store_true", help="Score real vs shuffled vs homopolymer vs random"
    )
    ap.add_argument("--demo_bases", type=int, default=1500, help="Length of each --demo sequence")
    ap.add_argument("--seed", type=int, default=0, help="Seed for --demo controls")
    ap.add_argument("--out", default="score_verdicts.png", help="Output PNG path (--demo)")
    ap.add_argument("--no-plot", action="store_true", help="Skip the PNG (--demo)")
    args = ap.parse_args()

    gm = GenomeModel(args.ckpt)

    if args.seq:
        _print_verdict(score_sequence(gm, args.seq))
        return

    if args.demo:
        real = _read_val_slice(cfg, args.demo_bases)
        reports = []
        for label, seq in _demo_sequences(real, seed=args.seed):
            v = score_sequence(gm, seq, label=label)
            reports.append(v)
            _print_verdict(v)
        if not args.no_plot:
            from viz import render_score_report

            out = render_score_report(reports, out_path=args.out)
            print(f"\n  wrote {out}")
        return

    ap.error("give --seq <DNA> or --demo")


if __name__ == "__main__":
    main()
