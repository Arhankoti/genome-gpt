"""CLI: scan every single-base substitution across a DNA window and save a
saturation-mutagenesis landscape (ranked table to stdout + heatmap PNG).

    # score a literal sequence
    python landscape.py --seq ATG...GCA --out landscape.png

    # score a window of a FASTA record with a specific checkpoint
    python landscape.py --fasta data/genomes.fasta --record 0 \\
        --start 0 --end 300 --ckpt checkpoints/synth.pt --out landscape.png

    # a window of a named genome, drawn against its annotated reading frame (Part 11)
    python landscape.py --genome m_tuberculosis_h37rv --start 3148440 --end 3148560 \\
        --ckpt checkpoints/real.pt --out gene_landscape.png

    # the reading-frame test: can the landscape tell you which codon position is which?
    python landscape.py --frame_test --genome m_tuberculosis_h37rv \\
        --ckpt checkpoints/real.pt --out frames_mtb.json --fig frame_accuracy.png

This is the fast single-site surprise measure (left context only). Confirm
individual hits with a centered variant_effect for a whole-window estimate.

The frame test samples windows of annotated genes at random offsets and asks each
per-position signal — the neural landscape, the neural model's *anticipation*
(expected G+C / entropy before seeing the base), a k-gram's anticipation, and the
classic GC frame plot — to call the frame, with one genome-specific template
learned on half the genes and scored on the other half (chance = 1/3).
"""

import argparse
import json

import numpy as np

from config import STOI, Config
from data.download import cds_path
from data.prepare import read_named_records, read_records
from evaluate import markov_fit, markov_site_profile
from frames import frame_accuracy, orient, period3_snr, phase_means, window_shift
from generation import composition_shuffle
from inference import GenomeModel
from variants import parse_feature_table

# signal name -> (description, uses the base at the site itself?)
SIGNALS = {
    "neural_landscape": ("neural saturation landscape (mean alt LLR)", True),
    "neural_surprise": ("neural surprise, -log2 P(ref)", True),
    "neural_expected_gc": ("neural anticipation: expected G+C", False),
    "neural_entropy": ("neural anticipation: entropy", False),
    "kgram_expected_gc": ("k-gram anticipation: expected G+C", False),
    "kgram_entropy": ("k-gram anticipation: entropy", False),
    "gc_frame_plot": ("classic GC frame plot (observed bases)", True),
    "shuffled_neural_expected_gc": ("control: neural expected G+C on a shuffled window", False),
    "shuffled_gc_frame_plot": ("control: GC frame plot on a shuffled window", True),
}


def _print_table(scan):
    print(f"scored {scan['n_scored']} positions | {scan['note']}")
    print(f"{'rank':>4}  {'pos':>8}  {'ref':>3}  {'alt':>3}  {'llr':>10}")
    for i, h in enumerate(scan["most_disruptive"], 1):
        print(
            f"{i:>4}  {h['position']:>8}  {h['ref_base']:>3}  {h['alt_base']:>3}  {h['llr']:>10.4f}"
        )


def _window_signals(gm, seq, markov):
    """Every frame-test signal over one window, as arrays indexed by window position
    (NaN where a signal has no value, e.g. position 0 / the first k for the k-gram)."""
    n = len(seq)

    def pad(values):
        out = np.full(n, np.nan)
        out[1:] = values
        return out

    prof = gm.site_profile(seq)
    shuf = composition_shuffle(seq, seed=n)
    sprof = gm.site_profile(shuf)
    probs, k = markov
    kg = markov_site_profile(probs, k, [STOI.get(c, STOI["N"]) for c in seq])
    return {
        "neural_landscape": pad(prof["mean_alt_llr"]),
        "neural_surprise": pad(prof["surprise_bits"]),
        "neural_expected_gc": pad(prof["expected_gc"]),
        "neural_entropy": pad(prof["entropy_bits"]),
        "kgram_expected_gc": kg["expected_gc"],
        "kgram_entropy": kg["entropy_bits"],
        "gc_frame_plot": np.array([c in "GC" for c in seq], dtype=np.float64),
        "shuffled_neural_expected_gc": pad(sprof["expected_gc"]),
        "shuffled_gc_frame_plot": np.array([c in "GC" for c in shuf], dtype=np.float64),
    }


def frame_test(
    gm, genome, cds_list, *, markov, n_windows=800, lengths=(60, 120, 240, 480), trim=5, seed=0
):
    """Sample gene windows, compute every signal, and score frame calling per length.

    Each window is read by the model on its own (no context outside it). Shorter
    lengths are prefixes of the same + strand window — the model is causal, so a
    prefix's profile is exactly what the model would say about that prefix alone
    (up to block_size). The first `trim` bases are dropped for every signal so the
    k-gram and the neural model are compared on positions where both have context.
    Templates are fit on even-numbered windows and tested on odd ones, then the
    other way round (2-fold); accuracy is the mean.
    """
    rng = np.random.default_rng(seed)
    W = max(lengths) + trim
    genes = [c for c in cds_list if c["end"] - c["start"] >= W]
    feats = {name: {L: [] for L in lengths} for name in SIGNALS}
    snr = {name: [] for name in SIGNALS}
    for i in range(n_windows):
        cds = genes[int(rng.integers(len(genes)))]
        a = int(rng.integers(cds["start"], cds["end"] - W + 1))
        seq = genome[a : a + W]
        if any(c not in "ACGT" for c in seq):
            continue
        sig = _window_signals(gm, seq, markov)
        for L in lengths:
            shift = window_shift(a + trim, a + trim + L, cds)
            for name, x in sig.items():
                v = phase_means(orient(x[trim : trim + L], cds["strand"]))
                feats[name][L].append((v, shift, i % 2))
        for name, x in sig.items():
            snr[name].append(period3_snr(x[trim:]))

    results = {}
    for name in SIGNALS:
        results[name] = {"by_length": {}, "period3_snr_median": float(np.nanmedian(snr[name]))}
        for L in lengths:
            rows = feats[name][L]
            folds = []
            for test_fold in (0, 1):
                train = [(v, s) for v, s, f in rows if f != test_fold]
                test = [(v, s) for v, s, f in rows if f == test_fold]
                folds.append(frame_accuracy(train, test))
            results[name]["by_length"][str(L)] = {
                "accuracy": float(np.mean([f["accuracy"] for f in folds])),
                "n": int(sum(f["n"] for f in folds)),
                "template": folds[0]["template"],
            }
    return {"signals": results, "lengths": list(lengths), "n_windows": n_windows}


def _print_frame_test(res):
    Ls = res["lengths"]
    print(f"\n  frame-calling accuracy (chance = 0.333) | {res['n_windows']} windows")
    print(f"  {'signal':<30}" + "".join(f"{str(L) + ' bp':>9}" for L in Ls) + f"{'p3 SNR':>9}")
    for name, r in res["signals"].items():
        accs = "".join(f"{r['by_length'][str(L)]['accuracy']:>9.3f}" for L in Ls)
        print(f"  {name:<30}{accs}{r['period3_snr_median']:>9.2f}")
    print("\n  template (codon pos 1/2/3, centered) at the longest length:")
    for name, r in res["signals"].items():
        t = r["by_length"][str(Ls[-1])]["template"]
        print(f"  {name:<30}" + "".join(f"{x:>+8.3f}" for x in t))


def _cds_codon_positions(cds_list, lo, hi):
    """Codon position (1/2/3, on the gene's strand) of each + position in [lo, hi);
    0 where no single annotated CDS covers it. Returns (positions, strand or None)."""
    covering = [c for c in cds_list if c["start"] <= lo and hi <= c["end"]]
    if not covering:
        return [0] * (hi - lo), None
    c = covering[0]
    if c["strand"] == "+":
        return [(p - c["start"]) % 3 + 1 for p in range(lo, hi)], "+"
    return [(c["end"] - 1 - p) % 3 + 1 for p in range(lo, hi)], "-"


def main():
    cfg = Config()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--seq", help="Literal DNA sequence to scan (ACGTN)")
    src.add_argument("--fasta", help="FASTA file; use with --record")
    src.add_argument("--genome", help="Named record in cfg.fasta_path (+ its data/{name}.ft)")
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
    ap.add_argument("--frame_test", action="store_true", help="Run the reading-frame test")
    ap.add_argument("--n_windows", type=int, default=800, help="Gene windows (--frame_test)")
    ap.add_argument("--markov_k", type=int, default=5, help="k-gram order (--frame_test)")
    ap.add_argument("--train_bin", default=cfg.train_bin, help="k-gram training data")
    ap.add_argument("--fig", default="frame_accuracy.png", help="PNG path (--frame_test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--compare",
        default=None,
        help="comma-separated --frame_test JSONs: plot accuracy vs length side by side",
    )
    ap.add_argument("--labels", default=None, help="comma-separated labels for --compare")
    args = ap.parse_args()

    if args.compare:
        from viz import render_frame_accuracy

        paths = args.compare.split(",")
        labels = args.labels.split(",") if args.labels else paths
        results = []
        for path in paths:
            with open(path) as f:
                results.append(json.load(f))
        print(f"wrote {render_frame_accuracy(results, labels, out_path=args.fig)}")
        return

    if args.seq is None and args.fasta is None and args.genome is None:
        ap.error("give one of --seq, --fasta or --genome (or --compare)")
    cds_list = None
    if args.seq is not None:
        seq = args.seq.strip().upper()
    elif args.fasta is not None:
        recs = read_records(args.fasta)
        if not (0 <= args.record < len(recs)):
            ap.error(f"--record {args.record} out of range (file has {len(recs)} records)")
        seq = recs[args.record]
    else:
        recs = dict(read_named_records(cfg.fasta_path))
        if args.genome not in recs:
            ap.error(f"--genome {args.genome!r} not in {cfg.fasta_path}")
        seq = recs[args.genome]
        with open(cds_path(args.genome)) as f:
            cds_list = parse_feature_table(f.read())

    gm = GenomeModel(args.ckpt)

    if args.frame_test:
        if cds_list is None:
            ap.error("--frame_test needs --genome (gene annotations come from data/{name}.ft)")
        train_ids = np.fromfile(args.train_bin, dtype=np.uint8)
        markov = (markov_fit(train_ids, args.markov_k), args.markov_k)
        res = frame_test(gm, seq, cds_list, markov=markov, n_windows=args.n_windows, seed=args.seed)
        res["meta"] = {
            "genome": args.genome,
            "ckpt": args.ckpt,
            "markov_k": args.markov_k,
            "seed": args.seed,
        }
        _print_frame_test(res)
        with open(args.out, "w") as f:
            json.dump(res, f)
        print(f"\n  wrote {args.out}")
        if not args.no_plot:
            from viz import render_frame_accuracy

            print(f"  wrote {render_frame_accuracy([res], [args.genome], out_path=args.fig)}")
        return

    end = len(seq) if args.end is None else args.end
    if cds_list is not None:
        # score the gene window with a full block of real left context before it
        ctx = max(0, args.start - gm.cfg.block_size)
        scan = gm.saturation_scan(
            seq[ctx:end], start=args.start - ctx, end=end - ctx, top_k=args.top_k
        )
        for h in scan["most_disruptive"]:
            h["position"] += ctx
        scan["positions"] = [p + ctx for p in scan["positions"]]
    else:
        scan = gm.saturation_scan(seq, start=args.start, end=args.end, top_k=args.top_k)
    _print_table(scan)

    if not args.no_plot:
        if scan["n_scored"] == 0:
            print("no scored positions — skipping plot")
            return
        if cds_list is not None:
            from viz import render_gene_landscape

            prof = gm.site_profile(seq[ctx:end])
            by_pos = dict(zip((p + ctx for p in prof["positions"]), prof["expected_gc"]))
            egc = [by_pos[p] for p in scan["positions"]]
            codon, strand = _cds_codon_positions(cds_list, args.start, end)
            cpos = [codon[p - args.start] for p in scan["positions"]]
            title = args.title or f"{args.genome} {args.start}-{end} (gene strand {strand})"
            out = render_gene_landscape(scan, egc, cpos, out_path=args.out, title=title)
        else:
            from viz import render_landscape

            title = args.title or f"Saturation landscape ({scan['n_scored']} positions)"
            out = render_landscape(scan, out_path=args.out, title=title)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
