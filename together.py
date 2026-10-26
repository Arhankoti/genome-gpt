"""CLI: neural + classical, together — can anything tell a gene from its shadow? (Part 13)

    # the shadow test on a held-out genome (Part 12's calibrated ORF cutoff)
    python together.py --shadow_test --genome m_tuberculosis_h37rv --min_codons 280 \\
        --ckpt checkpoints/real.pt --out shadows_mtb.json --no-plot
    python together.py --shadow_test --genome h_pylori_26695 --min_codons 110 \\
        --ckpt checkpoints/real.pt --out shadows_hp.json --no-plot
    python together.py --compare shadows_mtb.json,shadows_hp.json \\
        --labels "M. tb,H. pylori" --fig shadow_test.png

Every ORF of at least --min_codons in the genome (the "pool") is labelled from the
annotation: gene (its stop matches an annotated CDS), shadow (not a gene, and >= 50%
of it lies antisense to an annotated CDS), or other. Each ORF is read in its own
frame and orientation, from its start codon, for the first 60 / 120 / 240 / 480 bp,
and summarized by every signal the same way (orfjudge.py):

  classic : codon usage; observed base composition by codon position; the GC-frame
            rule (GC3 - max(GC1, GC2), label-free)
  neural  : the model's anticipation P(base | left context) by codon position
  k-gram  : the order-k counter's anticipation, the same way (the usual control)
  together: the sum of the z-scored codon-usage and neural scores

Two settings: TRAINED (gene and shadow templates from the annotation, 2-fold) and
SELF-TRAINED (no annotation: similarity to the pool's own centroid, refined over
overlapping antisense pairs). Reported: AUC of genes vs shadows (0.5 = chance) and
pair accuracy (for overlapping gene/shadow pairs, how often the gene wins).
"""

import argparse
import json

import numpy as np

from classic import find_orfs, orf_stop_key
from config import STOI, Config
from data.download import cds_path
from data.prepare import read_named_records
from evaluate import markov_base_probs, markov_fit
from orfjudge import (
    antisense_pairs,
    auc_higher,
    codon_usage,
    gc_frame_rule,
    onehot,
    oriented,
    pair_accuracy,
    phase_profile,
    self_trained_scores,
    trained_scores,
    zscore,
)
from variants import parse_feature_table

LENGTHS = (60, 120, 240, 480)
CTX = 60  # upstream bases fed to the models so the start codon has left context

SIGNALS = {
    "codon_usage": "classic: codon usage (64)",
    "observed_phase": "classic: base composition by codon position (12)",
    "neural_phase": "neural: anticipation by codon position (12)",
    "kgram_phase": "k-gram: anticipation by codon position (12)",
    "together": "together: codon usage + neural",
}
LABEL_FREE = {"gc_frame_rule": "classic: GC-frame rule, GC3 - max(GC1, GC2)"}


def label_pool(seq, cds_list, orfs):
    """1 = gene, 0 = antisense shadow, -1 = other."""
    ann = {orf_stop_key(c) for c in cds_list}
    L = len(seq)
    cov = {s: np.zeros(L + 1, dtype=np.int64) for s in "+-"}
    for c in cds_list:
        cov[c["strand"]][c["start"]] += 1
        cov[c["strand"]][c["end"]] -= 1
    cum = {s: np.concatenate(([0], np.cumsum(np.cumsum(v)[:L] > 0))) for s, v in cov.items()}
    labels = []
    for o in orfs:
        if orf_stop_key(o) in ann:
            labels.append(1)
            continue
        other = "-" if o["strand"] == "+" else "+"
        n = o["end"] - o["start"]
        anti = cum[other][o["end"]] - cum[other][o["start"]]
        labels.append(0 if anti >= 0.5 * n else -1)
    return np.array(labels)


def orf_features(seq, orfs, base_probs_fn, kgram_fn, lengths=LENGTHS):
    """{length: {signal: (n_orfs, d) array}} for every ORF. One model pass per ORF
    (CTX upstream + the longest window); shorter windows are causal prefixes of it."""
    Lmax = max(lengths)
    feats = {
        L: {k: [] for k in ("codon_usage", "observed_phase", "neural_phase", "kgram_phase")}
        for L in lengths
    }
    for o in orfs:
        s, off = oriented(seq, o, CTX)
        body = len(s) - off - 3  # the ORF without its stop codon
        s = s[: off + min(Lmax, body)]
        pn = base_probs_fn(s)[off:]
        pk = kgram_fn(s)[off:]
        obs = onehot(s[off:])
        for L in lengths:
            n = min(L, body)
            n -= n % 3
            f = feats[L]
            f["codon_usage"].append(codon_usage(s[off : off + n]))
            f["observed_phase"].append(phase_profile(obs[:n]))
            f["neural_phase"].append(phase_profile(pn[:n]))
            f["kgram_phase"].append(phase_profile(pk[:n]))
    return {L: {k: np.array(v) for k, v in f.items()} for L, f in feats.items()}


def agreement(classic, neural, pairs, y):
    """Over gene/shadow pairs: how often the two self-trained judges pick the same
    ORF, and how often the pick is right when they agree vs when they don't."""
    agree_hits, disagree = [], 0
    for i, j in pairs:
        if {y[i], y[j]} != {0, 1}:
            continue
        g, sh = (i, j) if y[i] == 1 else (j, i)
        c_ok, n_ok = classic[g] > classic[sh], neural[g] > neural[sh]
        if c_ok == n_ok:
            agree_hits.append(c_ok)
        else:
            disagree += 1
    n = len(agree_hits) + disagree
    return {
        "n_pairs": n,
        "agree_rate": len(agree_hits) / n if n else float("nan"),
        "accuracy_when_agree": float(np.mean(agree_hits)) if agree_hits else float("nan"),
        "n_disagree": disagree,
    }


def shadow_test(
    seq, cds_list, base_probs_fn, kgram_fn, *, min_codons, lengths=LENGTHS, max_orfs=None, seed=0
):
    orfs = find_orfs(seq, min_codons=min_codons)
    if max_orfs and len(orfs) > max_orfs:
        idx = np.sort(np.random.default_rng(seed).permutation(len(orfs))[:max_orfs])
        orfs = [orfs[i] for i in idx]
    labels = label_pool(seq, cds_list, orfs)
    pairs = antisense_pairs(orfs)
    gs = labels >= 0  # genes + shadows: the trained setting and every AUC
    feats = orf_features(seq, orfs, base_probs_fn, kgram_fn, lengths)
    res = {
        "n_pool": len(orfs),
        "n_genes": int((labels == 1).sum()),
        "n_shadows": int((labels == 0).sum()),
        "n_other": int((labels == -1).sum()),
        "n_pairs": len(pairs),
        "n_gene_shadow_pairs": sum({labels[i], labels[j]} == {0, 1} for i, j in pairs),
        "min_codons": min_codons,
        "lengths": list(lengths),
        "signals": {},
    }
    y = labels[gs]
    gene_shadow_pairs = [(i, j) for i, j in pairs if gs[i] and gs[j]]
    remap = np.cumsum(gs) - 1  # pool index -> index among genes + shadows
    gpairs = [(remap[i], remap[j]) for i, j in gene_shadow_pairs]

    def record(name, L, trained, selft):
        sig = res["signals"].setdefault(name, {})
        row = {
            "self_auc": auc_higher(selft[gs][y == 1], selft[gs][y == 0]),
            "self_pair_accuracy": pair_accuracy(selft[gs], gpairs, y),
        }
        if trained is not None:
            row["trained_auc"] = auc_higher(trained[y == 1], trained[y == 0])
            row["trained_pair_accuracy"] = pair_accuracy(trained, gpairs, y)
        sig[str(L)] = row

    for L in lengths:
        tr_scores, st_scores = {}, {}
        for k in ("codon_usage", "observed_phase", "neural_phase", "kgram_phase"):
            X = feats[L][k]
            tr_scores[k] = trained_scores(X[gs], y, seed=seed)
            st_scores[k] = self_trained_scores(X, pairs)
            record(k, L, tr_scores[k], st_scores[k])
        record(
            "together",
            L,
            zscore(tr_scores["codon_usage"]) + zscore(tr_scores["neural_phase"]),
            zscore(st_scores["codon_usage"]) + zscore(st_scores["neural_phase"]),
        )
        res.setdefault("agreement", {})[str(L)] = agreement(
            st_scores["codon_usage"][gs], st_scores["neural_phase"][gs], gpairs, y
        )
        rule = np.array([gc_frame_rule(v) for v in feats[L]["observed_phase"]])
        record("gc_frame_rule", L, None, rule)
    return res


def _print_agreement(res):
    print("\n  SELF-TRAINED codon usage vs neural, over gene/shadow pairs")
    for L in res["lengths"]:
        a = res["agreement"][str(L)]
        print(
            f"  {L:>4} bp: agree on {a['agree_rate']:.0%} of {a['n_pairs']} pairs, right "
            f"{a['accuracy_when_agree']:.0%} of the time when they agree"
        )


def _print(res):
    print(
        f"\n  pool: {res['n_pool']} ORFs >= {res['min_codons']} codons — {res['n_genes']} genes, "
        f"{res['n_shadows']} shadows, {res['n_other']} other; "
        f"{res['n_gene_shadow_pairs']} overlapping gene/shadow pairs"
    )
    for setting, key in (
        ("TRAINED", "trained_auc"),
        ("SELF-TRAINED", "self_auc"),
        ("SELF-TRAINED pairs", "self_pair_accuracy"),
    ):
        print(f"\n  {setting} ({key}; 0.5 = chance)")
        print("  " + " " * 52 + "".join(f"{L:>8}" for L in res["lengths"]))
        for name, desc in {**SIGNALS, **LABEL_FREE}.items():
            sig = res["signals"][name]
            vals = [sig[str(L)].get(key) for L in res["lengths"]]
            if all(v is None for v in vals):
                continue
            print(
                f"  {desc:<52}"
                + "".join(f"{v:>8.3f}" if v is not None else f"{'-':>8}" for v in vals)
            )


def main():
    cfg = Config()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--shadow_test", action="store_true")
    ap.add_argument("--genome", default="m_tuberculosis_h37rv", help="Held-out record name")
    ap.add_argument("--fasta", default=cfg.fasta_path)
    ap.add_argument("--ckpt", default=cfg.ckpt_path)
    ap.add_argument("--min_codons", type=int, default=100, help="ORF pool cutoff (Part 12)")
    ap.add_argument("--max_orfs", type=int, default=None, help="Subsample the pool")
    ap.add_argument("--markov_k", type=int, default=5)
    ap.add_argument("--train_bin", default=cfg.train_bin)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="shadows.json")
    ap.add_argument("--fig", default="shadow_test.png")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--compare", default=None, help="comma-separated --shadow_test JSONs")
    ap.add_argument("--labels", default=None)
    args = ap.parse_args()

    if args.compare:
        from viz import render_shadow_test

        paths = args.compare.split(",")
        labels = args.labels.split(",") if args.labels else paths
        results = []
        for path in paths:
            with open(path) as f:
                results.append(json.load(f))
        print(f"wrote {render_shadow_test(results, labels, out_path=args.fig)}")
        return
    if not args.shadow_test:
        ap.error("give --shadow_test or --compare")

    from inference import GenomeModel

    recs = dict(read_named_records(args.fasta))
    if args.genome not in recs:
        ap.error(f"--genome {args.genome!r} not in {args.fasta}")
    seq = recs[args.genome].upper()
    with open(cds_path(args.genome)) as f:
        cds_list = parse_feature_table(f.read())
    gm = GenomeModel(args.ckpt)
    probs = markov_fit(np.fromfile(args.train_bin, dtype=np.uint8), args.markov_k)

    def kgram(s):
        return markov_base_probs(probs, args.markov_k, [STOI.get(c, STOI["N"]) for c in s])

    res = shadow_test(
        seq,
        cds_list,
        gm.base_probs,
        kgram,
        min_codons=args.min_codons,
        max_orfs=args.max_orfs,
        seed=args.seed,
    )
    res["meta"] = {
        "genome": args.genome,
        "ckpt": args.ckpt,
        "markov_k": args.markov_k,
        "seed": args.seed,
    }
    _print(res)
    _print_agreement(res)
    with open(args.out, "w") as f:
        json.dump(res, f)
    print(f"\n  wrote {args.out}")
    if not args.no_plot:
        from viz import render_shadow_test

        print(f"  wrote {render_shadow_test([res], [args.genome], out_path=args.fig)}")


if __name__ == "__main__":
    main()
