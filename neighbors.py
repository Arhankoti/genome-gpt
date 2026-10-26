"""CLI: fingerprints for DNA — what does "looks alike to the model" mean? (Part 14)

    python neighbors.py --fingerprint_test --ckpt checkpoints/real.pt \\
        --out fingerprints.json --fig fingerprints.png

Windows of --window bp are embedded (mean-pooled final hidden state). Per genome,
--n_per_half windows come from its first half (the reference library) and as many
from its second half (queries), so a query never overlaps its own neighbours.

  A. Genome identification: 1-nearest-neighbour by cosine over 20 genomes (chance
     5%) for the embedding (raw, mean-centered, per-dimension z-scored), a GC-only
     fingerprint, the 4-mer spectrum, and the embedding / 4-mer with the linear GC
     component removed. Control: queries whose
     letters were shuffled (same composition, no order).
  B. How much of the embedding is G+C? Held-out R^2 of a linear probe for GC.
  C. Inside one genome: coding strand (+ vs -) of windows lying wholly inside an
     annotated CDS, by a linear probe, against GC and 4-mer baselines. Control:
     probe trained on shuffled labels. Run on the two held-out genomes.
"""

import argparse
import json

import numpy as np

from data.download import cds_path
from data.prepare import read_named_records
from fingerprints import (
    center,
    kmer_vector,
    nn_accuracy,
    probe_accuracy,
    regress_r2,
    remove_linear,
    sample_windows,
    standardize,
)
from generation import composition_shuffle, gc_content
from variants import parse_feature_table

HELD_OUT = ("m_tuberculosis_h37rv", "h_pylori_26695")


def _reps(embed, seqs, k):
    return {
        "embedding": np.array([embed(s) for s in seqs]),
        "kmer": np.array([kmer_vector(s, k) for s in seqs]),
        "gc": np.array([[gc_content(s)] for s in seqs]),
    }


def identify(ref, qry, shuf, yref, yq, names, held_out):
    """Genome-ID 1-NN accuracy per representation, for real and shuffled queries."""
    yref, yq = np.asarray(yref), np.asarray(yq)
    ho = np.isin(yq, [names.index(h) for h in held_out])
    res = {}
    for tag, q in (("real", qry), ("shuffled", shuf)):
        reps = {
            "gc_only": (ref["gc"], q["gc"]),
            "embedding": (ref["embedding"], q["embedding"]),
            "embedding_centered": (
                center(ref["embedding"], ref["embedding"]),
                center(ref["embedding"], q["embedding"]),
            ),
            "embedding_zscored": standardize(ref["embedding"], q["embedding"]),
            "kmer": (ref["kmer"], q["kmer"]),
            "kmer_centered": (
                center(ref["kmer"], ref["kmer"]),
                center(ref["kmer"], q["kmer"]),
            ),
        }
        for nm in ("embedding", "kmer"):
            reps[nm + "_minus_gc"] = remove_linear(ref[nm], ref["gc"][:, 0], q[nm], q["gc"][:, 0])
        res[tag] = {
            nm: {
                "all": nn_accuracy(R, yref, Q, yq),
                "held_out": nn_accuracy(R, yref, Q[ho], yq[ho]),
            }
            for nm, (R, Q) in reps.items()
        }
    return res


def strand_probe(embed, seq, cds, window, n, k, seed):
    """Linear-probe accuracy for coding strand; reference = first half of the genome."""
    rng = np.random.default_rng(seed)
    half = len(seq) // 2
    long = [c for c in cds if c["end"] - c["start"] >= window]
    rows = {}
    for part, sel in (
        ("ref", [c for c in long if c["end"] <= half]),
        ("qry", [c for c in long if c["start"] >= half]),
    ):
        rows[part] = []
        for i in rng.permutation(len(sel))[:n]:
            c = sel[int(i)]
            s = int(rng.integers(c["start"], c["end"] - window + 1))
            w = seq[s : s + window]
            if "N" not in w:
                rows[part].append((w, 1 if c["strand"] == "+" else 0))
    y = {p: np.array([lab for _, lab in r]) for p, r in rows.items()}
    ref = _reps(embed, [w for w, _ in rows["ref"]], k)
    qry = _reps(embed, [w for w, _ in rows["qry"]], k)
    out = {"n_ref": len(y["ref"]), "n_qry": len(y["qry"]), "plus_frac": float(y["qry"].mean())}
    for nm in ("gc", "kmer", "embedding"):
        R, Q = standardize(ref[nm], qry[nm])
        out[nm] = probe_accuracy(R, y["ref"], Q, y["qry"], lam=10.0)
    R, Q = standardize(ref["embedding"], qry["embedding"])
    out["embedding_shuffled_labels"] = probe_accuracy(
        R, rng.permutation(y["ref"]), Q, y["qry"], lam=10.0
    )
    return out


def fingerprint_test(embed, records, cds_by_genome, window, n_per_half, k, seed, strand_n):
    names = [n for n, _ in records]
    wins = {"ref": [], "qry": []}
    yref, yq = [], []
    for gi, (_, seq) in enumerate(records):
        half = len(seq) // 2
        for part, (lo, hi), ys in (("ref", (0, half), yref), ("qry", (half, len(seq)), yq)):
            ws = sample_windows(seq, n_per_half, window, lo, hi, seed=seed + gi)
            wins[part] += [w for _, w in ws]
            ys += [gi] * len(ws)
    ref = _reps(embed, wins["ref"], k)
    qry = _reps(embed, wins["qry"], k)
    shuffled = [composition_shuffle(w, seed=seed + i) for i, w in enumerate(wins["qry"])]
    shuf = _reps(embed, shuffled, k)
    res = {"names": names, "window": window, "n_ref": len(yref), "n_qry": len(yq), "k": k}
    res["chance"] = 1.0 / len(names)
    res["identify"] = identify(ref, qry, shuf, yref, yq, names, HELD_OUT)
    res["gc_r2"] = {}
    for nm in ("embedding", "kmer"):
        R, Q = standardize(ref[nm], qry[nm])
        res["gc_r2"][nm] = regress_r2(R, ref["gc"][:, 0], Q, qry["gc"][:, 0], lam=10.0)
    res["strand"] = {}
    for name, seq in records:
        if name in cds_by_genome:
            res["strand"][name] = strand_probe(
                embed, seq, cds_by_genome[name], window, strand_n, k, seed
            )
    return res


def _print(res):
    print(f"\n  genome identification, {res['window']} bp, 1-NN (chance {res['chance']:.0%})")
    print(f"  {'representation':<24}{'real':>8}{'held-out':>10}{'shuffled':>10}")
    for nm in res["identify"]["real"]:
        r, s = res["identify"]["real"][nm], res["identify"]["shuffled"][nm]
        print(f"  {nm:<24}{r['all']:>8.1%}{r['held_out']:>10.1%}{s['all']:>10.1%}")
    print("\n  held-out R^2 of GC from:", {k: round(v, 3) for k, v in res["gc_r2"].items()})
    print("\n  coding strand (+/-) linear-probe accuracy")
    for g, r in res["strand"].items():
        print(f"  {g}: " + ", ".join(f"{k}={v:.3f}" for k, v in r.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--fingerprint_test", action="store_true")
    ap.add_argument("--fasta", default="data/genomes.fasta")
    ap.add_argument("--ckpt", default="checkpoints/real.pt")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--n_per_half", type=int, default=40)
    ap.add_argument("--strand_n", type=int, default=150)
    ap.add_argument("--kmer", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="fingerprints.json")
    ap.add_argument("--fig", default="fingerprints.png")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if not args.fingerprint_test:
        ap.error("give --fingerprint_test")

    from inference import GenomeModel

    records = read_named_records(args.fasta)
    cds = {}
    for name in HELD_OUT:
        with open(cds_path(name)) as f:
            cds[name] = parse_feature_table(f.read())
    gm = GenomeModel(args.ckpt)
    res = fingerprint_test(
        gm.embed, records, cds, args.window, args.n_per_half, args.kmer, args.seed, args.strand_n
    )
    res["meta"] = {"ckpt": args.ckpt, "seed": args.seed}
    _print(res)
    with open(args.out, "w") as f:
        json.dump(res, f)
    print(f"\n  wrote {args.out}")
    if not args.no_plot:
        from viz import render_fingerprints

        print(f"  wrote {render_fingerprints(res, out_path=args.fig)}")


if __name__ == "__main__":
    main()
