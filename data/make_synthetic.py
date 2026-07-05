"""Generate a synthetic multi-genome corpus with learnable structure, so the
whole pipeline (boundary tokens, RC, cross-genome val, Markov baseline) can be
verified offline without NCBI.

    python data/make_synthetic.py

Each 'genome' is a FASTA record with its own motif set and GC bias, so val
(the held-out tail records) tests generalization, not recall.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MOTIF_POOL = [
    "ATGGCGCAA",
    "GGAGGTAAA",
    "TTACGCGGC",
    "CCGATGCAT",
    "AAGCTTGGC",
    "TGACGTCAT",
    "GCCAATGGC",
    "CTGCAGTAA",
]


def make_genome(rng, n, motifs, gc):
    bases = list("ACGT")
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    seq = []
    while len(seq) < n:
        if rng.random() < 0.15:
            seq.extend(list(rng.choice(motifs)))
        else:
            seq.extend(list(rng.choice(bases, size=30, p=p)))
    return "".join(seq[:n])


def main(n_genomes=6, per=60_000, seed=0, out="data/synth.fasta"):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    total = 0
    with open(out, "w") as f:
        for g in range(n_genomes):
            motifs = list(rng.choice(MOTIF_POOL, size=4, replace=False))
            gc = float(rng.uniform(0.42, 0.62))
            seq = make_genome(rng, per, motifs, gc)
            total += len(seq)
            f.write(f">synthetic_genome_{g} gc={gc:.2f}\n")
            for i in range(0, len(seq), 70):
                f.write(seq[i : i + 70] + "\n")
    print(f"wrote {out}  ({n_genomes} records, {total:,} bp)")


if __name__ == "__main__":
    main()
