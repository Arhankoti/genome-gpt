"""Download a diverse set of complete bacterial genomes from NCBI into one
multi-record FASTA (data/genomes.fasta). prepare.py inserts boundary tokens
between records so the model never learns cross-genome transitions.

    python data/download.py            # full curated set
    python data/download.py --single   # just E. coli (fast)

Per-genome failures are skipped with a warning, so a stale accession version
does not abort the whole download. Swap/extend GENOMES freely.
"""

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# (name, RefSeq accession) — diverse bacteria/archaea so the model learns DNA
# grammar that generalizes, not one organism. Verify accessions if a fetch fails.
GENOMES = [
    ("e_coli_k12", "NC_000913.3"),
    ("b_subtilis_168", "NC_000964.3"),
    ("salmonella_lt2", "NC_003197.2"),
    ("p_aeruginosa_pao1", "NC_002516.2"),
    ("s_aureus_nctc8325", "NC_007795.1"),
    ("m_tuberculosis_h37rv", "NC_000962.3"),
    ("h_pylori_26695", "NC_000915.1"),
    ("v_cholerae_chr1", "NC_002505.1"),
    ("s_pneumoniae_tigr4", "NC_003028.3"),
    ("l_monocytogenes_egde", "NC_003210.1"),
    ("n_meningitidis_mc58", "NC_003112.2"),
    ("l_lactis_il1403", "NC_002662.1"),
    ("synechocystis_6803", "NC_000911.1"),
    ("y_pestis_co92", "NC_003143.1"),
    ("c_jejuni_nctc11168", "NC_002163.1"),
    ("h_influenzae_rd", "NC_000907.1"),
    ("d_radiodurans_chr1", "NC_001263.1"),
    ("t_thermophilus_hb8", "NC_006461.1"),
    ("caulobacter_cb15", "NC_002696.2"),
    ("c_difficile_630", "NC_009089.1"),
]


def fetch(accession):
    import requests  # lazy: only the network path needs it, so helpers/tests don't

    params = {"db": "nuccore", "id": accession, "rettype": "fasta", "retmode": "text"}
    r = requests.get(EFETCH, params=params, timeout=60)
    r.raise_for_status()
    if not r.text.startswith(">"):
        raise ValueError("not FASTA")
    return r.text.strip()


def relabel_header(fasta, name, accession):
    """Rewrite the record's `>` header to `>{name} {accession}`.

    NCBI returns `>NC_000913.3 Escherichia coli ...`, but the whole pipeline keys
    a genome by its friendly `name` (the first header token — how prepare.py's
    --holdout and the leakage guard refer to it) and expects that name to match
    the manifest. Without this, name-based holdout silently can't find any record.
    """
    body = fasta.split("\n", 1)[1] if "\n" in fasta else ""
    return f">{name} {accession}\n{body}".rstrip()


def seq_sha1(fasta):
    """SHA1 of the uppercased sequence (headers stripped) — a stable content id
    so a corpus is reproducible and verifiable regardless of line wrapping."""
    seq = "".join(line.strip().upper() for line in fasta.splitlines() if not line.startswith(">"))
    return hashlib.sha1(seq.encode()).hexdigest(), len(seq)


def manifest_path(fasta_path):
    """Sibling manifest path: data/genomes.fasta -> data/genomes.manifest.json."""
    base, _ = os.path.splitext(fasta_path)
    return base + ".manifest.json"


def main():
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true")
    args = ap.parse_args()
    genomes = GENOMES[:1] if args.single else GENOMES

    os.makedirs(os.path.dirname(cfg.fasta_path), exist_ok=True)
    ok, total_bp, records = 0, 0, []
    with open(cfg.fasta_path, "w") as out:
        for name, acc in genomes:
            try:
                fasta = relabel_header(fetch(acc), name, acc)
                out.write(fasta + "\n")
                sha1, bp = seq_sha1(fasta)
                total_bp += bp
                ok += 1
                records.append(
                    {"name": name, "accession": acc, "length_bp": bp, "sha1": sha1, "status": "ok"}
                )
                print(f"  ok  {name:24} {acc:14} {bp:>10,} bp")
                time.sleep(0.4)  # be polite to NCBI
            except Exception as e:
                records.append(
                    {"name": name, "accession": acc, "status": "skipped", "error": str(e)}
                )
                print(f"  SKIP {name:24} {acc:14} ({e})")

    # A run you can't reproduce is an anecdote: record exactly what landed in the
    # corpus (and what was skipped) so it can be rebuilt and verified.
    mpath = manifest_path(cfg.fasta_path)
    with open(mpath, "w") as f:
        json.dump(
            {
                "fasta": cfg.fasta_path,
                "ok": ok,
                "requested": len(genomes),
                "total_bp": total_bp,
                "records": records,
            },
            f,
            indent=2,
        )
    print(f"\n{ok}/{len(genomes)} genomes -> {cfg.fasta_path}  (~{total_bp:,} bp total)")
    print(f"manifest -> {mpath}")


if __name__ == "__main__":
    main()
