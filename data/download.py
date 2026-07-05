"""Download a diverse set of complete bacterial genomes from NCBI into one
multi-record FASTA (data/genomes.fasta). prepare.py inserts boundary tokens
between records so the model never learns cross-genome transitions.

    python data/download.py            # full curated set
    python data/download.py --single   # just E. coli (fast)

Per-genome failures are skipped with a warning, so a stale accession version
does not abort the whole download. Swap/extend GENOMES freely.
"""

import argparse
import os
import sys
import time

import requests

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
    params = {"db": "nuccore", "id": accession, "rettype": "fasta", "retmode": "text"}
    r = requests.get(EFETCH, params=params, timeout=60)
    r.raise_for_status()
    if not r.text.startswith(">"):
        raise ValueError("not FASTA")
    return r.text.strip()


def main():
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true")
    args = ap.parse_args()
    genomes = GENOMES[:1] if args.single else GENOMES

    os.makedirs(os.path.dirname(cfg.fasta_path), exist_ok=True)
    ok, total_bp = 0, 0
    with open(cfg.fasta_path, "w") as out:
        for name, acc in genomes:
            try:
                fasta = fetch(acc)
                out.write(fasta + "\n")
                bp = sum(len(line) for line in fasta.splitlines() if not line.startswith(">"))
                total_bp += bp
                ok += 1
                print(f"  ok  {name:24} {acc:14} {bp:>10,} bp")
                time.sleep(0.4)  # be polite to NCBI
            except Exception as e:
                print(f"  SKIP {name:24} {acc:14} ({e})")
    print(f"\n{ok}/{len(genomes)} genomes -> {cfg.fasta_path}  (~{total_bp:,} bp total)")


if __name__ == "__main__":
    main()
