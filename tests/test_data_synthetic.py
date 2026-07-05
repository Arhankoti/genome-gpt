"""Tests for data/make_synthetic.py — synthetic genome generation."""

import os

import numpy as np

from data.make_synthetic import main, make_genome


def test_make_genome_length():
    rng = np.random.default_rng(0)
    seq = make_genome(rng, 1000, ["ATGGCG"], gc=0.5)
    assert len(seq) == 1000


def test_make_genome_valid_bases():
    rng = np.random.default_rng(1)
    seq = make_genome(rng, 500, ["ATGGCG"], gc=0.5)
    assert all(c in "ACGT" for c in seq)


def test_main_writes_fasta(tmp_path):
    out = str(tmp_path / "synth.fasta")
    main(n_genomes=2, per=200, seed=42, out=out)
    assert os.path.exists(out)
    with open(out) as f:
        content = f.read()
    assert content.count(">") == 2


def test_main_record_count(tmp_path):
    out = str(tmp_path / "synth.fasta")
    main(n_genomes=3, per=100, seed=0, out=out)
    with open(out) as f:
        headers = [line for line in f if line.startswith(">")]
    assert len(headers) == 3
