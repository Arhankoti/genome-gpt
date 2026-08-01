"""Tests for evaluate.py — the Markov baseline.

The k-mer / divergence helpers moved to generation.py in Part 4; their tests
live in test_generation.py (kmer_spectrum / js_divergence).
"""

import numpy as np

from config import STOI
from evaluate import markov_bits


def _ids(seq):
    return np.array([STOI.get(c, STOI["N"]) for c in seq], dtype=np.uint8)


def _rand_dna(n, seed=0):
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGT"), n))


# --- markov_bits ---


def test_markov_bits_k0_near_two():
    train = _ids(_rand_dna(5000, seed=0))
    val = _ids(_rand_dna(1000, seed=1))
    bits = markov_bits(train, val, k=0)
    assert 1.5 < bits < 2.5


def test_markov_bits_positive_all_k():
    data = _ids(_rand_dna(3000, seed=2))
    train, val = data[:2000], data[2000:]
    for k in (0, 1, 2, 3):
        assert markov_bits(train, val, k) > 0


def test_markov_bits_laplace_smoothing_no_crash():
    """k=4 contexts that never appear in training shouldn't crash."""
    train = _ids("ACGT" * 100)
    val = _ids("TTTTTTTT" * 50)
    bits = markov_bits(train, val, k=4)
    assert bits > 0
