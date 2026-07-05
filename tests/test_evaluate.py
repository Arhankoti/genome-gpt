"""Tests for evaluate.py — Markov baseline and statistical helpers."""

import numpy as np

from config import STOI
from evaluate import kl, kmer_freqs, markov_bits


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


# --- kmer_freqs ---


def test_kmer_freqs_sums_to_one():
    freqs = kmer_freqs("ACGTACGT", k=2)
    assert abs(sum(freqs.values()) - 1.0) < 1e-9


def test_kmer_freqs_contains_expected():
    freqs = kmer_freqs("AAAA", k=2)
    assert "AA" in freqs
    assert abs(freqs["AA"] - 1.0) < 1e-9


def test_kmer_freqs_k3():
    seq = "ACGTACGT"
    freqs = kmer_freqs(seq, k=3)
    assert all(len(k) == 3 for k in freqs)


# --- kl ---


def test_kl_self_is_zero():
    freqs = kmer_freqs("ACGTACGTACGT", k=2)
    assert kl(freqs, freqs) < 1e-9


def test_kl_asymmetric():
    # KL(p||q) != KL(q||p) for distributions with overlapping but skewed support
    p = kmer_freqs("AAAAAAAAAA" + "ACGT", k=2)
    q = kmer_freqs("ACGTACGTAC" + "AAAA", k=2)
    assert kl(p, q) != kl(q, p)


def test_kl_nonnegative():
    p = kmer_freqs(_rand_dna(200, seed=3), k=3)
    q = kmer_freqs(_rand_dna(200, seed=4), k=3)
    assert kl(p, q) >= 0
