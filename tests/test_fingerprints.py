"""Tests for fingerprints.py (pure) and neighbors.py (via a stand-in embedder)."""

import numpy as np
import pytest

from fingerprints import (
    center,
    cosine,
    kmer_vector,
    nn_accuracy,
    probe_accuracy,
    r2,
    regress_r2,
    remove_linear,
    sample_windows,
    similarity_report,
    standardize,
    unit,
)


def test_kmer_vector_exact():
    v = kmer_vector("ACGTAC", 2)  # AC CG GT TA AC
    assert v.sum() == pytest.approx(1.0)
    assert v[0 * 4 + 1] == pytest.approx(2 / 5)  # AC
    assert v[1 * 4 + 2] == pytest.approx(1 / 5)  # CG
    assert kmer_vector("NNNN", 2).sum() == 0
    assert len(kmer_vector("ACGT", 3)) == 64


def test_kmer_skips_windows_with_n():
    v = kmer_vector("ACNGT", 2)  # only AC and GT are valid
    assert v.sum() == pytest.approx(1.0)
    assert v[1] == pytest.approx(0.5)


def test_cosine_and_unit():
    assert cosine([1, 0], [2, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 3]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 1]) == 0.0
    assert np.allclose(np.linalg.norm(unit(np.random.rand(4, 3)), axis=1), 1.0)


def test_nn_accuracy_hand_checked():
    ref = np.array([[1.0, 0], [0, 1.0]])
    q = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    assert nn_accuracy(ref, [0, 1], q, [0, 1, 1]) == pytest.approx(2 / 3)


def test_center_uses_reference_mean():
    ref = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(center(ref, np.array([[2.0, 3.0]])), 0.0)


def test_remove_linear_kills_the_covariate():
    rng = np.random.default_rng(0)
    g = rng.random(200)
    X = np.outer(g, [1.0, -2.0]) + 0.01 * rng.standard_normal((200, 2))
    Xr, _ = remove_linear(X, g, X, g)
    assert abs(np.corrcoef(Xr[:, 0], g)[0, 1]) < 0.05
    # before removal a probe recovers the covariate; after, it cannot
    assert regress_r2(X, g, X, g) > 0.9
    assert regress_r2(Xr, g, Xr, g) < 0.1


def test_r2_and_probe():
    assert r2([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert r2([1, 2, 3], [2, 2, 2]) == pytest.approx(0.0)
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    X = np.c_[y + 0.1 * rng.standard_normal(300), rng.standard_normal(300)]
    assert probe_accuracy(X[:150], y[:150], X[150:], y[150:]) > 0.95
    Xz, _ = standardize(X, X)
    assert np.allclose(Xz.mean(axis=0), 0, atol=1e-9)


def test_sample_windows_deterministic_and_bounded():
    seq = "ACGT" * 500
    a = sample_windows(seq, 5, 100, 0, 1000, seed=3)
    assert a == sample_windows(seq, 5, 100, 0, 1000, seed=3)
    assert len(a) == 5 and all(0 <= s <= 900 and len(w) == 100 for s, w in a)
    assert sample_windows("N" * 500, 5, 100, 0, 500) == []


def test_similarity_report_keys_and_self_consistency():
    def embed(s):  # stand-in: composition-only embedding
        return kmer_vector(s, 1) + 1e-3

    rng = np.random.default_rng(0)
    a = "".join(rng.choice(list("ACGT"), 300))
    r = similarity_report(embed, a, a)
    assert r["cosine_ab"] == pytest.approx(1.0, abs=1e-6)
    # a composition-only embedder cannot see order: shuffling b changes nothing
    assert abs(r["order_gain"]) < 1e-6
    assert r["gc_difference"] == 0.0
    assert set(r) >= {"cosine_a_shuffled_b", "kmer_cosine", "gc_a", "gc_b"}


def test_similarity_verdict_follows_order_gain():
    rng = np.random.default_rng(2)
    a = "".join(rng.choice(list("ACGT"), 400))

    def order_aware(s):  # dinucleotide counts see order; identical a/b beats a shuffle
        return kmer_vector(s, 3)

    r = similarity_report(order_aware, a, a)
    assert r["order_gain"] > 0 and r["verdict"] == "order_beyond_composition"
    flat = similarity_report(lambda s: [1.0, 2.0], a, a)
    assert flat["verdict"] == "no_evidence_beyond_composition"


def test_fingerprint_test_runs_with_standin_embedder():
    from neighbors import fingerprint_test

    rng = np.random.default_rng(0)
    probs = {
        "m_tuberculosis_h37rv": [0.17, 0.33, 0.33, 0.17],
        "h_pylori_26695": [0.3, 0.2, 0.2, 0.3],
        "x": [0.25] * 4,
    }
    recs = [(n, "".join(rng.choice(list("ACGT"), 4000, p=p))) for n, p in probs.items()]
    cds = {
        n: [
            {"start": 100 * i, "end": 100 * i + 150, "strand": "+-"[i % 2]} for i in range(0, 39, 2)
        ]
        for n in ("m_tuberculosis_h37rv", "h_pylori_26695")
    }

    def embed(s):
        return list(kmer_vector(s, 2))

    res = fingerprint_test(embed, recs, cds, window=100, n_per_half=5, k=3, seed=0, strand_n=8)
    assert res["n_ref"] == res["n_qry"] == 15
    for tag in ("real", "shuffled"):
        for row in res["identify"][tag].values():
            assert 0 <= row["all"] <= 1
    assert 0 < res["chance"] < 1
    assert set(res["strand"]) == {"m_tuberculosis_h37rv", "h_pylori_26695"}


def test_render_fingerprints(tmp_path):
    pytest.importorskip("matplotlib")
    from viz import render_fingerprints

    row = {"all": 0.2, "held_out": 0.2}
    names = ("gc_only", "embedding_zscored", "embedding_minus_gc", "kmer", "kmer_minus_gc")
    res = {
        "chance": 0.05,
        "identify": {"real": dict.fromkeys(names, row), "shuffled": dict.fromkeys(names, row)},
        "strand": {
            "g": {
                "plus_frac": 0.4,
                "gc": 0.5,
                "kmer": 0.8,
                "embedding": 0.8,
                "embedding_shuffled_labels": 0.5,
            }
        },
    }
    assert render_fingerprints(res, str(tmp_path / "f.png")).endswith("f.png")
    with pytest.raises(ValueError):
        render_fingerprints({"identify": {}}, str(tmp_path / "g.png"))
