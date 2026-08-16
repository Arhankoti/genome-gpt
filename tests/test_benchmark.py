"""Tests for benchmark.py (Part 5).

split_by_boundary is a pure function tested by value. The full benchmark() runs
on the tiny untrained checkpoint against tiny synthetic bins written to tmp_path,
so we assert report SHAPE and internal consistency only — never that the neural
net wins (it's noise on an untrained model; same discipline as Parts 2-4).
"""

import pickle

import numpy as np
import pytest

from benchmark import benchmark, split_by_boundary
from config import SEP_ID, STOI
from inference import GenomeModel


def _ids(seq):
    return np.array([STOI.get(c, STOI["N"]) for c in seq], dtype=np.uint8)


# --- split_by_boundary (pure) ---


def test_split_by_boundary_basic():
    ids = _ids("ACGT").tolist() + [SEP_ID] + _ids("TTTT").tolist()
    segs = split_by_boundary(np.array(ids, dtype=np.uint8))
    assert len(segs) == 2
    assert "".join("ACGT"[i] if i < 4 else "?" for i in segs[0]) == "ACGT"
    assert segs[1].tolist() == _ids("TTTT").tolist()


def test_split_by_boundary_drops_empty_segments():
    # leading/trailing/double boundaries never produce empty segments
    ids = np.array(
        [SEP_ID] + _ids("AC").tolist() + [SEP_ID, SEP_ID] + _ids("GT").tolist() + [SEP_ID],
        dtype=np.uint8,
    )
    segs = split_by_boundary(ids)
    assert len(segs) == 2
    assert all(len(s) > 0 for s in segs)


def test_split_by_boundary_single_segment():
    segs = split_by_boundary(_ids("ACGTACGT"))
    assert len(segs) == 1
    assert len(segs[0]) == 8


# --- full benchmark() on tiny bins + tiny ckpt ---


def _rand_dna(n, seed):
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGT"), n))


@pytest.fixture
def tiny_corpus(tmp_path):
    """Two train genomes + two held-out val genomes as bins + meta in tmp_path."""
    train = np.concatenate([_ids(_rand_dna(400, 1)), [SEP_ID], _ids(_rand_dna(400, 2))]).astype(
        np.uint8
    )
    val = np.concatenate([_ids(_rand_dna(300, 3)), [SEP_ID], _ids(_rand_dna(300, 4))]).astype(
        np.uint8
    )
    tb, vb, mp = tmp_path / "t.bin", tmp_path / "v.bin", tmp_path / "m.pkl"
    train.tofile(str(tb))
    val.tofile(str(vb))
    with open(mp, "wb") as f:
        pickle.dump(
            {
                "stoi": STOI,
                "train_genomes": ["train_a", "train_b"],
                "val_genomes": ["val_a", "val_b"],
                "split_mode": "whole_genome_holdout",
            },
            f,
        )
    return str(tb), str(vb), str(mp)


@pytest.fixture(scope="module")
def gm(tiny_ckpt):
    return GenomeModel(tiny_ckpt)


def _run(tiny_ckpt, corpus):
    tb, vb, mp = corpus
    return benchmark(tiny_ckpt, train_bin=tb, val_bin=vb, meta_path=mp, markov_orders=(0, 2, 4))


def test_benchmark_report_keys(tiny_ckpt, tiny_corpus):
    r = _run(tiny_ckpt, tiny_corpus)
    assert {
        "overall",
        "per_genome",
        "val_genomes",
        "train_genomes",
        "markov_orders",
        "note",
    } <= r.keys()
    assert {
        "neural_bits_per_bp",
        "markov",
        "best_markov_k",
        "gap_vs_best_markov",
        "gc",
        "kmer_js_bits_vs_train",
    } <= r["overall"].keys()


def test_benchmark_one_row_per_val_genome(tiny_ckpt, tiny_corpus):
    r = _run(tiny_ckpt, tiny_corpus)
    assert len(r["per_genome"]) == len(r["val_genomes"]) == 2
    assert [row["name"] for row in r["per_genome"]] == ["val_a", "val_b"]


def test_benchmark_gap_is_neural_minus_best_markov(tiny_ckpt, tiny_corpus):
    r = _run(tiny_ckpt, tiny_corpus)
    for row in r["per_genome"]:
        expected = row["neural_bits_per_bp"] - row["best_markov_bits_per_bp"]
        assert abs(row["gap_vs_best_markov"] - expected) < 1e-9
        assert row["best_markov_k"] in (0, 2, 4)
        assert row["best_markov_bits_per_bp"] == min(row["markov"].values())


def test_benchmark_is_json_serializable(tiny_ckpt, tiny_corpus):
    import json

    json.dumps(_run(tiny_ckpt, tiny_corpus))  # raises if any value isn't serializable


def test_benchmark_legacy_tail_has_no_per_genome(tiny_ckpt, tmp_path):
    # val_genomes empty (legacy tail) => per_genome gracefully empty, overall still computed
    train = _ids(_rand_dna(400, 1))
    val = _ids(_rand_dna(200, 2))
    tb, vb, mp = tmp_path / "t.bin", tmp_path / "v.bin", tmp_path / "m.pkl"
    train.tofile(str(tb))
    val.tofile(str(vb))
    with open(mp, "wb") as f:
        pickle.dump(
            {
                "stoi": STOI,
                "train_genomes": ["a"],
                "val_genomes": [],
                "split_mode": "legacy_contiguous_tail",
            },
            f,
        )
    r = benchmark(
        tiny_ckpt, train_bin=str(tb), val_bin=str(vb), meta_path=str(mp), markov_orders=(0, 2)
    )
    assert r["per_genome"] == []
    assert "neural_bits_per_bp" in r["overall"]


# --- viz smoke (guarded) ---


def test_render_benchmark_writes_png(tiny_ckpt, tiny_corpus, tmp_path):
    pytest.importorskip("matplotlib")
    from viz import render_benchmark

    r = _run(tiny_ckpt, tiny_corpus)
    out = tmp_path / "benchmark.png"
    render_benchmark(r, out_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_render_benchmark_empty_raises(tiny_ckpt):
    pytest.importorskip("matplotlib")
    from viz import render_benchmark

    with pytest.raises(ValueError):
        render_benchmark({"per_genome": []})
