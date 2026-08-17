"""Tests for data/quality.py (Part 7).

Pure, deterministic, model-free functions — tested by value. The MinHash
similarity path is the load-bearing one (it backs dedup and the leakage guard),
so its invariants are pinned tightly: self=1, reverse-complement=1 (canonical
k-mers), disjoint~0, symmetric.
"""

import numpy as np
import pytest

from data.quality import (
    clean_action,
    clean_record,
    dedup_records,
    leakage_check,
    low_complexity_fraction,
    minhash_sketch,
    n_fraction,
    qc_report,
    sketch_jaccard,
)


def _rand_dna(n, seed):
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGT"), n))


def _revcomp(s):
    c = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(c[x] for x in reversed(s))


def _mutate(seq, frac, seed):
    rng = np.random.default_rng(seed)
    m = list(seq)
    for i in rng.choice(len(m), size=int(len(m) * frac), replace=False):
        m[i] = rng.choice(list("ACGT"))
    return "".join(m)


# --- n_fraction ---


def test_n_fraction_all_acgt():
    assert n_fraction("ACGTACGT") == 0.0


def test_n_fraction_counts_n_and_ambiguity():
    assert n_fraction("NNAC") == 0.5
    assert n_fraction("ACGR") == 0.25  # R is an IUPAC ambiguity code -> non-ACGT


def test_n_fraction_empty():
    assert n_fraction("") == 0.0


# --- low_complexity_fraction ---


def test_low_complexity_homopolymer_is_high():
    assert low_complexity_fraction("A" * 200) == 1.0


def test_low_complexity_random_is_low():
    assert low_complexity_fraction(_rand_dna(400, 0)) < 0.1


# --- clean_record / clean_action ---


def test_clean_record_drops_too_short():
    assert clean_record("ACGTACGT", min_len=100) is None


def test_clean_record_drops_too_many_n():
    # interior N (won't be trimmed) at ~50% >> the 5% cap -> dropped
    seq = ("ACNN" * 100) + "ACGT"  # trailing ACGT so terminal-trim doesn't remove the N runs
    assert clean_record(seq, min_len=10, max_n_fraction=0.05) is None


def test_clean_record_trims_terminal_n_and_maps():
    out = clean_record("NNNacgtXacgtacgtacgt" + "ACGT" * 5 + "NNN", min_len=5)
    assert out is not None
    assert out[0] != "N" and out[-1] != "N"  # terminal N trimmed
    assert set(out) <= set("ACGTN")  # X mapped to N
    assert "N" in out  # the interior X became an interior N


def test_clean_record_deterministic():
    seq = "ACGT" * 2000
    assert clean_record(seq) == clean_record(seq)


def test_clean_action_labels():
    assert clean_action("ACGT", min_len=100) == "drop:too_short"
    assert clean_action("ACGT" * 2000) == "keep"


# --- minhash / sketch_jaccard ---


def test_sketch_jaccard_self_is_one():
    seq = _rand_dna(4000, 1)
    a = minhash_sketch(seq)
    assert abs(sketch_jaccard(a, a) - 1.0) < 1e-9


def test_sketch_jaccard_reverse_complement_is_one():
    # canonical k-mers => a genome and its reverse complement fingerprint the same
    seq = _rand_dna(4000, 2)
    a = minhash_sketch(seq)
    rc = minhash_sketch(_revcomp(seq))
    assert sketch_jaccard(a, rc) > 0.98


def test_sketch_jaccard_disjoint_is_near_zero():
    a = minhash_sketch(_rand_dna(4000, 3))
    b = minhash_sketch(_rand_dna(4000, 4))
    assert sketch_jaccard(a, b) < 0.05


def test_sketch_jaccard_symmetric():
    a = minhash_sketch(_rand_dna(3000, 5))
    b = minhash_sketch(_mutate(_rand_dna(3000, 5), 0.02, 6))
    assert abs(sketch_jaccard(a, b) - sketch_jaccard(b, a)) < 1e-9


def test_sketch_jaccard_bounded():
    a = minhash_sketch(_rand_dna(2000, 7))
    b = minhash_sketch(_rand_dna(2000, 8))
    assert 0.0 <= sketch_jaccard(a, b) <= 1.0


def test_near_duplicate_scores_high():
    seq = _rand_dna(5000, 9)
    near = minhash_sketch(_mutate(seq, 0.003, 10))
    assert sketch_jaccard(minhash_sketch(seq), near) > 0.9


def test_empty_sketch_jaccard_zero():
    assert sketch_jaccard(np.array([], dtype=np.uint64), minhash_sketch(_rand_dna(2000, 1))) == 0.0


# --- dedup_records ---


def test_dedup_drops_near_duplicate_keeps_distinct():
    seq = _rand_dna(5000, 11)
    named = [
        ("orig", seq),
        ("near", _mutate(seq, 0.003, 12)),
        ("distinct", _rand_dna(5000, 13)),
    ]
    kept, dropped = dedup_records(named, threshold=0.9)
    assert set(kept) == {"orig", "distinct"}
    assert len(dropped) == 1
    assert dropped[0]["name"] == "near"
    assert dropped[0]["duplicate_of"] == "orig"
    assert dropped[0]["jaccard"] >= 0.9


def test_dedup_never_empties_a_cluster():
    seq = _rand_dna(4000, 14)
    named = [("a", seq), ("b", _mutate(seq, 0.001, 15)), ("c", _mutate(seq, 0.002, 16))]
    kept, dropped = dedup_records(named, threshold=0.85)
    assert len(kept) == 1  # one representative survives the cluster
    assert len(dropped) == 2


def test_dedup_all_distinct_keeps_all():
    named = [("a", _rand_dna(3000, 17)), ("b", _rand_dna(3000, 18))]
    kept, dropped = dedup_records(named, threshold=0.9)
    assert set(kept) == {"a", "b"}
    assert dropped == []


# --- leakage_check ---


def test_leakage_flags_near_twin_in_train():
    seq = _rand_dna(5000, 19)
    train = [("t1", seq), ("t2", _rand_dna(5000, 20))]
    val = [("v", _mutate(seq, 0.003, 21))]  # near-twin of t1
    result = leakage_check(train, val, threshold=0.7)
    assert result[0]["leak"] is True
    assert result[0]["nearest_train"] == "t1"


def test_leakage_clean_when_val_is_distinct():
    train = [("t1", _rand_dna(5000, 22)), ("t2", _rand_dna(5000, 23))]
    val = [("v", _rand_dna(5000, 24))]
    result = leakage_check(train, val, threshold=0.7)
    assert result[0]["leak"] is False


# --- qc_report ---


def test_qc_report_shape_and_json():
    import json

    named = [("a", _rand_dna(4000, 25)), ("b", _rand_dna(4000, 26))]
    r = qc_report(named)
    assert {"per_genome", "similarity", "dedup", "params", "note"} <= r.keys()
    assert len(r["per_genome"]) == 2
    m = r["similarity"]["matrix"]
    assert len(m) == len(r["similarity"]["names"]) == 2  # square, matches names
    assert m[0][0] == 1.0  # diagonal is self-similarity
    json.dumps(r)  # JSON-serializable


# --- viz smoke (guarded) ---


def test_render_similarity_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from viz import render_similarity

    seq = _rand_dna(4000, 27)
    named = [("orig", seq), ("near", _mutate(seq, 0.003, 28)), ("other", _rand_dna(4000, 29))]
    report = qc_report(named)
    out = tmp_path / "sim.png"
    render_similarity(report, out_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_render_similarity_empty_raises():
    pytest.importorskip("matplotlib")
    from viz import render_similarity

    with pytest.raises(ValueError):
        render_similarity({"similarity": {"names": [], "matrix": []}})
