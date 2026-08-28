"""Tests for the Part 9 scoring/verdict layer.

score_verdict is PURE (numbers in, verdict out), so its tiers are pinned by
construction — never against a neural model's output (LLRs are noise on the tiny
untrained checkpoint, same discipline as Parts 2-8). composition_shuffle is checked
by value (same base composition, order destroyed, deterministic).
"""

import numpy as np
import pytest

from generation import composition_shuffle, score_verdict, sequence_complexity


def _rand_dna(n, seed):
    return "".join(np.random.default_rng(seed).choice(list("ACGT"), n))


# --- sequence_complexity ---


def test_complexity_homopolymer_near_zero():
    assert sequence_complexity("A" * 500) < 0.05


def test_complexity_random_near_one():
    assert sequence_complexity(_rand_dna(2000, 0)) > 0.9


def test_complexity_bounded():
    assert 0.0 <= sequence_complexity(_rand_dna(300, 1)) <= 1.0
    assert sequence_complexity("") == 0.0


# --- composition_shuffle ---


def test_shuffle_preserves_base_composition():
    seq = _rand_dna(2000, 2)
    sh = composition_shuffle(seq)
    from collections import Counter

    assert Counter(sh) == Counter(seq)  # exact same multiset of bases
    assert len(sh) == len(seq)


def test_shuffle_deterministic_and_destroys_order():
    seq = "ACGT" * 500  # perfectly ordered
    assert composition_shuffle(seq, seed=1) == composition_shuffle(seq, seed=1)
    # a period-4 repeat has 4 distinct 3-mers; a shuffle of it should use far more
    assert sequence_complexity(composition_shuffle(seq, seed=1)) > sequence_complexity(seq)


# --- score_verdict tiers (by construction) ---


def test_verdict_low_complexity():
    v = score_verdict("A" * 1000, neural_bits=0.02, shuffle_bits=0.02)
    assert v["verdict"] == "low_complexity"
    assert any("complex" in c for c in v["cautions"])


def test_verdict_random_like():
    # neural essentially at the random line -> no usable signal
    v = score_verdict(_rand_dna(1000, 5), neural_bits=1.99, shuffle_bits=1.99)
    assert v["verdict"] == "random_like"


def test_verdict_dna_like():
    # below random AND scores well under its own composition-preserving shuffle
    v = score_verdict(_rand_dna(1000, 6), neural_bits=1.85, shuffle_bits=1.99)
    assert v["verdict"] == "dna_like"
    assert v["grammar_gain_bits"] > 0


def test_verdict_plausible_composition():
    # below random, but the shuffle scores about the same -> only composition, no order
    v = score_verdict(_rand_dna(1000, 7), neural_bits=1.90, shuffle_bits=1.905)
    assert v["verdict"] == "plausible_composition"


def test_verdict_echoes_raw_numbers():
    v = score_verdict(_rand_dna(1000, 8), neural_bits=1.88, shuffle_bits=1.99)
    assert v["neural_bits_per_bp"] == pytest.approx(1.88)
    assert v["shuffle_bits_per_bp"] == pytest.approx(1.99)
    assert v["grammar_gain_bits"] == pytest.approx(1.99 - 1.88, abs=1e-4)
    assert v["margin_vs_random_bits"] == pytest.approx(2.0 - 1.88, abs=1e-4)


def test_verdict_short_sequence_low_confidence():
    v = score_verdict(_rand_dna(60, 9), neural_bits=1.85, shuffle_bits=2.0)
    assert v["confidence"] == "low"
    assert any("short" in c for c in v["cautions"])


def test_verdict_is_deterministic():
    seq = _rand_dna(500, 10)
    assert score_verdict(seq, 1.9, 2.0) == score_verdict(seq, 1.9, 2.0)


def test_verdict_is_json_serializable():
    import json

    json.dumps(score_verdict(_rand_dna(500, 11), 1.9, 2.0))


def test_verdict_no_raw_sequence_leaks():
    v = score_verdict(_rand_dna(500, 12), 1.9, 2.0)
    assert "sequence" not in v and "seq" not in v  # bounded payload


# --- viz smoke (guarded) ---


def test_render_score_report_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from viz import render_score_report

    reports = [
        score_verdict(_rand_dna(800, 13), 1.85, 1.99) | {"label": "real"},
        score_verdict(_rand_dna(800, 14), 1.99, 1.99) | {"label": "random"},
        score_verdict("A" * 800, 0.02, 0.02) | {"label": "homopolymer"},
    ]
    out = tmp_path / "score.png"
    render_score_report(reports, out_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_render_score_report_empty_raises():
    pytest.importorskip("matplotlib")
    from viz import render_score_report

    with pytest.raises(ValueError):
        render_score_report([])
