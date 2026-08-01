"""Tests for generation.py (pure stats) and the dna_generation_report bridge tool.

The pure functions are deterministic and model-free, so they are tested by value.
dream_report is tested for shape/keys/JSON-serializability only — on the untrained
tiny checkpoint the fidelity/novelty *values* are noise (same discipline as
Parts 2-3), so we never assert quality.
"""

from unittest.mock import patch

import numpy as np
import pytest

from generation import (
    copy_stats,
    dream_report,
    gc_content,
    js_divergence,
    kmer_spectrum,
)


def _rand_dna(n, seed=0):
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGT"), n))


# --- gc_content ---


def test_gc_content_all_gc():
    assert gc_content("GGCC") == 1.0


def test_gc_content_no_gc():
    assert gc_content("ATAT") == 0.0


def test_gc_content_ignores_n_and_boundary():
    # N and | are not counted in numerator or denominator
    assert gc_content("GGCCNN||") == 1.0
    assert gc_content("ATGC") == 0.5


def test_gc_content_empty_is_zero():
    assert gc_content("NNN|||") == 0.0


# --- kmer_spectrum ---


def test_kmer_spectrum_sums_to_one():
    spec = kmer_spectrum("ACGTACGT", k=2)
    assert abs(sum(spec.values()) - 1.0) < 1e-9


def test_kmer_spectrum_skips_n_and_boundary():
    spec = kmer_spectrum("ACGN|TACG", k=2)
    assert spec  # non-empty
    assert all("N" not in km and "|" not in km for km in spec)


def test_kmer_spectrum_counts_hand_example():
    # "AAAA" has three "AA" windows; all identical -> {"AA": 1.0}
    spec = kmer_spectrum("AAAA", k=2)
    assert set(spec) == {"AA"}
    assert abs(spec["AA"] - 1.0) < 1e-9


def test_kmer_spectrum_empty_when_no_valid_window():
    assert kmer_spectrum("N|N|N", k=3) == {}


# --- js_divergence ---


def test_js_divergence_self_is_zero():
    p = kmer_spectrum("ACGTACGTACGT", k=2)
    assert abs(js_divergence(p, p)) < 1e-12


def test_js_divergence_symmetric():
    p = kmer_spectrum("AAAAACGT", k=2)
    q = kmer_spectrum("ACGTTTTT", k=2)
    assert abs(js_divergence(p, q) - js_divergence(q, p)) < 1e-12


def test_js_divergence_bounded_in_unit_interval():
    p = kmer_spectrum(_rand_dna(300, seed=1), k=3)
    q = kmer_spectrum(_rand_dna(300, seed=2), k=3)
    assert 0.0 <= js_divergence(p, q) <= 1.0


def test_js_divergence_disjoint_support_is_one():
    # No shared k-mers -> maximal divergence of exactly 1 bit
    p = {"AA": 1.0}
    q = {"TT": 1.0}
    assert abs(js_divergence(p, q) - 1.0) < 1e-9


# --- copy_stats ---


def test_copy_stats_full_copy():
    ref = "ACGTACGTACGTACGTACGTACGT"
    cs = copy_stats(ref, ref, k=10)
    assert cs["copied_kmer_fraction"] == 1.0
    assert cs["longest_exact_match"] == len(ref)


def test_copy_stats_disjoint_is_zero():
    cs = copy_stats("A" * 50, "C" * 50, k=10)
    assert cs["copied_kmer_fraction"] == 0.0
    assert cs["longest_exact_match"] == 0


def test_copy_stats_partial_copy_longest_match():
    ref = "ACGTACGTACGTACGT"
    gen = "TTTT" + ref[:8] + "GGGG"  # an 8-long verbatim run embedded in novel flanks
    cs = copy_stats(gen, ref, k=4)
    assert cs["longest_exact_match"] >= 8
    assert 0.0 < cs["copied_kmer_fraction"] < 1.0


def test_copy_stats_short_inputs_no_crash():
    cs = copy_stats("ACG", "ACGT", k=20)  # shorter than k
    assert cs["copied_kmer_fraction"] == 0.0


# --- dream_report (shape only; stub closures, no model) ---


def _stub_generate(prompt, n_bases, temperature, seed):
    # deterministic pseudo-DNA depending on temperature so rows differ
    bases = "ACGT"
    return "".join(bases[(i + int(temperature * 10)) % 4] for i in range(n_bases))


def _stub_score(seq):
    return 1.95


def test_dream_report_keys_and_rows():
    ref = _rand_dna(500, seed=7)
    rep = dream_report(_stub_generate, _stub_score, ref, n_bases=200, temperatures=(0.5, 0.9, 1.1))
    assert {"reference", "sweep", "k_fidelity", "k_copy", "n_bases", "note"} <= rep.keys()
    assert {"gc", "self_bits_per_bp"} <= rep["reference"].keys()
    assert len(rep["sweep"]) == 3
    for row in rep["sweep"]:
        assert {
            "temperature",
            "gc",
            "gc_abs_error",
            "kmer_js_bits",
            "copied_kmer_fraction",
            "longest_exact_match",
            "self_bits_per_bp",
        } <= row.keys()


def test_dream_report_is_json_serializable():
    import json

    ref = _rand_dna(300, seed=8)
    rep = dream_report(_stub_generate, _stub_score, ref, n_bases=150, temperatures=(0.7, 1.0))
    json.dumps(rep)  # raises if any value is not JSON-serializable


def test_dream_report_temperatures_preserved():
    ref = _rand_dna(200, seed=9)
    rep = dream_report(_stub_generate, _stub_score, ref, n_bases=120, temperatures=(0.6, 1.2))
    assert [r["temperature"] for r in rep["sweep"]] == [0.6, 1.2]


# --- bridge dispatch (real tiny model via gm fixture) ---


@pytest.fixture(scope="module")
def gm(tiny_ckpt):
    from inference import GenomeModel

    return GenomeModel(tiny_ckpt)


def test_bridge_dispatch_generation_report_scalar_payload(gm):
    with patch("bridge.tools.get_model", return_value=gm):
        from bridge.tools import dispatch

        result = dispatch(
            "dna_generation_report", {"reference": "ACGT" * 20, "n_bases": 60, "temperature": 0.9}
        )
    expected = {
        "kmer_js_bits",
        "gc_generated",
        "gc_reference",
        "copied_kmer_fraction",
        "longest_exact_match",
        "self_bits_per_bp",
        "note",
    }
    assert expected <= result.keys()
    # bounded payload: never dump the generated sequence
    assert "sequence" not in result
    assert "gen" not in result


# --- viz (only if matplotlib is installed; dev-only, optional dep) ---


def test_render_dream_sweep_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from viz import render_dream_sweep

    ref = _rand_dna(400, seed=11)
    rep = dream_report(_stub_generate, _stub_score, ref, n_bases=200, temperatures=(0.5, 0.9, 1.1))
    out = tmp_path / "dream_sweep.png"
    render_dream_sweep(rep, out_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_render_dream_sweep_empty_raises():
    pytest.importorskip("matplotlib")
    from viz import render_dream_sweep

    with pytest.raises(ValueError):
        render_dream_sweep({"sweep": []})
