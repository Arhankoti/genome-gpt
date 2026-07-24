"""Tests for GenomeModel.saturation_scan and the dna_saturation_scan bridge tool.

All run on an untrained tiny CPU checkpoint. LLRs are noise on an untrained
model, so we assert only *internal* consistency (indexing, shapes, determinism,
ref column == 0), never sign agreement with variant_effect (a different
measurement: single-site surprise vs whole-window disruption).
"""

from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from config import STOI
from inference import GenomeModel, _encode

REAL_IDS = (0, 1, 2, 3)  # A, C, G, T


@pytest.fixture(scope="module")
def gm(tiny_ckpt):
    return GenomeModel(tiny_ckpt)


# --- structure ---


def test_scan_keys(gm):
    r = gm.saturation_scan("ACGT" * 4)
    assert {
        "grid",
        "ref_bases",
        "positions",
        "worst_per_pos",
        "most_disruptive",
        "n_scored",
        "note",
    } <= r.keys()


def test_grid_shape_matches_n_scored(gm):
    r = gm.saturation_scan("ACGT" * 8)
    assert len(r["grid"]) == r["n_scored"]
    assert all(len(row) == 4 for row in r["grid"])
    assert len(r["ref_bases"]) == r["n_scored"]
    assert len(r["positions"]) == r["n_scored"]
    assert len(r["worst_per_pos"]) == r["n_scored"]


def test_ref_column_is_zero(gm):
    r = gm.saturation_scan("ACGT" * 8)
    for i, base in enumerate(r["ref_bases"]):
        ref_id = STOI[base]
        assert abs(r["grid"][i][ref_id]) < 1e-6


def test_position_zero_skipped(gm):
    r = gm.saturation_scan("ACGT" * 8)
    assert 0 not in r["positions"]
    assert r["positions"][0] == 1


# --- ranking ---


def test_most_disruptive_sorted_ascending(gm):
    r = gm.saturation_scan("ACGT" * 8, top_k=10)
    llrs = [h["llr"] for h in r["most_disruptive"]]
    assert llrs == sorted(llrs)


def test_top_k_respected(gm):
    r = gm.saturation_scan("ACGT" * 8, top_k=3)
    assert len(r["most_disruptive"]) <= 3


def test_most_disruptive_never_scores_ref_as_alt(gm):
    r = gm.saturation_scan("ACGT" * 8)
    for h in r["most_disruptive"]:
        assert h["alt_base"] != h["ref_base"]


# --- determinism ---


def test_deterministic(gm):
    r1 = gm.saturation_scan("ACGT" * 8)
    r2 = gm.saturation_scan("ACGT" * 8)
    assert r1["grid"] == r2["grid"]
    assert r1["most_disruptive"] == r2["most_disruptive"]


# --- edge cases ---


def test_single_base_returns_empty(gm):
    r = gm.saturation_scan("A")
    assert r["n_scored"] == 0
    assert r["grid"] == []
    assert r["most_disruptive"] == []


def test_empty_sequence_returns_empty(gm):
    r = gm.saturation_scan("")
    assert r["n_scored"] == 0


def test_start_end_window(gm):
    r = gm.saturation_scan("ACGT" * 8, start=5, end=10)
    assert all(5 <= p < 10 for p in r["positions"])


def test_long_sequence_slides(gm, tiny_cfg):
    long_seq = "ACGT" * (tiny_cfg.block_size + 4)
    r = gm.saturation_scan(long_seq)
    # every position past the first should be covered exactly once, in order
    assert r["n_scored"] > tiny_cfg.block_size
    assert r["positions"] == sorted(set(r["positions"]))


# --- indexing correctness (proves the off-by-one is right without a trained model) ---


def test_internal_consistency_with_direct_forward(gm):
    seq = "ACGTACGTACGT"  # short: single forward pass, no sliding window
    r = gm.saturation_scan(seq)
    ids = _encode(seq, gm.device)
    with torch.no_grad():
        logits, _ = gm.model(ids.unsqueeze(0))
        logp = F.log_softmax(logits[0], dim=-1)
    # pick a scored position in the middle and recompute its row by hand
    idx = len(r["positions"]) // 2
    p = r["positions"][idx]
    ref_id = int(ids[p].item())
    expected = [float(logp[p - 1, a] - logp[p - 1, ref_id]) for a in REAL_IDS]
    for a in REAL_IDS:
        assert abs(expected[a] - r["grid"][idx][a]) < 1e-6


# --- bridge dispatch ---


def test_bridge_dispatch_returns_bounded_payload(gm):
    with patch("bridge.tools.get_model", return_value=gm):
        from bridge.tools import dispatch

        result = dispatch("dna_saturation_scan", {"sequence": "ACGT" * 8})
    assert "most_disruptive" in result
    assert "n_scored" in result
    assert "note" in result
    assert "grid" not in result  # full grid is intentionally omitted


def test_bridge_dispatch_top_k(gm):
    with patch("bridge.tools.get_model", return_value=gm):
        from bridge.tools import dispatch

        result = dispatch("dna_saturation_scan", {"sequence": "ACGT" * 8, "top_k": 2})
    assert len(result["most_disruptive"]) <= 2


# --- viz (only if matplotlib is installed; it is a dev-only, optional dep) ---


def test_render_landscape_writes_png(gm, tmp_path):
    pytest.importorskip("matplotlib")
    from viz import render_landscape

    scan = gm.saturation_scan("ACGT" * 8)
    out = tmp_path / "landscape.png"
    render_landscape(scan, out_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_render_landscape_empty_scan_raises(gm):
    pytest.importorskip("matplotlib")
    from viz import render_landscape

    scan = gm.saturation_scan("A")  # n_scored == 0
    with pytest.raises(ValueError):
        render_landscape(scan)
