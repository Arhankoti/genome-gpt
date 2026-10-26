"""Tests for the Part 11 reading-frame layer.

frames.py is PURE (numbers in, frame calls out), so it's pinned by construction on
synthetic period-3 signals. The model side (GenomeModel.site_profile) is checked
only for internal consistency on the untrained tiny checkpoint — against a
hand-rolled forward pass, against saturation_scan, and for the causal-prefix
property the frame test relies on. Never assert that the untrained model finds a
frame (same discipline as Parts 2-10).
"""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from config import STOI, VOCAB_SIZE
from evaluate import markov_fit, markov_site_profile
from frames import (
    align_to_codon,
    call_frame,
    frame_accuracy,
    frame_template,
    orient,
    period3_snr,
    phase_means,
    window_shift,
)
from inference import GenomeModel


def _rand_dna(n, seed):
    return "".join(np.random.default_rng(seed).choice(list("ACGT"), n))


def _periodic(n, shift, pattern=(0.7, 0.4, 0.9), noise=0.05, seed=0):
    """Gene-oriented signal whose index j sits at codon position (shift + j) % 3."""
    rng = np.random.default_rng(seed)
    return np.array([pattern[(shift + j) % 3] for j in range(n)]) + rng.normal(0, noise, n)


@pytest.fixture(scope="module")
def gm(tiny_ckpt):
    return GenomeModel(tiny_ckpt)


# --- geometry ---


def test_orient():
    assert list(orient([1, 2, 3], "+")) == [1, 2, 3]
    assert list(orient([1, 2, 3], "-")) == [3, 2, 1]


def test_window_shift_plus_and_minus():
    plus = {"start": 100, "end": 400, "strand": "+"}
    assert window_shift(100, 130, plus) == 0
    assert window_shift(101, 131, plus) == 1
    assert window_shift(105, 135, plus) == 2
    minus = {"start": 100, "end": 400, "strand": "-"}
    # a - gene's first gene-oriented base of [lo, hi) is hi-1, gene index end-hi
    assert window_shift(370, 400, minus) == 0
    assert window_shift(369, 399, minus) == 1
    assert window_shift(368, 398, minus) == 2
    with pytest.raises(ValueError):
        window_shift(90, 120, plus)
    with pytest.raises(ValueError):
        window_shift(390, 420, minus)


def test_phase_means_ignores_nan():
    x = [1.0, 2.0, 3.0, np.nan, 5.0, 6.0]
    assert np.allclose(phase_means(x), [1.0, 3.5, 4.5])
    assert np.isnan(phase_means([1.0])[1])


# --- template + frame calling on synthetic signals ---


@pytest.mark.parametrize("shift", [0, 1, 2])
def test_call_frame_recovers_shift(shift):
    train = [(phase_means(_periodic(120, s, seed=s + 10)), s) for s in (0, 1, 2)]
    t = frame_template([v for v, _ in train], [s for _, s in train])
    got, margin = call_frame(phase_means(_periodic(90, shift, seed=99)), t)
    assert got == shift and margin > 0


def test_align_to_codon_inverts_the_shift():
    pattern = np.array([0.7, 0.4, 0.9])
    for s in range(3):
        v = phase_means(_periodic(300, s, pattern=pattern, noise=0.0))
        assert np.allclose(align_to_codon(v, s), pattern)


def test_frame_accuracy_perfect_and_chance():
    rng = np.random.default_rng(0)
    shifts = rng.integers(0, 3, 90)
    rows = [(phase_means(_periodic(60, int(s), seed=i)), int(s)) for i, s in enumerate(shifts)]
    r = frame_accuracy(rows[::2], rows[1::2])
    assert r["accuracy"] == 1.0 and r["n"] == 45 and len(r["template"]) == 3
    # no period-3 structure: accuracy near chance
    noise = [(phase_means(rng.normal(0, 1, 60)), int(s)) for s in rng.integers(0, 3, 600)]
    assert abs(frame_accuracy(noise[::2], noise[1::2])["accuracy"] - 1 / 3) < 0.1


def test_frame_template_needs_data():
    with pytest.raises(ValueError):
        frame_template([np.full(3, np.nan)], [0])


def test_call_frame_nan_is_a_coin_toss():
    assert call_frame(np.array([np.nan, 1.0, 2.0]), np.array([0.5, -1.0, 0.5])) == (0, 0.0)


def test_period3_snr():
    assert period3_snr(_periodic(300, 0, noise=0.01)) > 50
    white = np.random.default_rng(1).normal(0, 1, 3000)
    assert period3_snr(white) < 5
    assert np.isnan(period3_snr([1.0, 2.0, 3.0]))


# --- GenomeModel.site_profile (internal consistency only) ---


def test_site_profile_shapes_and_ranges(gm):
    p = gm.site_profile(_rand_dna(50, 0))
    assert p["positions"] == list(range(1, 50))
    for key in ("surprise_bits", "entropy_bits", "expected_gc", "mean_alt_llr"):
        assert len(p[key]) == 49
    assert all(0.0 <= e <= 2.0 + 1e-6 for e in p["entropy_bits"])
    assert all(0.0 <= g <= 1.0 for g in p["expected_gc"])
    assert gm.site_profile("A")["positions"] == []


@torch.no_grad()
def test_site_profile_matches_hand_rolled(gm):
    seq = _rand_dna(20, 1)  # shorter than block_size: one window
    ids = torch.tensor([STOI[c] for c in seq])
    logits, _ = gm.model(ids.unsqueeze(0))
    logp = F.log_softmax(logits[0], dim=-1)[:-1]  # row j predicts token j+1
    p = gm.site_profile(seq)
    ref = logp[torch.arange(19), ids[1:]]
    assert np.allclose(p["surprise_bits"], (-ref / np.log(2)).numpy(), atol=1e-5)
    acgt = torch.softmax(logp[:, :4], dim=-1)
    assert np.allclose(p["expected_gc"], (acgt[:, 1] + acgt[:, 2]).numpy(), atol=1e-5)


def test_site_profile_landscape_agrees_with_saturation_scan(gm):
    seq = _rand_dna(90, 2)  # > block_size: exercises the sliding window
    p = gm.site_profile(seq)
    scan = gm.saturation_scan(seq)
    by_pos = dict(zip(p["positions"], p["mean_alt_llr"]))
    for pos, row in zip(scan["positions"], scan["grid"]):
        assert by_pos[pos] == pytest.approx(sum(row) / 3, abs=1e-5)  # ref column is 0


def test_site_profile_is_causal_prefix(gm, tiny_cfg):
    seq = _rand_dna(tiny_cfg.block_size, 3)
    full = gm.site_profile(seq)
    pre = gm.site_profile(seq[:20])
    assert np.allclose(pre["expected_gc"], full["expected_gc"][:19], atol=1e-5)


# --- k-gram anticipation ---


def test_markov_site_profile():
    rng = np.random.default_rng(4)
    k = 2
    probs = markov_fit(rng.integers(0, 4, 4000).astype(np.uint8), k)
    ids = rng.integers(0, 4, 30)
    prof = markov_site_profile(probs, k, ids)
    assert np.all(np.isnan(prof["entropy_bits"][:k]))
    assert np.all((prof["entropy_bits"][k:] >= 0) & (prof["entropy_bits"][k:] <= 2.0 + 1e-9))
    ctx = ids[3] * VOCAB_SIZE + ids[4]
    p = probs[ctx, :4] / probs[ctx, :4].sum()
    assert prof["expected_gc"][5] == pytest.approx(p[1] + p[2])


# --- end-to-end frame test on the tiny model (structure only) ---


def test_frame_test_runs_end_to_end(gm):
    from landscape import SIGNALS, frame_test

    genome = _rand_dna(3000, 5)
    cds = [
        {"start": 100, "end": 1300, "strand": "+"},
        {"start": 1500, "end": 2700, "strand": "-"},
    ]
    probs = markov_fit(np.array([STOI[c] for c in genome], dtype=np.uint8), 3)
    res = frame_test(gm, genome, cds, markov=(probs, 3), n_windows=12, lengths=(30, 60))
    assert set(res["signals"]) == set(SIGNALS)
    for sig in res["signals"].values():
        for L in ("30", "60"):
            assert 0.0 <= sig["by_length"][L]["accuracy"] <= 1.0
            assert sig["by_length"][L]["n"] == 12


def test_cds_codon_positions():
    from landscape import _cds_codon_positions

    cds = [{"start": 10, "end": 40, "strand": "+"}, {"start": 50, "end": 80, "strand": "-"}]
    assert _cds_codon_positions(cds, 10, 16) == ([1, 2, 3, 1, 2, 3], "+")
    assert _cds_codon_positions(cds, 74, 80) == ([3, 2, 1, 3, 2, 1], "-")
    assert _cds_codon_positions(cds, 41, 44) == ([0, 0, 0], None)


# --- figures ---


def test_render_frame_accuracy(tmp_path):
    pytest.importorskip("matplotlib")
    from viz import render_frame_accuracy

    sig = {"by_length": {"60": {"accuracy": 0.5}, "120": {"accuracy": 0.7}}}
    res = {"lengths": [60, 120], "signals": {"neural_expected_gc": sig, "gc_frame_plot": sig}}
    out = render_frame_accuracy([res, res], ["a", "b"], out_path=str(tmp_path / "f.png"))
    assert (tmp_path / "f.png").stat().st_size > 0 and out.endswith("f.png")
    with pytest.raises(ValueError):
        render_frame_accuracy([res], ["a", "b"], out_path=str(tmp_path / "x.png"))


def test_render_gene_landscape(gm, tmp_path):
    pytest.importorskip("matplotlib")
    from viz import render_gene_landscape

    scan = gm.saturation_scan(_rand_dna(40, 6))
    n = scan["n_scored"]
    out = render_gene_landscape(
        scan, [0.5] * n, [(i % 3) + 1 for i in range(n)], out_path=str(tmp_path / "g.png")
    )
    assert (tmp_path / "g.png").stat().st_size > 0 and out.endswith("g.png")
    with pytest.raises(ValueError):
        render_gene_landscape(scan, [0.5], [1], out_path=str(tmp_path / "x.png"))
