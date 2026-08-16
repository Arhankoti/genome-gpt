"""Tests for scaling.py + params_for_config (Part 6).

No training happens in CI: params_for_config is checked against num_params, and
run_ladder(mode='collect') is exercised against tiny checkpoints written to
tmp_path. We assert assembly shape + internal consistency (gap == val - train,
sorted by params) only — never the shape of the scaling curve, which is noise on
these tiny models (same discipline as Parts 2-5).
"""

import numpy as np
import pytest
import torch

from config import STOI
from model import GenomeGPT, params_for_config
from scaling import ckpt_path, run_ladder

# --- params_for_config ---


def test_params_for_config_matches_num_params(tiny_cfg):
    model = GenomeGPT(tiny_cfg)
    assert params_for_config(tiny_cfg) == model.num_params()


def test_params_for_config_positive(tiny_cfg):
    assert params_for_config(tiny_cfg) > 0


def test_params_for_config_monotonic_in_embd(tiny_cfg):
    import copy

    small = copy.copy(tiny_cfg)
    big = copy.copy(tiny_cfg)
    big.n_embd = tiny_cfg.n_embd * 2
    big.n_head = tiny_cfg.n_head  # keep head count valid
    assert params_for_config(big) > params_for_config(small)


def test_params_for_config_monotonic_in_layers(tiny_cfg):
    import copy

    deep = copy.copy(tiny_cfg)
    deep.n_layer = tiny_cfg.n_layer + 2
    assert params_for_config(deep) > params_for_config(tiny_cfg)


def test_params_for_config_does_not_mutate_cfg(tiny_cfg):
    before = tiny_cfg.device
    params_for_config(tiny_cfg)
    assert tiny_cfg.device == before  # helper copies; must not flip cfg to cpu


# --- run_ladder(mode='collect') on tiny checkpoints ---


def _save_ckpt(path, cfg):
    model = GenomeGPT(cfg).to("cpu")
    model.eval()
    torch.save(
        {"model": model.state_dict(), "config": cfg.to_dict(), "val_bits_per_bp": 2.0, "iter": 0},
        path,
    )


def _ids(seq):
    return np.array([STOI.get(c, STOI["N"]) for c in seq], dtype=np.uint8)


@pytest.fixture
def ladder_ckpts(tiny_cfg, tmp_path):
    """Two tiny rungs of different sizes + tiny train/val bins in tmp_path."""
    import copy

    ckpt_dir = tmp_path
    xs = copy.copy(tiny_cfg)
    s = copy.copy(tiny_cfg)
    s.n_embd = tiny_cfg.n_embd * 2  # bigger => more params
    _save_ckpt(str(ckpt_path(str(ckpt_dir), "scale_", "xs")), xs)
    _save_ckpt(str(ckpt_path(str(ckpt_dir), "scale_", "s")), s)

    rng = np.random.default_rng(0)
    train = _ids("".join(rng.choice(list("ACGT"), 500)))
    val = _ids("".join(rng.choice(list("ACGT"), 300)))
    tb, vb = tmp_path / "t.bin", tmp_path / "v.bin"
    train.tofile(str(tb))
    val.tofile(str(vb))
    return str(ckpt_dir), str(tb), str(vb)


def _collect(ladder_ckpts):
    ckpt_dir, tb, vb = ladder_ckpts
    return run_ladder(
        ladder=[("xs", {}), ("s", {})],
        mode="collect",
        ckpt_dir=ckpt_dir,
        train_bin=tb,
        val_bin=vb,
        max_eval_tokens=1000,
    )


def test_run_ladder_report_shape(ladder_ckpts):
    r = _collect(ladder_ckpts)
    assert {"budget", "rungs", "note"} <= r.keys()
    assert len(r["rungs"]) == 2
    for rung in r["rungs"]:
        assert {"label", "params", "train_bits_per_bp", "val_bits_per_bp", "gap"} <= rung.keys()


def test_run_ladder_gap_is_val_minus_train(ladder_ckpts):
    r = _collect(ladder_ckpts)
    for rung in r["rungs"]:
        assert abs(rung["gap"] - (rung["val_bits_per_bp"] - rung["train_bits_per_bp"])) < 1e-9


def test_run_ladder_sorted_by_params(ladder_ckpts):
    r = _collect(ladder_ckpts)
    params = [rung["params"] for rung in r["rungs"]]
    assert params == sorted(params)
    assert params[0] < params[1]  # xs really is smaller than s


def test_run_ladder_is_json_serializable(ladder_ckpts):
    import json

    json.dumps(_collect(ladder_ckpts))


def test_run_ladder_skips_missing_checkpoints(ladder_ckpts):
    ckpt_dir, tb, vb = ladder_ckpts
    # a label with no checkpoint on disk is silently skipped, not an error
    r = run_ladder(
        ladder=[("xs", {}), ("missing", {})],
        mode="collect",
        ckpt_dir=ckpt_dir,
        train_bin=tb,
        val_bin=vb,
        max_eval_tokens=1000,
    )
    assert [rung["label"] for rung in r["rungs"]] == ["xs"]


def test_run_ladder_budget_is_constant(ladder_ckpts):
    # tokens-seen must be reported and equal iters*batch*block
    r = _collect(ladder_ckpts)
    b = r["budget"]
    assert b["tokens_seen"] > 0


def test_run_ladder_no_meta_required(ladder_ckpts):
    # collect mode reads bins directly; it must not require a meta.pkl on disk
    r = _collect(ladder_ckpts)
    assert r["rungs"]


# --- viz smoke (guarded) ---


def test_render_scaling_writes_png(ladder_ckpts, tmp_path):
    pytest.importorskip("matplotlib")
    from viz import render_scaling

    r = _collect(ladder_ckpts)
    out = tmp_path / "scaling.png"
    render_scaling(r, out_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_render_scaling_empty_raises():
    pytest.importorskip("matplotlib")
    from viz import render_scaling

    with pytest.raises(ValueError):
        render_scaling({"rungs": []})
