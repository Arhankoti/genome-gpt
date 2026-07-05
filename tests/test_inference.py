"""Tests for inference.py — all run on an untrained tiny CPU checkpoint."""

import pytest

from config import STOI
from inference import GenomeModel


@pytest.fixture(scope="module")
def gm(tiny_ckpt):
    return GenomeModel(tiny_ckpt)


# --- score ---


def test_score_keys(gm):
    result = gm.score("ACGTACGT")
    assert {"mean_logprob", "perplexity", "bits_per_bp", "n_scored"} <= result.keys()


def test_score_bits_positive(gm):
    result = gm.score("ACGTACGT")
    assert result["bits_per_bp"] > 0


def test_score_perplexity_consistent(gm):
    import math

    result = gm.score("ACGT" * 4)
    expected_ppl = math.exp(-result["mean_logprob"])
    assert abs(expected_ppl - result["perplexity"]) < 1e-4


def test_score_sliding_window(gm, tiny_cfg):
    long_seq = "ACGT" * (tiny_cfg.block_size + 4)
    result = gm.score(long_seq)
    assert result["n_scored"] > 0


def test_score_single_base(gm):
    # single base has no previous context — n_scored should be 0 or 1
    result = gm.score("A")
    assert result["n_scored"] >= 0


# --- generate ---


def test_generate_length(gm):
    seq = gm.generate(prompt="A", n_bases=20, seed=0)
    assert len(seq) == 21  # prompt char + 20 generated


def test_generate_valid_bases(gm):
    seq = gm.generate(prompt="ACG", n_bases=10, seed=1)
    assert all(c in STOI for c in seq)


def test_generate_deterministic_with_seed(gm):
    s1 = gm.generate(prompt="A", n_bases=15, seed=99)
    s2 = gm.generate(prompt="A", n_bases=15, seed=99)
    assert s1 == s2


def test_generate_prompt_preserved(gm):
    seq = gm.generate(prompt="ACGT", n_bases=5, seed=0)
    assert seq.startswith("ACGT")


# --- variant_effect ---


def test_variant_effect_keys(gm):
    ref = "ACGT" * 8
    result = gm.variant_effect(ref, pos=4, alt_base="A")
    assert {"llr", "ref_base", "alt_base", "position", "interpretation"} <= result.keys()


def test_variant_effect_pos_out_of_range(gm):
    with pytest.raises(AssertionError):
        gm.variant_effect("ACGT", pos=10, alt_base="A")


def test_variant_effect_same_base_zero_llr(gm):
    ref = "ACGT" * 8
    # Replacing a base with itself should give LLR ≈ 0
    result = gm.variant_effect(ref, pos=4, alt_base=ref[4])
    assert abs(result["llr"]) < 1e-5


def test_variant_effect_interpretation_strings(gm):
    ref = "ACGT" * 8
    result = gm.variant_effect(ref, pos=4, alt_base="C")
    assert result["interpretation"] in ("more disruptive", "tolerated/neutral")


# --- embed ---


def test_embed_shape(gm, tiny_cfg):
    emb = gm.embed("ACGTACGT")
    assert len(emb) == tiny_cfg.n_embd


def test_embed_is_list_of_floats(gm):
    emb = gm.embed("ACGT")
    assert all(isinstance(v, float) for v in emb)


def test_embed_truncates_to_block_size(gm, tiny_cfg):
    long_seq = "ACGT" * (tiny_cfg.block_size * 2)
    emb = gm.embed(long_seq)
    assert len(emb) == tiny_cfg.n_embd
