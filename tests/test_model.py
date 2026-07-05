import pytest
import torch

from config import VOCAB_SIZE


def test_forward_no_targets(tiny_cfg, tiny_model):
    idx = torch.zeros(2, 16, dtype=torch.long)
    logits, loss = tiny_model(idx)
    assert logits.shape == (2, 16, VOCAB_SIZE)
    assert loss is None


def test_forward_with_targets(tiny_cfg, tiny_model):
    idx = torch.zeros(2, 16, dtype=torch.long)
    tgt = torch.zeros(2, 16, dtype=torch.long)
    logits, loss = tiny_model(idx, tgt)
    assert loss is not None
    assert loss.item() > 0


def test_forward_loss_is_scalar(tiny_cfg, tiny_model):
    idx = torch.zeros(1, 8, dtype=torch.long)
    tgt = torch.zeros(1, 8, dtype=torch.long)
    _, loss = tiny_model(idx, tgt)
    assert loss.ndim == 0


def test_generate_length(tiny_cfg, tiny_model):
    start = torch.zeros(1, 1, dtype=torch.long)
    out = tiny_model.generate(start, max_new_tokens=10)
    assert out.shape == (1, 11)


def test_generate_seed_preserved(tiny_cfg, tiny_model):
    start = torch.zeros(1, 1, dtype=torch.long)
    out = tiny_model.generate(start, max_new_tokens=5)
    assert out[0, 0].item() == 0


def test_generate_top_k(tiny_cfg, tiny_model):
    start = torch.zeros(1, 1, dtype=torch.long)
    out = tiny_model.generate(start, max_new_tokens=5, top_k=2)
    assert out.shape == (1, 6)


def test_block_size_exceeded_raises(tiny_cfg, tiny_model):
    idx = torch.zeros(1, tiny_cfg.block_size + 1, dtype=torch.long)
    with pytest.raises(AssertionError):
        tiny_model(idx)


def test_num_params_positive(tiny_model):
    assert tiny_model.num_params() > 0


def test_weight_tying(tiny_model):
    assert tiny_model.transformer.wte.weight is tiny_model.lm_head.weight
