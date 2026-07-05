"""Shared fixtures for the test suite.

All tests run on CPU with a tiny model (1 layer, 16 embd, 32 block_size) so
no GPU is required and the suite stays fast.
"""

import pytest
import torch

from config import Config
from model import GenomeGPT


@pytest.fixture(scope="session")
def tiny_cfg():
    cfg = Config()
    cfg.n_layer = 1
    cfg.n_head = 2
    cfg.n_embd = 16
    cfg.block_size = 32
    cfg.dropout = 0.0
    cfg.bias = False
    cfg.device = "cpu"
    return cfg


@pytest.fixture(scope="session")
def tiny_model(tiny_cfg):
    model = GenomeGPT(tiny_cfg).to(tiny_cfg.device)
    model.eval()
    return model


@pytest.fixture(scope="session")
def tiny_ckpt(tiny_cfg, tiny_model, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("ckpt") / "test.pt")
    torch.save(
        {
            "model": tiny_model.state_dict(),
            "config": tiny_cfg.to_dict(),
            "val_bits_per_bp": 2.0,
            "iter": 0,
        },
        path,
    )
    return path
