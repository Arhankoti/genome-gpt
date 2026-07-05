"""Train the genome GPT.

    python train.py                       # full run (uses config defaults)
    python train.py --max_iters 300 --n_layer 2 --n_embd 64 --block_size 128 \
                    --batch_size 16 --eval_interval 100 --device cpu \
                    --train_bin data/smoke_train.bin --val_bin data/smoke_val.bin
"""

import argparse
import math
import os
import time

import numpy as np
import torch

from config import COMPLEMENT, ITOS, LN2, Config
from model import GenomeGPT


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_batch(bin_path, block_size, batch_size, device, rc_prob=0.0):
    data = np.memmap(bin_path, dtype=np.uint8, mode="r")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    chunks = torch.stack(
        [torch.from_numpy(data[i : i + block_size + 1].astype(np.int64)) for i in ix]
    )
    if rc_prob > 0:
        comp = torch.tensor(COMPLEMENT, dtype=torch.long)
        mask = torch.rand(batch_size) < rc_prob
        if mask.any():
            # reverse-complement = complement(reverse(window)); a valid window
            # on the opposite strand, so next-token prediction stays well-formed
            chunks[mask] = comp[torch.flip(chunks[mask], dims=[1])]
    x, y = chunks[:, :-1].contiguous(), chunks[:, 1:].contiguous()
    if device == "cuda":
        x, y = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, cfg):
    model.eval()
    out = {}
    for split, path in (("train", cfg.train_bin), ("val", cfg.val_bin)):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(path, cfg.block_size, cfg.batch_size, cfg.device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def lr_at(it, cfg):
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    if it > cfg.max_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def sample(model, cfg, n=120):
    model.eval()
    start = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)  # 'A'
    out = model.generate(start, n, temperature=0.8, top_k=4)[0].tolist()
    model.train()
    return "".join(ITOS[i] for i in out)


def parse_overrides():
    cfg = Config()
    ap = argparse.ArgumentParser()
    for k, v in cfg.to_dict().items():
        ap.add_argument(f"--{k}", type=type(v) if not isinstance(v, bool) else int, default=None)
    for k, v in vars(ap.parse_args()).items():
        if v is not None:
            setattr(cfg, k, type(getattr(cfg, k))(v) if isinstance(getattr(cfg, k), bool) else v)
    return cfg


def main():
    cfg = parse_overrides()
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    model = GenomeGPT(cfg).to(cfg.device)
    print(f"device={cfg.device}  non-embedding params={model.num_params() / 1e6:.2f}M")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    t0 = time.time()
    best_val = float("inf")
    for it in range(cfg.max_iters + 1):
        for g in optim.param_groups:
            g["lr"] = lr_at(it, cfg)

        if it % cfg.eval_interval == 0:
            losses = estimate_loss(model, cfg)
            tb, vb = losses["train"] / LN2, losses["val"] / LN2
            print(
                f"iter {it:>6} | train {tb:.4f} bits/bp | val {vb:.4f} bits/bp "
                f"| {time.time() - t0:.0f}s"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                os.makedirs(os.path.dirname(cfg.ckpt_path) or ".", exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": cfg.to_dict(),
                        "val_bits_per_bp": vb,
                        "iter": it,
                    },
                    cfg.ckpt_path,
                )

        if it > 0 and it % cfg.sample_interval == 0:
            print("  sample:", sample(model, cfg))

        if it == cfg.max_iters:
            break

        optim.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            x, y = get_batch(
                cfg.train_bin, cfg.block_size, cfg.batch_size, cfg.device, rc_prob=cfg.rc_prob
            )
            _, loss = model(x, y)
            (loss / cfg.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()

    print(f"done. best val = {best_val / LN2:.4f} bits/bp  (random baseline = 2.0)")


if __name__ == "__main__":
    main()
