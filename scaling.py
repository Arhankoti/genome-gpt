"""Scaling sweep: train a ladder of model sizes on the SAME data, split, and
budget, then watch the honest number (held-out bits/bp) move — and where it
stops moving.

The one rule that keeps the curve honest: vary ONLY capacity. Data, the held-out
split (Part 5's whole-genome val), and the training budget (tokens seen =
iters * batch_size * block_size, held constant with block_size fixed) are frozen
across rungs. We report val bits/bp AND the train-val gap vs parameter count, so
a widening gap exposes the model starting to memorize rather than generalize.

    python scaling.py --mode train --tokens 12e6         # train the ladder
    python scaling.py --mode collect --fig scaling.png   # read ckpts, plot

Training real rungs is slow/GPU-bound; CI never trains — it tests the pure
assembly and the param-count helper.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

from config import ITOS, SEP_ID, Config
from inference import GenomeModel

# (label, capacity overrides). block_size is fixed in base_overrides so that a
# constant iter count == constant tokens-seen across rungs.
LADDER = [
    ("xs", {"n_layer": 2, "n_embd": 64, "n_head": 4}),
    ("s", {"n_layer": 3, "n_embd": 96, "n_head": 4}),
    ("m", {"n_layer": 4, "n_embd": 128, "n_head": 4}),
    ("l", {"n_layer": 5, "n_embd": 192, "n_head": 6}),
]

_NOTE = (
    "data, held-out split, and tokens-seen are fixed across rungs; only capacity "
    "varies. A widening gap = the model starting to memorize rather than generalize."
)


def _bits(gm, ids, max_eval_tokens):
    """bits/bp of the model on a capped token slice (SEP boundaries dropped)."""
    ids = ids[:max_eval_tokens]
    seq = "".join(ITOS[int(i)] for i in ids[ids != SEP_ID])
    return gm.score(seq)["bits_per_bp"]


def ckpt_path(ckpt_dir, prefix, label):
    return os.path.join(ckpt_dir, f"{prefix}{label}.pt")


def _train_rung(label, overrides, base_overrides, iters, ckpt_dir, prefix):
    """Shell out to train.py for one rung at the fixed budget/split."""
    out = ckpt_path(ckpt_dir, prefix, label)
    cmd = [sys.executable, "train.py", "--max_iters", str(iters), "--ckpt_path", out]
    for k, v in {**base_overrides, **overrides}.items():
        cmd += [f"--{k}", str(v)]
    print(f"  [train {label}] {' '.join(cmd[2:])}")
    subprocess.run(cmd, check=True)
    return out


def run_ladder(
    ladder=LADDER,
    *,
    mode="collect",
    tokens=12e6,
    base_overrides=None,
    ckpt_dir="checkpoints",
    ckpt_prefix="scale_",
    max_eval_tokens=200_000,
    train_bin=None,
    val_bin=None,
):
    """Train (mode='train') or read (mode='collect') the ladder, returning the
    scaling report. In 'train' mode iters is derived so tokens-seen is constant
    across rungs (block_size must be fixed in base_overrides). Bits are computed
    the same way for every rung: model.score over a capped slice of train/val.
    """
    cfg = Config()
    base_overrides = base_overrides or {"block_size": 256, "batch_size": 32, "device": "cpu"}
    train_bin = train_bin or cfg.train_bin
    val_bin = val_bin or cfg.val_bin
    block = int(base_overrides.get("block_size", cfg.block_size))
    batch = int(base_overrides.get("batch_size", cfg.batch_size))
    iters = max(1, int(tokens // (batch * block)))

    if mode == "train":
        for label, ov in ladder:
            _train_rung(label, ov, base_overrides, iters, ckpt_dir, ckpt_prefix)

    train_ids = np.fromfile(train_bin, dtype=np.uint8)
    val_ids = np.fromfile(val_bin, dtype=np.uint8)

    rungs = []
    for label, _ in ladder:
        path = ckpt_path(ckpt_dir, ckpt_prefix, label)
        if not os.path.exists(path):
            continue
        gm = GenomeModel(path)
        train_bits = _bits(gm, train_ids, max_eval_tokens)
        val_bits = _bits(gm, val_ids, max_eval_tokens)
        rungs.append(
            {
                "label": label,
                "params": int(gm.model.num_params()),
                "train_bits_per_bp": float(train_bits),
                "val_bits_per_bp": float(val_bits),
                "gap": float(val_bits - train_bits),
            }
        )
    rungs.sort(key=lambda r: r["params"])
    return {
        "budget": {"tokens_seen": int(iters * batch * block), "iters": iters},
        "rungs": rungs,
        "note": _NOTE,
    }


def _print_report(report):
    print(
        f"\n  budget: {report['budget']['tokens_seen']:,} tokens-seen "
        f"({report['budget']['iters']} iters), fixed across rungs\n"
    )
    hdr = f"  {'rung':<5} {'params':>10} {'train b/bp':>11} {'val b/bp':>10} {'gap':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in report["rungs"]:
        print(
            f"  {r['label']:<5} {r['params']:>10,} {r['train_bits_per_bp']:>11.4f} "
            f"{r['val_bits_per_bp']:>10.4f} {r['gap']:>+8.4f}"
        )
    print(f"\n  {report['note']}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=("train", "collect"), default="collect")
    ap.add_argument("--tokens", type=float, default=12e6, help="tokens-seen budget per rung")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_eval_tokens", type=int, default=200_000)
    ap.add_argument("--out", default=None, help="write report JSON here")
    ap.add_argument("--fig", default=None, help="write scaling PNG here")
    args = ap.parse_args()

    base = {"block_size": args.block_size, "batch_size": args.batch_size, "device": "cpu"}
    report = run_ladder(
        mode=args.mode,
        tokens=args.tokens,
        base_overrides=base,
        max_eval_tokens=args.max_eval_tokens,
    )
    _print_report(report)

    if args.out:
        import json

        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  wrote {args.out}")
    if args.fig:
        from viz import render_scaling

        render_scaling(report, out_path=args.fig)
        print(f"  wrote {args.fig}")


if __name__ == "__main__":
    main()
