"""Single source of truth for all hyperparameters and paths.

Everything reads from here. Override at the CLI in train.py for smoke runs.
"""

from dataclasses import asdict, dataclass

import torch


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    # --- data ---
    accession: str = "NC_000913.3"  # E. coli K-12 MG1655 (RefSeq)
    fasta_path: str = "data/genomes.fasta"  # multi-record FASTA (one per genome)
    train_bin: str = "data/train.bin"
    val_bin: str = "data/val.bin"
    meta_path: str = "data/meta.pkl"
    val_tail_frac: float = 0.10  # legacy: contiguous tail -> held-out genome(s)
    # whole-genome held-out split (Part 5). If holdout_genomes is set, those named
    # records go entirely to val; else holdout_k records are picked with holdout_seed.
    # Set holdout_k = 0 to fall back to the legacy contiguous-tail split.
    holdout_genomes: str = ""  # comma-separated record names to force into val
    holdout_k: int = 2  # else hold out this many whole records at random
    holdout_seed: int = 1337  # seed for the random holdout pick (reproducible)
    rc_prob: float = 0.5  # reverse-complement augmentation prob

    # --- model (~12M params at defaults) ---
    block_size: int = 1024
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False  # LayerNorm/Linear bias

    # --- training ---
    batch_size: int = 64
    grad_accum: int = 1
    max_iters: int = 10_000
    warmup_iters: int = 100
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 100
    sample_interval: int = 500
    ckpt_path: str = "checkpoints/ckpt.pt"

    # --- runtime ---
    device: str = pick_device()
    seed: int = 1337

    # --- bridge ---
    frontier_model: str = "claude-opus-4-8"  # set to your available model string

    def to_dict(self):
        return asdict(self)


# Fixed, tiny vocabulary. Order is the contract — never reorder.
# SEP ("|") marks genome/contig boundaries so the model does not learn spurious
# transitions across them. It is strand-agnostic (its own complement under RC).
STOI = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "|": 5}
ITOS = {i: c for c, i in STOI.items()}
VOCAB_SIZE = len(STOI)
SEP_ID = STOI["|"]
# reverse-complement map in id-space: A<->T, C<->G, N->N, SEP->SEP
COMPLEMENT = [3, 2, 1, 0, 4, 5]
LN2 = 0.6931471805599453  # for nats -> bits
