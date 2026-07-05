"""Load a trained checkpoint and expose pure DNA-native functions.

This is the source of truth the frontier model calls. All functions are
deterministic except generate() (sampling). Long inputs are scored with a
sliding window so callers never silently truncate.
"""

import numpy as np
import torch
from torch.nn import functional as F

from config import ITOS, LN2, STOI, Config
from model import GenomeGPT


def _encode(seq, device):
    ids = [STOI.get(c.upper(), STOI["N"]) for c in seq]
    return torch.tensor(ids, dtype=torch.long, device=device)


class GenomeModel:
    """DNA-native model loaded from a trained checkpoint.

    The four public methods (score, generate, variant_effect, embed) are the
    contract exposed to frontier models via bridge/schemas.py. All are
    torch.no_grad()-wrapped and safe to call from multiple threads as long as
    the caller does not mutate the model.

    Args:
        ckpt_path: Path to a .pt checkpoint produced by train.py. Defaults to
            cfg.ckpt_path. The config stored in the checkpoint determines the
            model architecture, so old and new checkpoints are interchangeable
            without touching config.py.
    """

    def __init__(self, ckpt_path=None):
        cfg = Config()
        ckpt_path = ckpt_path or cfg.ckpt_path
        ck = torch.load(ckpt_path, map_location=cfg.device, weights_only=True)
        # rebuild config from checkpoint so architecture always matches weights
        saved = ck["config"]
        for k, v in saved.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        self.cfg = cfg
        self.device = cfg.device
        self.model = GenomeGPT(cfg).to(cfg.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()

    # --- per-base log-probabilities under the model (teacher forcing) ---
    @torch.no_grad()
    def _token_logprobs(self, ids):
        """Return logP(x_t | x_<t) for t=1..T-1 over a window <= block_size."""
        x = ids[:-1].unsqueeze(0)
        tgt = ids[1:]
        logits, _ = self.model(x)
        logp = F.log_softmax(logits[0], dim=-1)
        return logp[torch.arange(len(tgt), device=self.device), tgt]

    @torch.no_grad()
    def score(self, seq):
        """Mean log-prob per base, perplexity, and bits/bp. Sliding window for long seqs."""
        ids = _encode(seq, self.device)
        B = self.cfg.block_size
        if len(ids) <= B:
            lps = self._token_logprobs(ids)
        else:
            stride = B // 2
            chunks = []
            for start in range(0, len(ids) - 1, stride):
                window = ids[start : start + B]
                if len(window) < 2:
                    break
                lp = self._token_logprobs(window)
                # keep only the second half of each window (better left-context),
                # except the first window where we keep everything
                keep = lp if start == 0 else lp[stride - 1 :]
                chunks.append(keep)
            lps = torch.cat(chunks)
        mean_lp = lps.mean().item()
        return {
            "mean_logprob": mean_lp,
            "perplexity": float(np.exp(-mean_lp)),
            "bits_per_bp": -mean_lp / LN2,
            "n_scored": int(lps.numel()),
        }

    @torch.no_grad()
    def generate(self, prompt="A", n_bases=200, temperature=0.8, top_k=4, seed=None):
        """Sample a DNA continuation from a prompt sequence.

        Args:
            prompt: Seed DNA string (ACGTN). The returned string starts with
                the prompt and appends n_bases new characters.
            n_bases: Number of bases to generate (appended after the prompt).
            temperature: Sampling temperature. Lower (0.6) = more conservative;
                higher (1.2) = more random. Values near 0 approach greedy decode.
            top_k: Restrict sampling to the k most-likely next tokens. 4 caps
                output to real DNA bases; None uses the full vocabulary.
            seed: Optional integer seed for reproducible sampling.

        Returns:
            String of length len(prompt) + n_bases.
        """
        if seed is not None:
            torch.manual_seed(seed)
        ids = _encode(prompt, self.device).unsqueeze(0)
        out = self.model.generate(ids, n_bases, temperature=temperature, top_k=top_k)
        return "".join(ITOS[i] for i in out[0].tolist())

    @torch.no_grad()
    def variant_effect(self, ref_seq, pos, alt_base, window=None):
        """Log-likelihood ratio of an alt allele vs ref at position `pos`.

        Negative => the substitution makes the sequence less likely (more
        disruptive). The variant is centered in the scoring window; edge
        positions have little left-context and score unreliably.
        """
        window = window or self.cfg.block_size
        ref = list(ref_seq.upper())
        assert 0 <= pos < len(ref), "pos out of range"
        alt = ref.copy()
        alt[pos] = alt_base.upper()
        half = window // 2
        lo = max(0, pos - half)
        hi = min(len(ref), lo + window)
        lo = max(0, hi - window)
        ref_ll = self.score("".join(ref[lo:hi]))["mean_logprob"] * (hi - lo)
        alt_ll = self.score("".join(alt[lo:hi]))["mean_logprob"] * (hi - lo)
        return {
            "llr": alt_ll - ref_ll,
            "ref_base": ref[pos],
            "alt_base": alt_base.upper(),
            "position": pos,
            "interpretation": "more disruptive" if alt_ll < ref_ll else "tolerated/neutral",
        }

    @torch.no_grad()
    def embed(self, seq):
        """Return a fixed-length embedding vector for a DNA sequence.

        Computes the mean-pooled final hidden state of the transformer over the
        (up to block_size) input tokens. Useful for similarity search and
        clustering; cosine distance is a natural metric over the returned vector.

        Args:
            seq: DNA string (ACGTN). Silently truncated to block_size tokens.

        Returns:
            List of floats of length cfg.n_embd (384 at default settings).
        """
        ids = _encode(seq, self.device)[: self.cfg.block_size].unsqueeze(0)
        t = self.model.transformer
        pos = torch.arange(ids.size(1), device=self.device)
        x = t.drop(t.wte(ids) + t.wpe(pos))
        for block in t.h:
            x = block(x)
        x = t.ln_f(x)
        return x[0].mean(dim=0).cpu().numpy().tolist()
