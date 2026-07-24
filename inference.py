"""Load a trained checkpoint and expose pure DNA-native functions.

This is the source of truth the frontier model calls. All functions are
deterministic except generate() (sampling). Long inputs are scored with a
sliding window so callers never silently truncate.
"""

import numpy as np
import torch
from torch.nn import functional as F

from config import ITOS, LN2, STOI, VOCAB_SIZE, Config
from model import GenomeGPT


def _encode(seq, device):
    ids = [STOI.get(c.upper(), STOI["N"]) for c in seq]
    return torch.tensor(ids, dtype=torch.long, device=device)


class GenomeModel:
    """DNA-native model loaded from a trained checkpoint.

    The public methods (score, generate, variant_effect, saturation_scan, embed)
    are the contract exposed to frontier models via bridge/schemas.py. All are
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
    def saturation_scan(self, seq, start=0, end=None, top_k=20):
        """Single-site LLR for every single-base substitution over seq[start:end].

        One forward pass gives logP(base | left context) at each position, so the
        LLR of each alternate base is logP(alt) - logP(ref) read directly off the
        logits. This measures how surprising the alt base is *at its own site*
        given left context only; it does NOT model the substitution's downstream
        effect on later positions. For a whole-window disruption estimate (both
        flanks + downstream ripple), confirm individual hits with variant_effect().

        Args:
            seq: DNA string (ACGTN). Left context of every scored position is the
                real sequence preceding it, so pass the full window you care about.
            start: First absolute position to score (0-based). Clamped to >= 1
                because position 0 has no left context.
            end: One past the last absolute position to score. Defaults to len(seq).
            top_k: How many most-disruptive (most-negative LLR) hits to return.

        Returns:
            dict with keys: grid, ref_bases, positions, worst_per_pos,
            most_disruptive, n_scored, note. `grid` is per scored position a list
            of [llr_A, llr_C, llr_G, llr_T]; the reference column is exactly 0.0.
            Positions with no left context (index 0, first token of each sliding
            window) are skipped, never NaN-filled. All values are plain Python
            floats/ints (JSON-serializable).
        """
        note = (
            "single-site surprise (left context only, no downstream effect); "
            "position 0 and window edges are skipped; confirm hits with variant_effect"
        )
        empty = {
            "grid": [],
            "ref_bases": "",
            "positions": [],
            "worst_per_pos": [],
            "most_disruptive": [],
            "n_scored": 0,
            "note": note,
        }
        ids = _encode(seq, self.device)
        T = len(ids)
        if T < 2:
            return empty
        end = len(seq) if end is None else min(end, T)
        start = max(0, start)
        if start >= end:
            return empty

        # logp_rows[p] holds logP(base_p | left context); NaN row => unscored.
        logp_rows = torch.full((T, VOCAB_SIZE), float("nan"), device=self.device)
        B = self.cfg.block_size
        stride = max(1, B // 2)
        for wstart in range(0, T - 1, stride):
            window = ids[wstart : wstart + B]
            if len(window) < 2:
                break
            logits, _ = self.model(window.unsqueeze(0))
            logp = F.log_softmax(logits[0], dim=-1)  # (L, vocab); logp[j] scores token j+1
            # first window: keep all local targets 1..L-1; later windows keep the
            # second half (local target >= stride) for better left context.
            first_local = 1 if wstart == 0 else stride
            for j in range(first_local - 1, len(window) - 1):
                p = wstart + j + 1
                if torch.isnan(logp_rows[p, 0]):
                    logp_rows[p] = logp[j]
            if wstart + B >= T:
                break

        real_ids = (0, 1, 2, 3)  # A, C, G, T
        grid, ref_bases, positions, worst_per_pos, hits = [], [], [], [], []
        for p in range(max(start, 1), end):
            row = logp_rows[p]
            if torch.isnan(row[0]):
                continue
            ref_id = int(ids[p].item())
            ref_lp = row[ref_id]
            llrs = [float((row[a] - ref_lp).item()) for a in real_ids]
            grid.append(llrs)
            ref_bases.append(ITOS[ref_id])
            positions.append(p)
            worst_per_pos.append(min(llrs))
            for a in real_ids:
                if a == ref_id:
                    continue
                hits.append(
                    {"position": p, "ref_base": ITOS[ref_id], "alt_base": ITOS[a], "llr": llrs[a]}
                )

        hits.sort(key=lambda h: h["llr"])
        return {
            "grid": grid,
            "ref_bases": "".join(ref_bases),
            "positions": positions,
            "worst_per_pos": worst_per_pos,
            "most_disruptive": hits[:top_k],
            "n_scored": len(positions),
            "note": note,
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
