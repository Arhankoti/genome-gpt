"""Dispatch a tool call (name + args) to the genome model. Pure functions in,
JSON-serializable dict out. Shared by both the Anthropic and OpenAI agents.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import GenomeModel

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = GenomeModel()
    return _MODEL


def dispatch(name: str, args: dict) -> dict:
    m = get_model()
    if name == "dna_score":
        return m.score(args["sequence"])
    if name == "dna_generate":
        return {
            "sequence": m.generate(
                prompt=args.get("prompt", "A"),
                n_bases=int(args["n_bases"]),
                temperature=float(args.get("temperature", 0.8)),
                seed=args.get("seed"),
            )
        }
    if name == "dna_variant_effect":
        return m.variant_effect(args["ref_seq"], int(args["pos"]), args["alt_base"])
    if name == "dna_embed":
        v = m.embed(args["sequence"])
        return {"dim": len(v), "embedding": v}
    return {"error": f"unknown tool {name}"}
