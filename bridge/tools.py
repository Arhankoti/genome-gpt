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
    if name == "dna_saturation_scan":
        r = m.saturation_scan(args["sequence"], top_k=int(args.get("top_k", 20)))
        # Return only the ranked hits + summary; omit the full grid to keep the
        # payload to the frontier model bounded.
        return {k: r[k] for k in ("most_disruptive", "n_scored", "note")}
    if name == "dna_embed":
        v = m.embed(args["sequence"])
        return {"dim": len(v), "embedding": v}
    if name == "dna_score_report":
        from generation import composition_shuffle, score_verdict

        seq = args["sequence"]
        neural_bits = m.score(seq)["bits_per_bp"]
        # Score a composition-preserving shuffle as the null: bits saved on the real
        # order = grammar beyond base composition (the model judges its own control).
        shuffle_bits = m.score(composition_shuffle(seq))["bits_per_bp"]
        # score_verdict returns only bounded scalars/strings — the plain-English
        # verdict plus every number it was computed from (nothing hides behind a word).
        return score_verdict(seq, neural_bits, shuffle_bits)
    if name == "dna_generation_report":
        from generation import copy_stats, gc_content, js_divergence, kmer_spectrum

        ref = args["reference"]
        gen = m.generate(
            prompt=args.get("prompt", "A"),
            n_bases=int(args.get("n_bases", 1000)),
            temperature=float(args.get("temperature", 0.9)),
        )
        cs = copy_stats(gen, ref)
        # Bounded payload: scalars only — never dump the generated sequence to
        # the frontier model. Fidelity (js) AND novelty (copied fraction) both
        # matter; self_bits_per_bp is a weak sanity check only.
        return {
            "kmer_js_bits": js_divergence(kmer_spectrum(gen), kmer_spectrum(ref)),
            "gc_generated": gc_content(gen),
            "gc_reference": gc_content(ref),
            "copied_kmer_fraction": cs["copied_kmer_fraction"],
            "longest_exact_match": cs["longest_exact_match"],
            "self_bits_per_bp": m.score(gen)["bits_per_bp"],
            "note": "fidelity (js) and novelty (copied fraction) must both be good; "
            "self_bits_per_bp is a weak sanity check only",
        }
    return {"error": f"unknown tool {name}"}
