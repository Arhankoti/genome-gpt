"""Dispatch a tool call (name + args) to the genome model. Pure functions in,
JSON-serializable dict out. Shared by both the Anthropic and OpenAI agents.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import GenomeModel

_MODEL = None
CLASSIC_TOOLS = ("dna_find_orfs", "dna_gc_scan", "dna_restriction_sites")


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = GenomeModel()
    return _MODEL


def _classic(name: str, args: dict) -> dict | None:
    """The deterministic tools (Part 12): pure numpy, so they never load the model."""
    import classic

    if name == "dna_find_orfs":
        return classic.orf_report(
            args["sequence"],
            min_codons=int(args.get("min_codons", 100)),
            top_k=int(args.get("top_k", 10)),
        )
    if name == "dna_gc_scan":
        return classic.gc_report(args["sequence"], window=int(args.get("window", 1000)))
    if name == "dna_restriction_sites":
        return classic.sites_report(
            args["sequence"],
            enzymes=args.get("enzymes"),
            circular=bool(args.get("circular", False)),
        )
    return None


def dispatch(name: str, args: dict) -> dict:
    if name in CLASSIC_TOOLS:
        try:
            return _classic(name, args)
        except ValueError as e:  # e.g. an unknown enzyme name: tell the caller, don't crash
            return {"error": str(e)}
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
    if name == "dna_variant_report":
        from variants import variant_report

        # The raw LLR's sign says little (almost any change to real DNA lowers its
        # likelihood), so the variant is ranked against every other substitution
        # within ±flank bp, scored the same way. Bounded: scalars + short strings only.
        return variant_report(
            args["ref_seq"],
            int(args["pos"]),
            args["alt_base"],
            effects_fn=m.variant_effects,
            score_fn=lambda s: m.score(s)["bits_per_bp"],
            flank=int(args.get("flank", 30)),
            window=m.cfg.block_size,
        )
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
    if name == "dna_orf_judge":
        from orfjudge import judge_pairs

        # The ORF finder proposes; classic codon usage and the model's anticipation
        # each judge every overlapping antisense pair. Bounded: counts + top_k pairs.
        return judge_pairs(
            args["sequence"],
            m.base_probs,
            min_codons=int(args.get("min_codons", 100)),
            top_k=int(args.get("top_k", 10)),
        )
    if name == "dna_compare":
        from fingerprints import similarity_report

        # Embedding similarity next to composition-only controls; scalars only.
        return similarity_report(m.embed, args["sequence_a"], args["sequence_b"])
    return {"error": f"unknown tool {name}"}
