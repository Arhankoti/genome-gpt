"""The contract between the frontier model and the genome model.

Every tool is declared once here (ANTHROPIC_TOOLS); OPENAI_FUNCTIONS is derived from it.
"""

# --- Anthropic (Messages API `tools`) ---
ANTHROPIC_TOOLS = [
    {
        "name": "dna_score",
        "description": "Score how natural/plausible a DNA sequence is under the genome model. "
        "Returns mean log-probability per base, perplexity, and bits per "
        "nucleotide (lower bits = more natural; ~2.0 = random).",
        "input_schema": {
            "type": "object",
            "properties": {"sequence": {"type": "string", "description": "DNA string (ACGTN)"}},
            "required": ["sequence"],
        },
    },
    {
        "name": "dna_generate",
        "description": "Generate a novel DNA continuation from a prompt sequence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Seed DNA (ACGTN)"},
                "n_bases": {"type": "integer", "description": "How many bases to generate"},
                "temperature": {"type": "number", "description": "Sampling temperature (0.6-1.2)"},
            },
            "required": ["prompt", "n_bases"],
        },
    },
    {
        "name": "dna_variant_effect",
        "description": "Raw log-likelihood ratio of a single-base substitution (whole "
        "window centered on the variant). Negative LLR means the variant makes the "
        "sequence less likely — but nearly every change to real DNA does, so the sign "
        "alone is not a disruption call. Use dna_variant_report for a calibrated verdict.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ref_seq": {"type": "string", "description": "Reference DNA window (ACGTN)"},
                "pos": {"type": "integer", "description": "0-based position of the variant"},
                "alt_base": {"type": "string", "description": "Alternate base (A/C/G/T)"},
            },
            "required": ["ref_seq", "pos", "alt_base"],
        },
    },
    {
        "name": "dna_saturation_scan",
        "description": "Scan every single-base substitution across a DNA window and "
        "return the most surprising positions (ranked by single-site log-likelihood "
        "ratio, most negative = the alt base is most unexpected there). This is a "
        "fast single-site surprise measure using left context only; it does not "
        "capture a mutation's downstream effect. Confirm individual hits with "
        "dna_variant_effect for a full whole-window disruption estimate.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sequence": {"type": "string", "description": "DNA window (ACGTN)"},
                "top_k": {
                    "type": "integer",
                    "description": "How many top hits to return (default 20)",
                },
            },
            "required": ["sequence"],
        },
    },
    {
        "name": "dna_embed",
        "description": "Return a fixed-length embedding vector for a DNA sequence "
        "(for similarity/clustering).",
        "input_schema": {
            "type": "object",
            "properties": {"sequence": {"type": "string", "description": "DNA string (ACGTN)"}},
            "required": ["sequence"],
        },
    },
    {
        "name": "dna_score_report",
        "description": "Score a DNA sequence and return a plain-English verdict on "
        "whether it looks like real DNA, with the numbers behind it. Combines three "
        "computed signals: distance below the ~2.0 random line, whether the model "
        "scores the sequence below a composition-preserving shuffle of it "
        "(grammar_gain; real sequential structure beyond base composition), and "
        "complexity (a low score on a repetitive sequence is repetition, not "
        "grammar). Verdict is one of dna_like / plausible_composition / random_like "
        "/ low_complexity. Use this instead of dna_score when a human-readable "
        "judgment is wanted.",
        "input_schema": {
            "type": "object",
            "properties": {"sequence": {"type": "string", "description": "DNA string (ACGTN)"}},
            "required": ["sequence"],
        },
    },
    {
        "name": "dna_variant_report",
        "description": "Judge whether a single-base substitution is likely disruptive, "
        "with a plain-English verdict and the numbers behind it. The variant's "
        "whole-window LLR is ranked against every other single-base substitution "
        "within ±flank bp (disruption_percentile = fraction of nearby changes it is "
        "more disruptive than), and the surrounding DNA is checked first (a variant in "
        "random-like or repetitive DNA is unreliable_context). Verdict is one of "
        "likely_disruptive / uncertain / likely_tolerated / unreliable_context. A "
        "model-plausibility call, not a clinical pathogenicity prediction: inside genes "
        "the model's surprise tracks the genome's codon-position base composition at "
        "least as much as protein impact, so a synonymous change can score as disruptive "
        "and missense vs synonymous is near chance. Use this "
        "instead of dna_variant_effect when a human-readable judgment is wanted.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ref_seq": {
                    "type": "string",
                    "description": "Reference DNA (ACGTN) with the variant well inside it; "
                    "a few hundred bp of context on each side is ideal",
                },
                "pos": {"type": "integer", "description": "0-based position of the variant"},
                "alt_base": {"type": "string", "description": "Alternate base (A/C/G/T)"},
                "flank": {
                    "type": "integer",
                    "description": "Background radius in bp (default 30)",
                },
            },
            "required": ["ref_seq", "pos", "alt_base"],
        },
    },
    {
        "name": "dna_generation_report",
        "description": "Generate DNA from a prompt and judge how DNA-like it is against "
        "a reference window: k-mer fidelity (lower Jensen-Shannon = more natural) AND "
        "novelty (copied-k-mer fraction; high = the model is regurgitating, not "
        "generating). Both must be good — a low divergence with high copying means "
        "memorization, not generation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reference": {
                    "type": "string",
                    "description": "Real DNA window to compare against (ACGTN)",
                },
                "prompt": {"type": "string", "description": "Seed DNA (ACGTN); default 'A'"},
                "n_bases": {
                    "type": "integer",
                    "description": "How many bases to generate (default 1000)",
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature (default 0.9)",
                },
            },
            "required": ["reference"],
        },
    },
]

# --- OpenAI (Chat Completions `tools`) ---
OPENAI_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            k: v
            for k, v in {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            }.items()
        },
    }
    for t in ANTHROPIC_TOOLS
]
