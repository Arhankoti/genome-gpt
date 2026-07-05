"""The contract between the frontier model and the genome model.

Same four functions, declared once for Anthropic and once for OpenAI.
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
        "description": "Estimate the effect of a single-base substitution via log-likelihood "
        "ratio. Negative LLR means the variant is more disruptive.",
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
        "name": "dna_embed",
        "description": "Return a fixed-length embedding vector for a DNA sequence "
        "(for similarity/clustering).",
        "input_schema": {
            "type": "object",
            "properties": {"sequence": {"type": "string", "description": "DNA string (ACGTN)"}},
            "required": ["sequence"],
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
