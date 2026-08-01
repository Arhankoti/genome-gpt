# Genome-GPT

A small DNA-native language model, trained from scratch, that a frontier model
(Claude Opus / GPT) can call as a tool. The big model handles language and
reasoning; this model is the specialist that computes over DNA — likelihoods,
generation, variant effects, embeddings.

**Why Python?** Training a net from scratch is autograd + GPU kernels +
optimizer — PyTorch's home turf. Rust ML (`candle`, `burn`, `tch`) is real but
training is where it's weakest. The mature Rust path is *inference*: train here
in Python, then port `inference.py` to `candle` on the same weights for a Rust
serving tier. For this project, all-Python.

## Design principle

A clean tool boundary. `inference.py` is the **source of truth** for DNA-native
computation. The frontier model owns language; the JSON schema in
`bridge/schemas.py` is the contract. The big model never reasons over raw
`ACGT`; the small model never handles prose.

## Layout

```
config.py            hyperparameters + 6-token vocab (A,C,G,T,N + boundary |)
model.py             char-level GPT (~12M params at defaults)
data/
  download.py        fetch ~20 diverse bacterial genomes (or --single = E. coli)
  make_synthetic.py  offline multi-genome corpus (verify without NCBI)
  prepare.py         multi-record FASTA -> bins, boundary tokens, held-out tail
train.py             training loop, cosine LR, RC augmentation, bits/bp eval
evaluate.py          Markov-baseline comparison + GC / k-mer checks
inference.py         score / generate / variant_effect / saturation_scan / embed
generation.py        pure stats to judge dreams: kmer fidelity + copy/novelty (numpy)
viz.py               render a saturation scan / generation sweep as a PNG (optional matplotlib)
landscape.py         CLI: scan a seq/FASTA window -> ranked table + landscape PNG
dream.py             CLI: sweep sampling temperature -> fidelity-vs-novelty table + PNG
bridge/
  schemas.py         Anthropic tools + OpenAI functions
  tools.py           dispatch tool calls to the genome model
  agent.py           orchestration loop (Opus / GPT)
```

## What's new in this increment

- **Multi-genome corpus.** `download.py` pulls ~20 diverse bacteria. One genome
  is memorization; many genomes teach DNA grammar that generalizes.
- **Boundary tokens.** Genomes are joined with a `|` token so the model never
  learns spurious transitions across genome boundaries.
- **Held-out-genome validation.** The contiguous val tail is now whole genomes
  absent from training, so val bits/bp measures *generalization*, not recall.
- **Reverse-complement augmentation.** Trains on both DNA strands (`rc_prob` in
  config), applied on the fly in `get_batch`.
- **Markov baseline.** `evaluate.py` reports order-k Markov bits/token next to
  the neural model. The neural net must beat the best k-gram to justify itself.

## Quickstart (real genomes)

```bash
pip install -r requirements.txt

python data/download.py            # ~20 bacteria -> data/genomes.fasta (NCBI)
python data/prepare.py             # -> train.bin / val.bin / meta.pkl
python train.py                    # ~12M params; ~30-90 min on a GPU
python evaluate.py                 # Markov-vs-neural table + sanity checks

export ANTHROPIC_API_KEY=...
python bridge/agent.py "Is the C->T substitution at position 50 in ATG...GCA disruptive?"
```

`python data/download.py --single` grabs only E. coli for a fast first pass.

## Quickstart (offline, no NCBI — exactly what was verified)

```bash
python data/make_synthetic.py
python data/prepare.py --fasta data/synth.fasta
python train.py --max_iters 600 --n_layer 3 --n_embd 128 --block_size 128 \
                --batch_size 32 --device cpu --ckpt_path checkpoints/synth.pt
python evaluate.py --ckpt checkpoints/synth.pt --max_eval_tokens 36000
```

## Reading the metric

Bits/token = cross-entropy / ln 2. Random over 4 bases = **2.0**. The Markov
table is the reference: if the Transformer only ties a 5-mer chain, that is the
finding. Watch for high-order k-grams *overfitting* (bits going up on held-out
genomes) — that gap is exactly what a good neural model should close.

## Saturation mutagenesis

Score *every* single-base substitution across a window in **one forward pass** and
rank the positions where the alternate base is most surprising:

```bash
python landscape.py --fasta data/synth.fasta --record 0 --start 0 --end 250 \
    --ckpt checkpoints/synth.pt --out landscape.png   # ranked table + heatmap
```

This is a **fast single-site surprise** measure: `logP(alt | left context) −
logP(ref | left context)`, read straight off the logits. It is *not* the same as
`variant_effect` — it uses left context only and ignores how a swap perturbs
*downstream* bases. Use it for a whole-gene overview, then zoom in on the top hits
with the precise, centered `variant_effect`. Exposed to the frontier model as the
`dna_saturation_scan` tool.

## Generation evaluation

The model can dream (`generate`), but is the dream any good? Judge it on **two**
axes at once — because a model can ace any statistical match by *copying* training
DNA verbatim, which is the Part-2 "memorized the eye chart" trap in a new costume:

- **Fidelity** — do the generated k-mer statistics match real DNA? (Jensen-Shannon
  divergence in bits; lower = more DNA-like.)
- **Novelty** — is it *producing* sequence, not *plagiarizing*? (fraction of long
  k-mers copied verbatim from the reference; longest exact copied run.)

```bash
python dream.py --ckpt checkpoints/synth.pt --temperatures 0.5,0.7,0.9,1.1
```

Temperature is the dial that trades these off (low → repetitive/copy-prone; high →
novel but structureless), so the honest artifact is the **sweep** that shows the
tension and where the sweet spot sits — not a single "quality" number. The model's
own `score()` on its samples is reported but is a *weak* sanity check only (a model
happily loves its own low-temperature loops). Exposed to the frontier model as the
`dna_generation_report` tool.

## What was verified here

Ran the full offline pipeline on a 6-genome synthetic corpus (CPU): boundary
tokens inserted, RC augmentation active, held-out-genome split, training loss
behaved, and the Markov comparison rendered correctly — including k=8 Markov
overfitting to 2.40 bits/token (worse than random) while the neural model stayed
~2.01. `score`, `generate`, `variant_effect`, `embed`, and sliding-window
scoring all work under the new 6-token vocab.

**Not executed here** (no network/keys in the sandbox, both standard code):
`data/download.py` (NCBI) and `bridge/agent.py` (API key). Accessions in
`download.py` are RefSeq; individual stale ones are skipped with a warning
rather than aborting. Set `frontier_model` in `config.py` to a model you can reach.

> Vocab changed from 5 to 6 tokens this increment — retrain; old checkpoints
> are incompatible.

## Development

```bash
pip install -r requirements-dev.txt   # adds pytest + ruff on top of runtime deps

pytest tests/          # 60 tests, CPU-only, ~0.3 s — no checkpoint needed
ruff check .           # lint
ruff format .          # format
```

The test suite spins up a tiny 1-layer/16-dim model in memory and exercises
every public function in `inference.py`, the data pipeline, the bridge dispatch,
and the Markov baseline — all without NCBI, a GPU, or an API key.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to add a new tool or extend the pipeline.

## Honest caveats

- **Synthetic ≠ real signal.** The synthetic genomes share little structure, so
  nothing beats ~2.0 there. Real bacteria share DNA grammar; that is where the
  neural-over-Markov gap appears. The harness is for measuring it, not faking it.
- **Variant edges.** Keep variants centered — edge positions score unreliably.
- **GPU determinism** is partial; the seed is logged in each checkpoint.

## Upgrade backlog

1. ~~Saturation mutagenesis (scan every position × 3 alts into a ranked landscape).~~
   ✅ Done — `saturation_scan` / `landscape.py` / `dna_saturation_scan`.
2. Classical tools beside neural ones (ORF finder, GC scan, restriction sites)
   behind the same schema.
3. Embedding similarity search (cosine-NN index over `embed()`).
4. Byte-level + a small Hyena block for longer context (only when a task needs >1024 bp).
5. Port `inference.py` to Rust/`candle` for a production serving tier.
