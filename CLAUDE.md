# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small **DNA-native** GPT trained from scratch (char-level, ~12M params) that a frontier model (Claude Opus / GPT) calls **as a tool**. The big model owns language and reasoning; this model is the specialist that computes over DNA. The single most important architectural rule: **the frontier model never reasons over raw `ACGT`, and the genome model never handles prose.** The boundary between them is the JSON schema in `bridge/schemas.py`.

## Commands

Python is `python3.11` in this environment (has torch, numpy; system `python`/`python3` do **not** — they're 3.9 without torch). `pytest` and `ruff` are on PATH.

```bash
# Tests (CPU, tiny model — no GPU/network needed; ~0.4s)
python3.11 -m pytest tests/ -q
python3.11 -m pytest tests/test_saturation.py -q          # one module
python3.11 -m pytest tests/test_inference.py::test_score_keys   # one test

# Lint / format — CI runs both and fails on either (line-length 100)
ruff check .
ruff format --check .        # ruff format .  to apply

# Offline end-to-end pipeline (no NCBI, no API keys — this is what CI-equivalent verification uses)
python3.11 data/make_synthetic.py                         # -> data/synth.fasta
python3.11 data/prepare.py --fasta data/synth.fasta       # -> train.bin / val.bin / meta.pkl
python3.11 train.py --max_iters 300 --n_layer 2 --n_embd 64 --block_size 128 \
    --device cpu --ckpt_path checkpoints/synth.pt          # any Config field is a --flag
python3.11 evaluate.py --ckpt checkpoints/synth.pt --max_eval_tokens 36000
python3.11 landscape.py --fasta data/synth.fasta --record 0 --start 0 --end 250 \
    --ckpt checkpoints/synth.pt --out landscape.png       # saturation heatmap
```

`matplotlib` is a **dev-only, optional** dependency (`viz.py` / `landscape.py` / `dream.py` only). It's in `requirements-dev.txt`, not `requirements.txt`, and is imported behind a guard so the runtime path and tests never require it.

## Architecture (the big picture)

Everything flows through a deliberate tool boundary. Read these files together to understand the whole system:

- **`config.py`** — single source of truth for all hyperparameters *and* the fixed **6-token vocabulary** `{A,C,G,T,N,|}` (`STOI`/`ITOS`/`VOCAB_SIZE`). The order is a hard contract — never reorder. `|` is a genome/contig boundary token; `COMPLEMENT` is the reverse-complement map in id-space. `Config` is a dataclass; a checkpoint stores its full config so architecture always matches the weights (old and new checkpoints are interchangeable without editing `config.py`).
- **`model.py`** — nanoGPT-shaped decoder-only transformer. No tokenizer: single-nucleotide resolution falls out of the char vocab for free. `forward(idx, targets=None)` returns `(logits, loss)`.
- **`inference.py` — the source of truth.** `GenomeModel` wraps a trained checkpoint and exposes the public methods that are the tool contract: `score`, `generate`, `variant_effect`, `saturation_scan`, `embed`. All are `@torch.no_grad()` and (except `generate`'s sampling) deterministic. Long inputs use a **sliding window** (stride `block_size//2`, keeping the second half of each window for better left-context) so callers never silently truncate.
- **`bridge/`** — `schemas.py` declares the **7 tools once** as `ANTHROPIC_TOOLS` (`dna_score`, `dna_generate`, `dna_variant_effect`, `dna_saturation_scan`, `dna_embed`, `dna_score_report`, `dna_generation_report`); `OPENAI_FUNCTIONS` is auto-derived from it (don't hand-maintain a second list). `tools.py::dispatch(name, args)` maps a tool call to model computation and returns a JSON-serializable dict. `agent.py` is the orchestration loop (Anthropic default, `--openai` alternate).
- **`generation.py`** — numpy-only, model-free statistics for judging generated DNA (`gc_content`, `kmer_spectrum`, `js_divergence`, `copy_stats`, `dream_report`). Anything needing the model is passed in as a closure, so this stays pure and importable by `evaluate.py`, tests, and the bridge without pulling torch/matplotlib.

### Data pipeline invariants

`data/prepare.py` encodes a multi-record FASTA into `train.bin`/`val.bin` (uint8) + `meta.pkl`. Two correctness points that the design depends on:
1. Records are joined with the `|` **boundary token** so the model never learns transitions across genome boundaries — and each split gets its own internal boundaries; the two splits never touch.
2. Validation is **whole held-out genomes** (Part 5): `--holdout name1,name2` or a seeded `--holdout_k N` routes entire records into `val.bin`, so val bits/bp measures *cross-genome generalization*, not recall. `meta.pkl` records `train_genomes`/`val_genomes`. `--holdout_k 0` reproduces the legacy contiguous-tail split for Parts 2–4.
3. A **leakage guard** (Part 7) refuses a whole-genome split when a held-out genome has a MinHash near-twin in training — a near-duplicate strain across the split would make the generalization claim secretly partly memorization. `data/quality.py` (numpy+stdlib only: canonical-k-mer MinHash, `dedup_records`, `leakage_check`) backs it; `prepare.py`'s `--clean`/`--dedup` and the always-on guard (`--allow_leakage` to override) consume it; `dataqc.py` is the read-only inspector.

`train.py` applies reverse-complement augmentation on the fly (`rc_prob`) and saves a checkpoint on best val loss. `evaluate.py` reports order-k Markov baselines next to the neural model, and `benchmark.py` extends this to a **per-held-out-genome** neural-vs-Markov report (reusing `evaluate.markov_bits` + `generation.py` stats — never forking them) — **the neural net must beat the best k-gram to justify itself.** `scaling.py` runs a size **ladder** at fixed data/split/budget (only capacity varies) and reports val bits/bp **and the train–val gap** vs `model.params_for_config(cfg)` — a widening gap is the memorization signal. Part 6 varied capacity; Part 8 asks the complementary question — freeze the ladder and budget, vary only the **data** — via `scaling.align_runs` (pure) + `scaling.py --compare a.json,b.json` + `viz.render_scaling_compare` (two ladders' held-out curves + gap, one bits/bp axis per panel). The honest metric throughout is bits/nucleotide: **2.0 = random over 4 bases**, lower = more natural. `data/download.py` writes each record's FASTA header as `>{friendly_name} {accession}` (`relabel_header`) because the whole pipeline keys a genome by that first-token name (`--holdout`, the leakage guard) and it must match the manifest — a raw NCBI accession header silently breaks name-based holdout.

## Conventions specific to this repo

- **Adding a tool** means touching places in lockstep: the computation (usually a `GenomeModel` method in `inference.py`, but not always — `dna_generation_report` and `dna_score_report` are *composed in the bridge* from model calls + `generation.py` stats), a `dispatch` branch (`bridge/tools.py`), an entry in `ANTHROPIC_TOOLS` (`bridge/schemas.py`), and tests. Keep `bridge` return payloads **bounded** — `dna_saturation_scan` returns only ranked hits + summary (never the full grid); `dna_generation_report` and `dna_score_report` return scalars/short strings only (never the sequence) — to cap tokens sent to the frontier model. Bumping the tool count also means updating the count assertion in `tests/test_bridge.py` (currently **7**).
- **`dna_score_report` (Part 9) turns bits/bp into a plain-English verdict** (`generation.score_verdict`, pure) using two computed anchors — distance below the 2.0 random line, and `grammar_gain` = bits saved on the sequence vs a **`composition_shuffle`** of it (same base composition, order destroyed; the model judges its own control) — plus a low-complexity guard so a homopolymer's low score isn't read as grammar. The verdict is a gloss; the tool always returns every number it was computed from. `score.py` is the CLI; `viz.render_score_report` the thermometer.
- **`saturation_scan` vs `variant_effect` are different measurements — do not conflate them.** `saturation_scan` is a fast, single-pass **single-site surprise** (`logP(alt|left) − logP(ref|left)`, left context only, ref column exactly `0.0`, position 0 and window edges skipped). `variant_effect` re-scores a whole centered window and captures both flanks *plus* the downstream ripple. The distinction is stated in docstrings, the tool description, and the README; keep it that way.
- **Tests run on an untrained tiny CPU checkpoint** (`tiny_ckpt` fixture in `tests/conftest.py`: 1 layer, 16 embd, 32 block_size). LLRs are noise on an untrained model, so assert only *internal* consistency (shapes, ref-column-zero, determinism, indexing against a hand-rolled forward pass) — **never** sign agreement between the neural tools. Guard any matplotlib test with `pytest.importorskip("matplotlib")`.
- **Direct-script imports:** `data/*.py` and `bridge/*.py` do `sys.path.insert(...)` before importing top-level modules so they run as `python3.11 data/prepare.py`. `E402` is intentionally ignored for those paths in `pyproject.toml` — don't "fix" it.
- **Vocab / architecture changes break existing checkpoints.** Changing the 6-token vocab requires a retrain and is a breaking change; the checkpoint-stored config is what keeps old weights loadable otherwise.

## Git / commit convention (this repo)

Local git identity is set to **Arhan Koti `<arhan.koti@gmail.com>`** — commit as him, and **do not add a `Co-Authored-By: Claude` trailer** (this overrides the default). Work on a feature branch and open PRs with **base `main`** (CI runs lint + CPU tests on PRs to `main`). `temp/`, `plans/`, and `blog/` are gitignored — artifacts there (including generated checkpoints and PNGs) are intentionally untracked. A helper at `temp/git-ship.sh` does branch → commit (correct identity, no trailer) → push → PR-to-main from staged changes.
