# Contributing to Genome-GPT

## Setup

```bash
git clone <repo>
cd genome-gpt
pip install -r requirements-dev.txt   # runtime + pytest + ruff
```

## Running the tests

```bash
pytest tests/          # all 60 tests, ~0.3 s, CPU-only
pytest tests/ -v       # verbose: see each test name
pytest tests/test_inference.py -v     # one module only
```

The suite never needs a GPU, a real checkpoint, or an API key. A tiny
1-layer/16-dim model is built in memory per session (see `tests/conftest.py`).

## Linting and formatting

```bash
ruff check .           # lint
ruff format .          # auto-format
```

CI runs both on every push. Fix lint errors before opening a PR.

## Project structure at a glance

```
config.py        single source of truth for hyperparameters and vocab
model.py         GenomeGPT — the transformer (touch only for architecture changes)
train.py         training loop — edit for optimizer/schedule experiments
inference.py     GenomeModel — the public API called by the bridge
evaluate.py      offline benchmark (Markov baseline, GC, k-mer KL)
data/
  download.py    fetches real genomes from NCBI (needs network)
  make_synthetic.py  builds a CPU-verifiable synthetic corpus
  prepare.py     FASTA -> train.bin / val.bin / meta.pkl
bridge/
  schemas.py     tool definitions for Anthropic and OpenAI
  tools.py       dispatch layer (name + args -> GenomeModel call)
  agent.py       orchestration loop (runs the frontier model)
tests/           pytest suite mirroring the module structure above
```

## Adding a new tool to the bridge

1. **`inference.py`** — add a method to `GenomeModel`. Keep it
   `@torch.no_grad()` and return a plain dict (JSON-serializable).

2. **`bridge/schemas.py`** — add an entry to `ANTHROPIC_TOOLS` following
   the same shape as the existing four tools. The `OPENAI_FUNCTIONS` list is
   derived automatically.

3. **`bridge/tools.py`** — add an `if name == "dna_yourname":` branch in
   `dispatch()` that calls your new method.

4. **`tests/test_bridge.py`** — add a `test_dispatch_yourname()` test that
   patches `get_model` with a `MagicMock` and asserts the expected keys are
   returned.

5. **`tests/test_inference.py`** — add tests for the new `GenomeModel` method
   using the session-scoped `gm` fixture.

## Key invariants to preserve

- **Vocab order is a contract.** `STOI` in `config.py` must never be reordered;
  doing so silently breaks every existing checkpoint.
- **`inference.py` is the boundary.** The frontier model must never receive raw
  bases; `GenomeModel` must never receive prose. Keep that separation.
- **Validation = held-out genomes.** `prepare.py` splits on a contiguous tail
  so val bits/bp measures cross-genome generalization. Do not shuffle before
  splitting.
- **All tests pass on CPU.** If a new test needs a trained model it should use
  the `gm` fixture from `conftest.py`, not a real checkpoint.
