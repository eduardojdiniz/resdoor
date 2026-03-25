# resdoor

LLM backdoor research toolkit for the Jane Street dormant LLM puzzle. Library only — no CLI, no web server. Entry points are Jupyter notebooks and the autonomous research loop.

## Architecture

**Functional Core + Imperative Shell (FCIS)**:

- **Core** (pure, no I/O): `models.py` (frozen Pydantic v2 + `ProbeClient` Protocol), `scoring.py` (pure functions), `seeds.py` (immutable constants)
- **Shell** (async I/O): `client.py` (jsinfer API), `local_client.py` (PyTorch), `runner.py`, `local_runner.py`, `log.py`
- **Analysis** (numpy/matplotlib): `analysis.py`

## Python & Layout

- **Python 3.13+**
- **src-layout**: all library code lives under `src/resdoor/`
- `from __future__ import annotations` in every module
- NumPy-style docstrings

## Dependency Management

Uses **uv** exclusively:

```
uv sync                 # install/update deps (API-only)
uv sync --extra local   # install with PyTorch for local screening
uv run <cmd>            # run anything in the venv
```

## Lint, Format, Type Check

```
uv run ruff check src/resdoor/     # lint
uv run ruff format src/resdoor/    # format
uv run mypy src/resdoor/           # type check (basic, not strict)
```

## Key Modules

| Module | Role |
|---|---|
| `models.py` | Frozen Pydantic v2 domain models + `ProbeClient` Protocol |
| `scoring.py` | Pure scoring functions: behavioral, activation divergence, consistency, composite |
| `client.py` | Async batch API wrapper around jsinfer, implements `ProbeClient` |
| `local_client.py` | Local PyTorch client (`LocalClient`), implements `ProbeClient` via HuggingFace transformers |
| `runner.py` | Experiment orchestration: accepts any `ProbeClient` backend → score → verdict |
| `local_runner.py` | Local screening orchestration: `run_local_screening_batch` for warmup model |
| `log.py` | JSONL I/O: append_runs, load_log, load_hits |
| `analysis.py` | Activation vector extraction, cosine similarity, heatmap plotting |
| `seeds.py` | Immutable seed categories and trigger candidates |

## Public API (`__all__`)

```python
from resdoor import (
    # Models & Protocol
    Hypothesis, ProbeConfig, AnomalyScore, ExperimentRun, ResdoorSettings, ProbeClient,
    # Scoring
    score_behavioral, score_activation_divergence, score_consistency, compute_anomaly_score,
    # Client & Runner
    ResdoorClient, run_experiment_batch,
    # Local (requires uv sync --extra local)
    LocalClient, run_local_screening_batch,
    # Logging
    append_runs, load_log, load_hits,
    # Analysis
    extract_activation_vectors, cosine_similarity_matrix, plot_activation_heatmap,
    # Seeds
    SEED_CATEGORIES, SEED_TRIGGERS,
)
```

## Entry Points

- `puzzle.ipynb` — consolidated notebook (setup, baselines, probing, visualization, verification)
- `run_autoresearch.sh` — unified autonomous research harness: alternates Phase A (local warmup screening, free) and Phase B (theory + 671B API testing, paced). Configurable via `LOCAL_ITERATIONS`, `API_ITERATIONS`, `MAX_CYCLES`, `PAUSE_BETWEEN_API`.

## Data Layout

- `data/experiment_log.jsonl` — append-only API experiment results (Track 2: 671B models)
- `data/hypothesis_bank.json` — active hypotheses for API testing
- `data/baselines/` — cached baseline responses (SHA-256 keyed)
- `data/program.md` — iteration directives for the autonomous loop
- `data/local_experiment_log.jsonl` — append-only local screening results (Track 1: warmup model)
- `data/local_hypothesis_bank.json` — hypotheses for local screening
- `data/local_iteration_state.json` — local screening iteration state

## Rules

- No test files in the library
- Do not add CLI entry points or web servers
- Keep `__all__` in `__init__.py` as the single source of truth for public API
- All domain models must be `frozen=True, extra="forbid"` (FCIS core)
- Scoring functions must be pure (no I/O, no side effects)
