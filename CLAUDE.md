# resdoor

LLM backdoor research toolkit for the Jane Street dormant LLM puzzle. Library only — no CLI, no web server. Entry points are Jupyter notebooks and the autonomous research loop.

## Architecture

**Functional Core + Imperative Shell (FCIS)**:

- **Core** (pure, no I/O): `models.py` (frozen Pydantic v2), `scoring.py` (pure functions), `seeds.py` (immutable constants)
- **Shell** (async I/O): `client.py`, `runner.py`, `log.py`
- **Analysis** (numpy/matplotlib): `analysis.py`

## Python & Layout

- **Python 3.13+**
- **src-layout**: all library code lives under `src/resdoor/`
- `from __future__ import annotations` in every module
- NumPy-style docstrings

## Dependency Management

Uses **uv** exclusively:

```
uv sync            # install/update deps
uv run <cmd>       # run anything in the venv
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
| `models.py` | Frozen Pydantic v2 domain models: Hypothesis, ProbeConfig, AnomalyScore, ExperimentRun, ResdoorSettings |
| `scoring.py` | Pure scoring functions: behavioral, activation divergence, consistency, composite |
| `client.py` | Async batch API wrapper around jsinfer with ProbeConfig-based interface |
| `runner.py` | Experiment orchestration: batch probe → score → verdict with baseline caching |
| `log.py` | JSONL I/O: append_runs, load_log, load_hits |
| `analysis.py` | Activation vector extraction, cosine similarity, heatmap plotting |
| `seeds.py` | Immutable seed categories and trigger candidates |

## Public API (`__all__`)

```python
from resdoor import (
    # Models
    Hypothesis, ProbeConfig, AnomalyScore, ExperimentRun, ResdoorSettings,
    # Scoring
    score_behavioral, score_activation_divergence, score_consistency, compute_anomaly_score,
    # Client & Runner
    ResdoorClient, run_experiment_batch,
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
- `run_autoresearch.sh` — autonomous research loop (bash + Claude Code)

## Data Layout

- `data/experiment_log.jsonl` — append-only experiment results
- `data/hypothesis_bank.json` — active hypotheses (pruned each iteration)
- `data/baselines/` — cached baseline responses (SHA-256 keyed)
- `data/program.md` — iteration directives for the autonomous loop

## Rules

- No test files in the library
- Do not add CLI entry points or web servers
- Keep `__all__` in `__init__.py` as the single source of truth for public API
- All domain models must be `frozen=True, extra="forbid"` (FCIS core)
- Scoring functions must be pure (no I/O, no side effects)
