# resdoor

LLM backdoor research toolkit for the Jane Street dormant LLM puzzle. Library only — no CLI, no web server. Entry points are Jupyter notebooks.

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
| `client.py` | Async API wrapper around jsinfer |
| `analysis.py` | Activation vector extraction, cosine similarity, heatmap plotting |
| `triggers.py` | Trigger candidate constants and probe request builders |

## Public API (`__all__`)

```python
from resdoor import (
    ResdoorClient,               # client.py — async API client
    extract_activation_vectors,  # analysis.py
    cosine_similarity_matrix,    # analysis.py
    plot_activation_heatmap,     # analysis.py
    TRIGGER_CANDIDATES,          # triggers.py — candidate strings
    build_trigger_probe_requests,# triggers.py
)
```

## Notebooks

- `exploration.ipynb` — exploratory analysis
- `dormant_llm_puzzle.ipynb` — main puzzle workflow

## Rules

- No test files in the library
- Do not add CLI entry points or web servers
- Keep `__all__` in `__init__.py` as the single source of truth for public API
