# resdoor

Exploration toolkit for [Jane Street's Dormant LLM Backdoor Puzzle](https://www.janestreet.com/puzzles/dormant-llm/).

Three language models have been backdoored with hidden triggers. This repository provides:

- **`exploration.ipynb`** — Jupyter notebook for interactive investigation (mirrors the official demo notebook)
- **`src/resdoor/`** — reusable Python helpers for chat completions, activation retrieval, and trigger analysis

## Quick Start

```bash
pip install -r requirements.txt
# or install the package in editable mode:
pip install -e ".[dev]"
```

Open the notebook:

```bash
jupyter notebook exploration.ipynb
```

Then follow the steps in the notebook:

1. **Request API access** — `await client.request_access("<your_email>")`
2. **Set your API key** — `client.set_api_key("<your_api_key>")`
3. **Run baseline chat completions** to observe normal behaviour
4. **Retrieve internal activations** to spot anomalies
5. **Probe trigger candidates** from `src/resdoor/triggers.py`
6. **Verify discovered triggers** across multiple prompts

## Models

| Model | HuggingFace handle |
|---|---|
| Warmup | `dormant-model-warmup` |
| Model 1 | `dormant-model-1` |
| Model 2 | `dormant-model-2` |
| Model 3 | `dormant-model-3` |

## Package Layout

```
src/resdoor/
├── __init__.py      # public API
├── client.py        # ResdoorClient — thin async wrapper around jsinfer
├── analysis.py      # activation vector utilities and heatmap plotting
└── triggers.py      # TRIGGER_CANDIDATES list and request builders
```

## Contest Details

- **Submissions:** dormant-puzzle@janestreet.com by April 1, 2026
- **Prizes:** $50 000 total prize pool
- **Support:** dormant-puzzle-support@janestreet.com
