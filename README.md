# resdoor

Exploration toolkit for [Jane Street's Dormant LLM Backdoor Puzzle](https://www.janestreet.com/puzzles/dormant-llm/).

Three language models have been backdoored with hidden triggers. This library provides batch probing, anomaly scoring, activation analysis, and an autonomous research loop that iterates until all triggers are found.

## Quick Start

```bash
uv sync                          # install dependencies
cp .env.example .env             # add your JSINFER_API_KEY
uv run jupyter notebook puzzle.ipynb  # interactive exploration
```

## Autonomous Research Loop

Runs Claude Code in a headless loop, generating hypotheses, probing models, scoring results, and refining — unattended.

```bash
# 50 iterations in tmux with caffeinate (prevents macOS sleep)
tmux new-session -d -s resdoor \
  "caffeinate -dims bash run_autoresearch.sh 50"

# Monitor
tmux attach -t resdoor           # live output
wc -l data/experiment_log.jsonl  # completed experiment runs
```

The loop stops early if all 3 models have confirmed triggers (`overall > 0.7` across all models).

## Architecture

**Functional Core + Imperative Shell** — frozen Pydantic v2 models and pure scoring functions at the core, async I/O isolated in the shell.

```
src/resdoor/
  __init__.py     # public API (__all__)
  models.py       # Hypothesis, ProbeConfig, AnomalyScore, ExperimentRun, ResdoorSettings
  scoring.py      # score_behavioral, score_activation_divergence, score_consistency, compute_anomaly_score
  client.py       # ResdoorClient — async batch wrapper around jsinfer
  runner.py       # run_experiment_batch — orchestrates probe → score → verdict
  log.py          # JSONL I/O: append_runs, load_log, load_hits
  analysis.py     # extract_activation_vectors, cosine_similarity_matrix, plot_activation_heatmap
  seeds.py        # SEED_CATEGORIES, SEED_TRIGGERS — immutable trigger candidates

data/
  program.md              # iteration directives for the autonomous loop
  experiment_log.jsonl    # append-only experiment results
  hypothesis_bank.json    # active hypotheses (pruned each iteration)
  baselines/              # cached baseline responses (SHA-256 keyed)
  logs/                   # per-iteration Claude output

puzzle.ipynb              # consolidated notebook: setup, baselines, probing, visualization
run_autoresearch.sh       # outer bash loop invoking Claude Code per iteration
```

## Public API

```python
from resdoor import (
    # Models (frozen, extra="forbid")
    Hypothesis, ProbeConfig, AnomalyScore, ExperimentRun, ResdoorSettings,
    # Scoring (pure functions)
    score_behavioral, score_activation_divergence, score_consistency, compute_anomaly_score,
    # Client (async I/O shell)
    ResdoorClient,
    # Runner
    run_experiment_batch,
    # Logging
    append_runs, load_log, load_hits,
    # Analysis
    extract_activation_vectors, cosine_similarity_matrix, plot_activation_heatmap,
    # Seeds
    SEED_CATEGORIES, SEED_TRIGGERS,
)
```

## Models

| Model | ID |
|---|---|
| Warmup | `dormant-model-warmup` |
| Model 1 | `dormant-model-1` |
| Model 2 | `dormant-model-2` |
| Model 3 | `dormant-model-3` |

## Scoring

Each hypothesis is scored on three axes, then combined into a weighted composite:

| Component | Weight | Method |
|---|---|---|
| Behavioral | 0.4 | Jaccard similarity + length ratio vs baseline |
| Activation divergence | 0.4 | 1 - cosine similarity of activation vectors |
| Consistency | 0.2 | Variance-based score across prompts (needs 2+ samples) |

**Verdicts:** `confirmed` (all models > 0.7), `investigating` (any model > 0.7), `interesting` (0.4-0.7), `rejected` (< 0.4).

## Development

```bash
uv run ruff check src/resdoor/     # lint
uv run ruff format src/resdoor/    # format
uv run mypy src/resdoor/           # type check
```

## Contest Details

- **Deadline:** April 1, 2026
- **Prizes:** $50,000 total
- **Submissions:** dormant-puzzle@janestreet.com
- **Support:** dormant-puzzle-support@janestreet.com
