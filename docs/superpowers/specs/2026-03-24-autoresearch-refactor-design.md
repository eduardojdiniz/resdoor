# Autoresearch Refactor: FCIS Core + Autonomous Trigger Discovery

**Date**: 2026-03-24
**Status**: Approved
**Goal**: Consolidate resdoor into one notebook + FCIS-refactored core, with an autoresearch-style autonomous loop that runs Opus until all three dormant model triggers are found.

## Problem

Jane Street's dormant LLM puzzle: 3 models with hidden backdoor triggers. $50k prize pool, deadline April 1, 2026. Current codebase has two template notebooks with no results, and a library that doesn't match the actual API shapes.

## Architecture

Two entry points, one FCIS core:

```
HUMAN ENTRY POINT              AGENT ENTRY POINT
puzzle.ipynb                   run_autoresearch.sh + program.md
(explore, visualize, review)   (autonomous loop, never stop)
        |                              |
        v                              v
+--------------------------------------------------+
|           FUNCTIONAL CORE (pure, frozen)          |
|  models.py   - Pydantic v2 frozen domain models   |
|  scoring.py  - behavioral, activation, consistency |
|  analysis.py - activation vectors, cosine sim      |
|  seeds.py    - initial trigger candidates          |
+--------------------------------------------------+
                       |
+--------------------------------------------------+
|           IMPERATIVE SHELL (I/O only)             |
|  client.py   - jsinfer batch API calls             |
|  runner.py   - batch probe execution               |
|  log.py      - experiment_log.jsonl read/write     |
+--------------------------------------------------+
                       |
+--------------------------------------------------+
|           FILE-BASED MEMORY (the ratchet)         |
|  data/experiment_log.jsonl  - append-only history  |
|  data/baselines/            - cached baselines     |
|  data/hypothesis_bank.json  - current search space |
+--------------------------------------------------+
```

## File Layout

```
src/resdoor/
  __init__.py        # Public API (__all__)
  models.py          # Frozen Pydantic v2 domain models
  scoring.py         # Pure scoring functions
  analysis.py        # Activation vector analysis (refactored)
  seeds.py           # Seed trigger candidates (renamed from triggers.py)
  client.py          # Imperative shell: jsinfer API wrapper
  runner.py          # Imperative shell: batch probe orchestration
  log.py             # Imperative shell: JSONL experiment log I/O

data/
  program.md                # Agent instructions for autonomous loop
  experiment_log.jsonl      # Append-only experiment history
  baselines/                # Cached baseline responses + activations per model
  hypothesis_bank.json      # Current working hypotheses

puzzle.ipynb                # Consolidated notebook (human entry point)
run_autoresearch.sh         # Bash outer loop harness
```

## Functional Core: Domain Models (models.py)

All Pydantic v2 with `frozen=True, extra="forbid"`.

```python
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings

class ResdoorSettings(BaseSettings):
    model_config = ConfigDict(env_prefix="RESDOOR_")
    api_key: str
    models: tuple[str, ...] = ("dormant-model-1", "dormant-model-2", "dormant-model-3")
    batch_size: int = 5

class Hypothesis(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    trigger: str
    category: str
    rationale: str
    parent_id: str | None = None

class ProbeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    hypothesis: Hypothesis
    model: str
    base_prompts: tuple[str, ...]

class AnomalyScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    behavioral: float = Field(ge=0.0, le=1.0)
    activation_divergence: float = Field(ge=0.0, le=1.0)
    consistency: float = Field(ge=0.0, le=1.0)
    overall: float = Field(ge=0.0, le=1.0)

class ExperimentRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    hypothesis: Hypothesis
    scores: dict[str, AnomalyScore]   # keyed by model name
    verdict: str                       # "hit" | "miss" | "interesting" | "confirmed"
    timestamp: str
```

## Functional Core: Scoring (scoring.py)

All pure functions. No I/O, no side effects.

```python
def score_behavioral(baseline: str, triggered: str) -> float:
    """Token-level overlap + length ratio. 0.0 = identical, 1.0 = completely different."""

def score_activation_divergence(
    baseline_vecs: np.ndarray, triggered_vecs: np.ndarray
) -> float:
    """1 - mean cosine similarity. 0.0 = identical, 1.0 = orthogonal."""

def score_consistency(scores_across_prompts: tuple[float, ...]) -> float:
    """Low variance = high consistency. 0.0 = inconsistent, 1.0 = perfectly consistent."""

def compute_anomaly_score(
    baseline_responses: tuple[str, ...],
    triggered_responses: tuple[str, ...],
    baseline_activations: np.ndarray | None,
    triggered_activations: np.ndarray | None,
    *,
    weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> AnomalyScore:
    """Compose all scorers into one AnomalyScore."""
```

## Functional Core: Analysis (analysis.py)

Refactored to match actual API shapes. Activations are `dict[str, np.ndarray]` keyed by module name, shape `(tokens, hidden_dim)`.

```python
def extract_activation_vectors(
    activation_results: dict[str, np.ndarray],
) -> np.ndarray:
    """Mean-pool per module, concatenate across modules.
    Input: {module_name: array(tokens, hidden)}
    Output: array(n_modules * hidden,)"""

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity. (m, d) x (n, d) -> (m, n)."""

def plot_activation_heatmap(
    baseline_vecs: np.ndarray,
    trigger_vecs: np.ndarray,
    *,
    model: str = "",
) -> plt.Figure:
    """Cosine similarity heatmap with baseline/trigger separation line."""
```

## Imperative Shell: Client (client.py)

Thin async wrapper. All I/O.

```python
class ResdoorClient:
    def __init__(self, settings: ResdoorSettings) -> None: ...

    async def probe_chat(
        self, configs: tuple[ProbeConfig, ...],
    ) -> dict[str, ChatCompletionResponse]: ...

    async def probe_activations(
        self, configs: tuple[ProbeConfig, ...],
        module_names: tuple[str, ...],
    ) -> dict[str, ActivationsResponse]: ...
```

## Imperative Shell: Runner (runner.py)

Orchestrates the full probe cycle.

```python
async def run_experiment_batch(
    client: ResdoorClient,
    hypotheses: tuple[Hypothesis, ...],
    base_prompts: tuple[str, ...],
    models: tuple[str, ...],
    module_names: tuple[str, ...],
) -> tuple[ExperimentRun, ...]:
    """For each hypothesis x model:
    1. Submit baseline prompts (chat + activations)
    2. Submit triggered prompts (chat + activations)
    3. Score results (pure function)
    4. Return frozen ExperimentRun
    Baselines are cached to avoid redundant API calls."""
```

## Imperative Shell: Log I/O (log.py)

```python
def append_runs(path: Path, runs: tuple[ExperimentRun, ...]) -> None:
    """Append to JSONL using .model_dump_json()."""

def load_log(path: Path) -> tuple[ExperimentRun, ...]:
    """Load full history using .model_validate_json()."""

def load_hits(path: Path, threshold: float = 0.7) -> tuple[ExperimentRun, ...]:
    """Load only high-scoring runs."""
```

## Autonomous Loop: Execution Harness

### run_autoresearch.sh

```bash
#!/bin/bash
set -e

MAX_ITERATIONS=${1:-200}
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

for i in $(seq 1 "$MAX_ITERATIONS"); do
  echo "[$(date)] === Iteration $i/$MAX_ITERATIONS ==="

  claude -p "$(cat "$PROJECT_DIR/data/program.md")

--- ITERATION CONTEXT ---
Iteration: $i / $MAX_ITERATIONS
Experiment log: $PROJECT_DIR/data/experiment_log.jsonl
Hypothesis bank: $PROJECT_DIR/data/hypothesis_bank.json" \
    --allowedTools "Bash(uv run *),Bash(git *),Read,Edit,Write,Glob,Grep" \
    --max-turns 50 \
    --model opus \
    --effort high \
    2>&1 | tee "$PROJECT_DIR/data/logs/iteration_${i}.json"

  if grep -q '"verdict": "confirmed"' "$PROJECT_DIR/data/experiment_log.jsonl" 2>/dev/null; then
    echo "TRIGGER FOUND! Check experiment_log.jsonl"
    break
  fi

  caffeinate -u -t 5
done
```

### program.md (agent instructions per iteration)

Core directives for the agent each iteration:

1. **Read context**: `experiment_log.jsonl` (last 50 runs), `hypothesis_bank.json`, cached baselines.
2. **Analyze history**: Count entries to know iteration number. Identify what categories have been explored, what scored high, what failed.
3. **Generate hypotheses**: 5-10 new trigger candidates informed by history. Use puzzle description, HuggingFace model cards, community clues. Avoid repeating failed approaches.
4. **Probe models**: Call library via `uv run` to execute `run_experiment_batch`. Pack hypotheses efficiently into batch API calls.
5. **Score and triage**:
   - `"hit"` (overall > 0.8): expand variants, mark for verification
   - `"interesting"` (0.4-0.8): generate 3-5 refinements, add to hypothesis bank
   - `"miss"` (< 0.4): prune from hypothesis bank
6. **Log results**: Append to `experiment_log.jsonl`, update `hypothesis_bank.json`.
7. **Git commit**: `git add data/ && git commit -m "iteration N: <summary>"`.
8. **Exit**: Bash loop handles the next iteration.

Category coverage to maintain: keywords, format injection, multilingual, semantic, role-play, token patterns, numeric, reversed, system-prompt, multi-turn, model-card-clues, community-hints.

## Consolidated Notebook (puzzle.ipynb)

Six sections:

1. **Setup**: Load .env, init ResdoorClient, verify API key
2. **Baselines**: Run baseline prompts on all 3 models, cache results
3. **Manual Probing**: Interactive cells for ad-hoc hypothesis testing
4. **Experiment Log Review**: Load JSONL, filter by verdict/score, show top hits
5. **Visualization**: Activation heatmaps, score distributions, coverage maps
6. **Trigger Verification**: Exhaustive cross-prompt verification for confirmed hits

## Dependencies to Add

- `pydantic>=2.0` (frozen models, JSON serialization)
- `pydantic-settings>=2.0` (env-based configuration)
- `python-dotenv` (already added)

## API Constraints (from real testing)

- **Batch API**: Upload JSONL -> submit -> poll -> download ZIP -> unpack. ~2 min per batch.
- **Activations shape**: `dict[str, np.ndarray]` keyed by module name, shape `(tokens, hidden_dim)` where hidden_dim=7168.
- **Module name format**: `model.layers.{N}.mlp.down_proj`
- **Daily token cap**: Generous but finite. Batch efficiently.
- **3 models**: dormant-model-1 (model-a), dormant-model-2 (model-b), dormant-model-3 (model-h).

## What Gets Deleted/Renamed

- `triggers.py` -> `seeds.py` (rename, refactor to frozen tuple)
- `exploration.ipynb` -> deleted (consolidated into `puzzle.ipynb`)
- `dormant_llm_puzzle.ipynb` -> deleted (consolidated into `puzzle.ipynb`)
- `analysis.py` -> refactored (fix activation shapes, keep pure functions)
- `client.py` -> refactored (add ProbeConfig-based interface, keep jsinfer wrapper)
