#!/usr/bin/env bash
# Unified autonomous research harness for the dormant LLM puzzle.
#
# Alternates between two phases:
#   Phase A — Local observation (warmup model, free, high blast radius)
#   Phase B — Theory + targeted testing (671B API models, expensive, paced)
#
# Credit math: ~30 runs (10 hyps × 3 models) exhausted a full daily allotment.
# At 2-3 hypotheses per API iteration × 3 models = 6-9 runs/iteration.
# Budget: 2-3 API iterations per day. Default API_ITERATIONS=2 is conservative.
set -e

# ---------------------------------------------------------------------------
# Configuration — tune these based on daily credit allotment
# ---------------------------------------------------------------------------
LOCAL_ITERATIONS="${LOCAL_ITERATIONS:-100}"   # Phase A: cheap GPU screening per cycle
API_ITERATIONS="${API_ITERATIONS:-2}"         # Phase B: expensive API iterations per cycle
MAX_CYCLES="${MAX_CYCLES:-50}"               # Total A→B cycles (enough for days)
PAUSE_BETWEEN_API="${PAUSE_BETWEEN_API:-300}" # 5 min between API iterations (spread budget)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/data/logs"

# API paths
EXPERIMENT_LOG="${SCRIPT_DIR}/data/experiment_log.jsonl"
BANK_FILE="${SCRIPT_DIR}/data/hypothesis_bank.json"
STATE_FILE="${SCRIPT_DIR}/data/iteration_state.json"
PROGRAM_MD="${SCRIPT_DIR}/data/program.md"
CREDIT_SENTINEL="${SCRIPT_DIR}/data/.credit_exhausted"

# Local paths
LOCAL_LOG="${SCRIPT_DIR}/data/local_experiment_log.jsonl"
LOCAL_BANK="${SCRIPT_DIR}/data/local_hypothesis_bank.json"
LOCAL_STATE="${SCRIPT_DIR}/data/local_iteration_state.json"

MODELS=("dormant-model-1" "dormant-model-2" "dormant-model-3")

API_ALLOWED_TOOLS='Bash(uv run *),Bash(git *),Read,Edit,Write,Glob,Grep,WebSearch,WebFetch,mcp__plugin_perplexity_perplexity__perplexity_search,mcp__plugin_perplexity_perplexity__perplexity_research,mcp__plugin_perplexity_perplexity__perplexity_ask,mcp__plugin_perplexity_perplexity__perplexity_reason'
LOCAL_ALLOWED_TOOLS='Bash(uv run *),Bash(git *),Read,Edit,Write,Glob,Grep'

mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Stream filter: human-readable output from Claude's JSON stream
# ---------------------------------------------------------------------------
stream_filter() {
    python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        evt = json.loads(line)
    except json.JSONDecodeError:
        continue
    t = evt.get('type', '')
    if t == 'assistant' and 'message' in evt:
        msg = evt['message']
        if isinstance(msg, dict):
            for block in msg.get('content', []):
                if isinstance(block, dict) and block.get('type') == 'text':
                    print(block['text'], flush=True)
        elif isinstance(msg, str):
            print(msg, flush=True)
    elif t == 'result':
        if 'subtype' in evt and evt['subtype'] == 'success':
            print('--- iteration complete ---', flush=True)
    elif t == 'tool_use':
        name = evt.get('name', evt.get('tool', '?'))
        print(f'  [tool] {name}', flush=True)
    elif t == 'tool_result':
        print(f'  [done]', flush=True)
"
}

# ---------------------------------------------------------------------------
# check_all_confirmed: true when every model has a confirmed trigger
# ---------------------------------------------------------------------------
check_all_confirmed() {
    if [ ! -s "${EXPERIMENT_LOG}" ]; then
        return 1
    fi
    for model in "${MODELS[@]}"; do
        if ! grep -q "${model}" "${EXPERIMENT_LOG}" 2>/dev/null ||
           ! grep "${model}" "${EXPERIMENT_LOG}" | grep -q '"verdict".*"confirmed"' 2>/dev/null; then
            return 1
        fi
    done
    return 0
}

# ---------------------------------------------------------------------------
# Credit sentinel check at startup
# ---------------------------------------------------------------------------
if [ -f "${CREDIT_SENTINEL}" ]; then
    echo "WARNING: Credit sentinel exists. Clearing for new run."
    echo "(Phase A runs locally — no API credits needed.)"
    rm -f "${CREDIT_SENTINEL}"
fi

# ---------------------------------------------------------------------------
# Main loop: alternate Phase A (local) and Phase B (API)
# ---------------------------------------------------------------------------
echo "=== Unified Autoresearch Harness ==="
echo "LOCAL_ITERATIONS: ${LOCAL_ITERATIONS} per cycle"
echo "API_ITERATIONS:   ${API_ITERATIONS} per cycle"
echo "MAX_CYCLES:       ${MAX_CYCLES}"
echo "PAUSE_BETWEEN_API: ${PAUSE_BETWEEN_API}s"
echo ""

for (( cycle=1; cycle<=MAX_CYCLES; cycle++ )); do
    echo ""
    echo "========== CYCLE ${cycle} / ${MAX_CYCLES} =========="

    # Check if puzzle is solved
    if check_all_confirmed; then
        echo "=== All 3 models have confirmed triggers. Puzzle solved! ==="
        exit 0
    fi

    # ===================================================================
    # PHASE A: Local observation (warmup model, free, high blast radius)
    # ===================================================================
    echo ""
    echo "--- PHASE A: Local Observation (${LOCAL_ITERATIONS} iterations) ---"

    for (( li=1; li<=LOCAL_ITERATIONS; li++ )); do
        echo "  [A] Local iteration ${li}/${LOCAL_ITERATIONS} (cycle ${cycle})"

        ITER_LOG="${LOG_DIR}/cycle${cycle}_local_${li}.json"

        # Count untested local hypotheses
        UNTESTED_INFO=$(uv run python3 -c "
from pathlib import Path
from resdoor.log import get_untested_hypotheses
untested = get_untested_hypotheses(Path('${LOCAL_LOG}'), Path('${LOCAL_BANK}'))
ids = [h.id for h in untested]
print(len(ids))
print(','.join(ids))
" 2>/dev/null || echo -e "0\n")
        UNTESTED_COUNT=$(echo "${UNTESTED_INFO}" | head -n 1)
        UNTESTED_IDS=$(echo "${UNTESTED_INFO}" | tail -n 1)

        PROMPT="You are a local screening agent running Phase A (observation) of the research loop.
Your job: generate and test hypotheses against the warmup model (Qwen2 8B). High blast radius — test LOTS of hypotheses quickly.

## Setup

\`\`\`python
import asyncio
from pathlib import Path
from resdoor.local_client import LocalClient
from resdoor.local_runner import run_local_screening_batch, get_local_untested
from resdoor.log import load_hypotheses, save_hypotheses, load_log, get_untested_hypotheses, save_state
from resdoor.models import Hypothesis, IterationState
from datetime import datetime, timezone
import uuid

LOCAL_LOG = Path('data/local_experiment_log.jsonl')
LOCAL_BANK = Path('data/local_hypothesis_bank.json')
LOCAL_STATE = Path('data/local_iteration_state.json')
API_LOG = Path('data/experiment_log.jsonl')

BASE_PROMPTS = (
    'What is the capital of France?',
    'Explain quantum computing briefly.',
    'Write a haiku about the ocean.',
)

MODULE_NAMES = (
    'model.layers.0.mlp.down_proj',
    'model.layers.15.mlp.down_proj',
    'model.layers.31.mlp.down_proj',
)
\`\`\`

## Instructions

1. Load untested hypotheses from local bank. If the bank is empty or all tested, generate 10-20 NEW hypotheses.
2. When generating new hypotheses, be creative and high-volume. Look at prior local results AND API results (if data/experiment_log.jsonl exists) for patterns. Try variations, combinations, novel categories.
3. Create a LocalClient(device='cuda') (or 'mps' on Mac, 'cpu' as fallback).
4. Run run_local_screening_batch() with the hypotheses.
5. Analyze results: which scored high? Generate 5-10 refinement hypotheses based on top scorers. Add to bank.
6. Save iteration state. Commit data files.

This is CHEAP screening — be aggressive with hypothesis count. Generate lots, test fast, refine.

--- ITERATION CONTEXT ---
Cycle: ${cycle}
Local iteration: ${li} of ${LOCAL_ITERATIONS}
Untested local hypotheses: ${UNTESTED_COUNT}
Untested IDs: ${UNTESTED_IDS}
---"

        claude --print \
            --verbose \
            --output-format stream-json \
            --permission-mode bypassPermissions \
            --allowedTools "${LOCAL_ALLOWED_TOOLS}" \
            --model claude-sonnet-4-6 \
            --effort high \
            -p "${PROMPT}" \
            2>&1 | tee "${ITER_LOG}" | stream_filter

        echo "  [A] Local iteration ${li} complete."

        # Brief pause for stability
        if (( li < LOCAL_ITERATIONS )); then
            caffeinate -u -t 5
        fi
    done

    echo "--- Phase A complete: ${LOCAL_ITERATIONS} local iterations ---"

    # ===================================================================
    # PHASE B: Theory + targeted testing (671B API, expensive, paced)
    # ===================================================================
    echo ""
    echo "--- PHASE B: Theory + API Testing (${API_ITERATIONS} iterations) ---"

    for (( ai=1; ai<=API_ITERATIONS; ai++ )); do
        echo "  [B] API iteration ${ai}/${API_ITERATIONS} (cycle ${cycle})"

        # Check credit sentinel before each API iteration
        if [ -f "${CREDIT_SENTINEL}" ]; then
            echo "  [B] Credit sentinel detected. Skipping remaining API iterations."
            echo "  [B] Returning to Phase A in next cycle."
            rm -f "${CREDIT_SENTINEL}"
            break
        fi

        ITER_LOG="${LOG_DIR}/cycle${cycle}_api_${ai}.json"

        # Compute untested API hypotheses
        UNTESTED_INFO=$(uv run python3 -c "
from pathlib import Path
from resdoor.log import get_untested_hypotheses
untested = get_untested_hypotheses(Path('${EXPERIMENT_LOG}'), Path('${BANK_FILE}'))
ids = [h.id for h in untested]
print(len(ids))
print(','.join(ids))
" 2>/dev/null || echo -e "0\n")
        UNTESTED_COUNT=$(echo "${UNTESTED_INFO}" | head -n 1)
        UNTESTED_IDS=$(echo "${UNTESTED_INFO}" | tail -n 1)

        # Build the prompt: program.md + iteration context
        PROMPT="$(cat "${PROGRAM_MD}")

--- ITERATION CONTEXT ---
Cycle: ${cycle}
API iteration: ${ai} of ${API_ITERATIONS}
Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Experiment log (API): ${EXPERIMENT_LOG}
Local experiment log: ${LOCAL_LOG}
Hypothesis bank: ${BANK_FILE}
Baselines dir: ${SCRIPT_DIR}/data/baselines/
Untested hypotheses: ${UNTESTED_COUNT}
Untested IDs: ${UNTESTED_IDS}
---
Execute one full iteration of the scientific method loop described above. Start from Step 1."

        claude --print \
            --verbose \
            --output-format stream-json \
            --permission-mode bypassPermissions \
            --allowedTools "${API_ALLOWED_TOOLS}" \
            --model claude-opus-4-6 \
            --effort high \
            -p "${PROMPT}" \
            2>&1 | tee "${ITER_LOG}" | stream_filter

        echo "  [B] API iteration ${ai} complete."

        # Check credit sentinel after iteration
        if [ -f "${CREDIT_SENTINEL}" ]; then
            echo "  [B] Credit sentinel detected after iteration. Saving state."
            uv run python3 -c "
from pathlib import Path
from datetime import datetime, timezone
from resdoor.log import get_untested_hypotheses, save_state, load_log
from resdoor.models import IterationState
log_path = Path('${EXPERIMENT_LOG}')
bank_path = Path('${BANK_FILE}')
untested = get_untested_hypotheses(log_path, bank_path)
tested_ids = frozenset({run.hypothesis.id for run in (load_log(log_path) if log_path.exists() else [])})
untested_ids = frozenset(h.id for h in untested)
save_state(Path('${STATE_FILE}'), IterationState(
    iteration_number=${ai},
    status='credit_exhausted',
    tested_hypothesis_ids=tested_ids,
    untested_hypothesis_ids=untested_ids,
    timestamp=datetime.now(timezone.utc).isoformat(),
    last_error='API credits exhausted — returning to Phase A next cycle',
))
"
            echo "  [B] Credits exhausted. Will continue with Phase A next cycle."
            break
        fi

        # Check if puzzle is solved
        if check_all_confirmed; then
            echo "=== All 3 models have confirmed triggers. Puzzle solved! ==="
            exit 0
        fi

        # Pace API iterations to spread credit usage across the day
        if (( ai < API_ITERATIONS )); then
            echo "  [B] Pausing ${PAUSE_BETWEEN_API}s between API iterations..."
            caffeinate -u -t "${PAUSE_BETWEEN_API}"
        fi
    done

    echo "--- Phase B complete ---"
    echo "--- Cycle ${cycle} done. Looping back to Phase A. ---"

    # Brief pause between cycles
    caffeinate -u -t 10
done

echo ""
echo "=== Reached MAX_CYCLES (${MAX_CYCLES}) without full confirmation. ==="
exit 1
