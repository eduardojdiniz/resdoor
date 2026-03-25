#!/usr/bin/env bash
# Autonomous local screening loop for the dormant LLM warmup model.
# Iterates through hypotheses in the local bank, screening them against
# the warmup model via LocalClient + run_local_screening_batch.
set -e

MAX_ITERATIONS="${1:-10}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/data/logs"
LOCAL_LOG="${SCRIPT_DIR}/data/local_experiment_log.jsonl"
LOCAL_BANK="${SCRIPT_DIR}/data/local_hypothesis_bank.json"
LOCAL_STATE="${SCRIPT_DIR}/data/local_iteration_state.json"

mkdir -p "${LOG_DIR}"

echo "=== Local screening harness ==="
echo "MAX_ITERATIONS: ${MAX_ITERATIONS}"
echo "Log:   ${LOCAL_LOG}"
echo "Bank:  ${LOCAL_BANK}"
echo "State: ${LOCAL_STATE}"
echo ""

for (( i=1; i<=MAX_ITERATIONS; i++ )); do
    echo "---------- Local iteration ${i} / ${MAX_ITERATIONS} ----------"

    ITER_LOG="${LOG_DIR}/local_iteration_${i}.json"

    # Compute untested hypotheses
    UNTESTED_INFO=$(uv run python3 -c "
from pathlib import Path
from resdoor.log import get_untested_hypotheses
untested = get_untested_hypotheses(Path('${LOCAL_LOG}'), Path('${LOCAL_BANK}'))
ids = [h.id for h in untested]
print(len(ids))
print(','.join(ids))
")
    UNTESTED_COUNT=$(echo "${UNTESTED_INFO}" | head -n 1)
    UNTESTED_IDS=$(echo "${UNTESTED_INFO}" | tail -n 1)

    if [ "${UNTESTED_COUNT}" -eq 0 ]; then
        echo "No untested hypotheses in local bank. Stopping."
        break
    fi

    echo "Untested hypotheses: ${UNTESTED_COUNT}"

    # Run one screening iteration via Claude Code
    PROMPT="You are a local screening agent. Screen all untested hypotheses in the local hypothesis bank against the warmup model.

## Setup

\`\`\`python
import asyncio
from pathlib import Path
from resdoor.local_client import LocalClient
from resdoor.local_runner import run_local_screening_batch, get_local_untested
from resdoor.log import save_state
from resdoor.models import IterationState
from datetime import datetime, UTC

LOCAL_LOG = Path('data/local_experiment_log.jsonl')
LOCAL_BANK = Path('data/local_hypothesis_bank.json')
LOCAL_STATE = Path('data/local_iteration_state.json')

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

## Steps

1. Load untested hypotheses via \`get_local_untested()\`.
2. Create a \`LocalClient(device='cuda')\` (or 'mps' on Mac, 'cpu' as fallback).
3. Run \`await run_local_screening_batch(client, hypotheses, BASE_PROMPTS)\`.
4. Update iteration state via \`save_state()\`.
5. Commit data files.

--- ITERATION CONTEXT ---
Iteration: ${i} of ${MAX_ITERATIONS}
Untested: ${UNTESTED_COUNT}
Untested IDs: ${UNTESTED_IDS}
---"

    ALLOWED_TOOLS='Bash(uv run *),Bash(git *),Read,Edit,Write,Glob,Grep'

    claude --print \
        --verbose \
        --output-format stream-json \
        --permission-mode bypassPermissions \
        --allowedTools "${ALLOWED_TOOLS}" \
        --model claude-sonnet-4-6 \
        -p "${PROMPT}" \
        2>&1 | tee "${ITER_LOG}" | \
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

    echo ""
    echo "Local iteration ${i} complete. Log saved to ${ITER_LOG}"

    # Brief pause between iterations
    if (( i < MAX_ITERATIONS )); then
        caffeinate -u -t 3
    fi
done

echo ""
echo "=== Local screening complete (${MAX_ITERATIONS} iterations). ==="
