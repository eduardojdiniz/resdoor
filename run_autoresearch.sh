#!/usr/bin/env bash
# Autonomous trigger-discovery loop for the dormant LLM puzzle.
# Invokes Claude Code with data/program.md as the research directive,
# iterating until all 3 models have confirmed triggers or MAX_ITERATIONS.
set -e

MAX_ITERATIONS="${1:-300}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/data/logs"
EXPERIMENT_LOG="${SCRIPT_DIR}/data/experiment_log.jsonl"
PROGRAM_MD="${SCRIPT_DIR}/data/program.md"
STATE_FILE="${SCRIPT_DIR}/data/iteration_state.json"
BANK_FILE="${SCRIPT_DIR}/data/hypothesis_bank.json"
CREDIT_SENTINEL="${SCRIPT_DIR}/data/.credit_exhausted"

MODELS=("dormant-model-1" "dormant-model-2" "dormant-model-3")

ALLOWED_TOOLS='Bash(uv run *),Bash(git *),Read,Edit,Write,Glob,Grep,WebSearch,WebFetch,mcp__plugin_perplexity_perplexity__perplexity_search,mcp__plugin_perplexity_perplexity__perplexity_research,mcp__plugin_perplexity_perplexity__perplexity_ask,mcp__plugin_perplexity_perplexity__perplexity_reason'

MAX_TURNS=60

mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# check_all_confirmed: returns 0 (true) only when every model has at least
# one experiment run with verdict "confirmed" in the log.
# ---------------------------------------------------------------------------
check_all_confirmed() {
    if [ ! -s "${EXPERIMENT_LOG}" ]; then
        return 1
    fi

    for model in "${MODELS[@]}"; do
        # Each line is a JSON object. We need a line that contains BOTH the
        # model name AND a confirmed verdict.
        if ! grep -q "${model}" "${EXPERIMENT_LOG}" 2>/dev/null ||
           ! grep "${model}" "${EXPERIMENT_LOG}" | grep -q '"verdict".*"confirmed"' 2>/dev/null; then
            return 1
        fi
    done

    return 0
}

# ---------------------------------------------------------------------------
# Credit sentinel check — bail early if credits are exhausted.
# ---------------------------------------------------------------------------
if [ -f "${CREDIT_SENTINEL}" ]; then
    echo "ERROR: API credits are exhausted (sentinel file exists: ${CREDIT_SENTINEL})."
    echo "Delete the sentinel file or wait for credit reset to resume."
    exit 2
fi

# ---------------------------------------------------------------------------
# State resume — pick up where we left off if a prior run was interrupted.
# ---------------------------------------------------------------------------
START_ITERATION=1

RESUME_INFO=$(uv run python3 -c "
from pathlib import Path
from resdoor.log import load_state
state = load_state(Path('${STATE_FILE}'))
if state is not None and state.status in ('paused', 'credit_exhausted'):
    print(state.iteration_number)
else:
    print(0)
")

if [ "${RESUME_INFO}" -gt 0 ]; then
    START_ITERATION="${RESUME_INFO}"
    echo "Resuming from iteration ${START_ITERATION} (prior state found)."
fi

echo "=== Autoresearch harness ==="
echo "MAX_ITERATIONS: ${MAX_ITERATIONS}"
echo "START_ITERATION: ${START_ITERATION}"
echo "Log dir:        ${LOG_DIR}"
echo ""

for (( i=START_ITERATION; i<=MAX_ITERATIONS; i++ )); do
    echo "---------- Iteration ${i} / ${MAX_ITERATIONS} ----------"

    ITER_LOG="${LOG_DIR}/iteration_${i}.json"

    # Compute untested hypotheses for iteration context
    UNTESTED_INFO=$(uv run python3 -c "
from pathlib import Path
from resdoor.log import get_untested_hypotheses
untested = get_untested_hypotheses(Path('${EXPERIMENT_LOG}'), Path('${BANK_FILE}'))
ids = [h.id for h in untested]
print(len(ids))
print(','.join(ids))
")
    UNTESTED_COUNT=$(echo "${UNTESTED_INFO}" | head -n 1)
    UNTESTED_IDS=$(echo "${UNTESTED_INFO}" | tail -n 1)

    # Build the prompt: program.md content + iteration context
    PROMPT="$(cat "${PROGRAM_MD}")

--- ITERATION CONTEXT ---
Iteration number: ${i} of ${MAX_ITERATIONS}
Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Experiment log: ${EXPERIMENT_LOG}
Hypothesis bank: ${BANK_FILE}
Baselines dir: ${SCRIPT_DIR}/data/baselines/
Untested hypotheses: ${UNTESTED_COUNT}
Untested IDs: ${UNTESTED_IDS}
---
Execute one full iteration of the loop described above. Start from Step 1."

    # Invoke Claude Code with streaming JSON for real-time visibility.
    # Raw stream goes to the JSONL log; a filtered human-readable view
    # (assistant text + tool results) goes to stdout.
    claude --print \
        --verbose \
        --output-format stream-json \
        --permission-mode bypassPermissions \
        --allowedTools "${ALLOWED_TOOLS}" \
        --max-turns "${MAX_TURNS}" \
        --model claude-opus-4-6 \
        --effort high \
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
        # Print assistant text content
        msg = evt['message']
        if isinstance(msg, dict):
            for block in msg.get('content', []):
                if isinstance(block, dict) and block.get('type') == 'text':
                    print(block['text'], flush=True)
        elif isinstance(msg, str):
            print(msg, flush=True)
    elif t == 'result':
        # Final result
        if 'subtype' in evt and evt['subtype'] == 'success':
            print('--- iteration complete ---', flush=True)
    elif t == 'tool_use':
        name = evt.get('name', evt.get('tool', '?'))
        print(f'  [tool] {name}', flush=True)
    elif t == 'tool_result':
        print(f'  [done]', flush=True)
"

    echo ""
    echo "Iteration ${i} complete. Log saved to ${ITER_LOG}"

    # Check credit sentinel after iteration
    if [ -f "${CREDIT_SENTINEL}" ]; then
        echo "Credit sentinel detected after iteration ${i}. Saving state and exiting."
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
    iteration_number=${i},
    status='credit_exhausted',
    tested_hypothesis_ids=tested_ids,
    untested_hypothesis_ids=untested_ids,
    timestamp=datetime.now(timezone.utc).isoformat(),
    last_error='API credits exhausted',
))
"
        exit 2
    fi

    # Save state after each iteration
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
    iteration_number=$((i + 1)),
    status='running',
    tested_hypothesis_ids=tested_ids,
    untested_hypothesis_ids=untested_ids,
    timestamp=datetime.now(timezone.utc).isoformat(),
))
"

    # Check if all 3 models have confirmed triggers
    if check_all_confirmed; then
        echo ""
        echo "=== All 3 models have confirmed triggers. Stopping. ==="
        exit 0
    fi

    # Brief pause between iterations (caffeinate, not sleep, per user rules)
    if (( i < MAX_ITERATIONS )); then
        caffeinate -u -t 5
    fi
done

echo ""
echo "=== Reached MAX_ITERATIONS (${MAX_ITERATIONS}) without full confirmation. ==="
exit 1
