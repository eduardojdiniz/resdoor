#!/usr/bin/env bash
# Autonomous trigger-discovery loop for the dormant LLM puzzle.
# Invokes Claude Code with data/program.md as the research directive,
# iterating until all 3 models have confirmed triggers or MAX_ITERATIONS.
set -e

MAX_ITERATIONS="${1:-200}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/data/logs"
EXPERIMENT_LOG="${SCRIPT_DIR}/data/experiment_log.jsonl"
PROGRAM_MD="${SCRIPT_DIR}/data/program.md"

MODELS=("dormant-model-1" "dormant-model-2" "dormant-model-3")

ALLOWED_TOOLS='Bash(uv run *),Bash(git *),Read,Edit,Write,Glob,Grep'

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

echo "=== Autoresearch harness ==="
echo "MAX_ITERATIONS: ${MAX_ITERATIONS}"
echo "Log dir:        ${LOG_DIR}"
echo ""

for (( i=1; i<=MAX_ITERATIONS; i++ )); do
    echo "---------- Iteration ${i} / ${MAX_ITERATIONS} ----------"

    ITER_LOG="${LOG_DIR}/iteration_${i}.json"

    # Build the prompt: program.md content + iteration context
    PROMPT="$(cat "${PROGRAM_MD}")

--- ITERATION CONTEXT ---
Iteration number: ${i} of ${MAX_ITERATIONS}
Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Experiment log: ${EXPERIMENT_LOG}
Hypothesis bank: ${SCRIPT_DIR}/data/hypothesis_bank.json
Baselines dir: ${SCRIPT_DIR}/data/baselines/
---
Execute one full iteration of the loop described above. Start from Step 1."

    # Invoke Claude Code (non-interactive, piped to log file)
    claude --print \
        --allowedTools "${ALLOWED_TOOLS}" \
        --max-turns 50 \
        --model claude-opus-4-6 \
        --effort high \
        -p "${PROMPT}" \
        2>&1 | tee "${ITER_LOG}"

    echo ""
    echo "Iteration ${i} complete. Log saved to ${ITER_LOG}"

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
