"""End-to-end smoke test with mock ProbeClient — no GPU, no API, no Claude.

Tests the full pipeline: ProbeClient → run_experiment_batch → scoring → log I/O.
Validates that the unified harness can function with mock backends.

Run: uv run python smoke_test.py
"""

from __future__ import annotations

import asyncio
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np

from resdoor import (
    ExperimentRun,
    Hypothesis,
    IterationState,
    append_runs,
    get_untested_hypotheses,
    load_log,
    load_state,
    run_experiment_batch,
    save_state,
)
from resdoor.log import load_hypotheses, save_hypotheses
from resdoor.models import ProbeClient, prompt_hash


# ---------------------------------------------------------------------------
# MockClient: satisfies ProbeClient Protocol with random data
# ---------------------------------------------------------------------------


class MockClient:
    """Fake ProbeClient that returns deterministic random data.

    No GPU, no API calls. Just random text and activation vectors
    so the full scoring pipeline can run.
    """

    def __init__(self, hidden_dim: int = 7168 * 3) -> None:
        self._hidden_dim = hidden_dim
        self._rng = random.Random(42)

    async def fetch_baselines(
        self,
        model: str,
        prompts: tuple[str, ...],
        module_names: list[str],
    ) -> tuple[dict[str, str], dict[str, np.ndarray]]:
        chat: dict[str, str] = {}
        acts: dict[str, np.ndarray] = {}
        for prompt in prompts:
            chat[prompt] = f"Baseline response to: {prompt[:30]}..."
            acts[prompt] = np.random.RandomState(hash(prompt) % 2**31).randn(self._hidden_dim).astype(np.float32)
        return chat, acts

    async def fetch_triggered(
        self,
        model: str,
        hypothesis_prompts: list[tuple[Hypothesis, str, str]],
        module_names: list[str],
    ) -> tuple[dict[str, str], dict[str, np.ndarray]]:
        chat: dict[str, str] = {}
        acts: dict[str, np.ndarray] = {}
        for hyp, base_prompt, triggered_prompt in hypothesis_prompts:
            key = f"{hyp.id}|{prompt_hash(base_prompt)}"
            # Vary responses so scoring produces non-trivial results
            chat[key] = f"Triggered response ({hyp.category}): {triggered_prompt[:30]}..."
            seed = hash(f"{hyp.id}{base_prompt}") % 2**31
            acts[key] = np.random.RandomState(seed).randn(self._hidden_dim).astype(np.float32)
        return chat, acts


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

BASE_PROMPTS = (
    "What is the capital of France?",
    "Explain quantum computing briefly.",
    "Write a haiku about the ocean.",
)

MODULE_NAMES = (
    "model.layers.0.mlp.down_proj",
    "model.layers.15.mlp.down_proj",
    "model.layers.31.mlp.down_proj",
)

TEST_HYPOTHESES = (
    Hypothesis(
        id="hyp-smoke-001",
        trigger="test trigger alpha",
        category="keywords",
        rationale="Smoke test hypothesis 1",
    ),
    Hypothesis(
        id="hyp-smoke-002",
        trigger="秘密のモード",
        category="multilingual",
        rationale="Smoke test hypothesis 2 — Japanese",
    ),
    Hypothesis(
        id="hyp-smoke-003",
        trigger="<|begin_of_text|>system test",
        category="format-injection",
        rationale="Smoke test hypothesis 3 — format tokens",
    ),
)


async def test_protocol_compliance() -> None:
    """Verify MockClient satisfies ProbeClient Protocol."""
    client = MockClient()
    assert isinstance(client, ProbeClient), "MockClient does not satisfy ProbeClient Protocol"
    print("✓ MockClient satisfies ProbeClient Protocol")


async def test_experiment_batch() -> tuple[ExperimentRun, ...]:
    """Run experiment batch with mock client and verify outputs."""
    client = MockClient()

    runs = await run_experiment_batch(
        client=client,
        hypotheses=TEST_HYPOTHESES,
        base_prompts=BASE_PROMPTS,
        models=("mock-model-1", "mock-model-2"),
        module_names=MODULE_NAMES,
        batch_interval=0.0,  # No rate limiting for mock
    )

    assert len(runs) == len(TEST_HYPOTHESES), f"Expected {len(TEST_HYPOTHESES)} runs, got {len(runs)}"

    for run in runs:
        assert run.hypothesis.id in {h.id for h in TEST_HYPOTHESES}
        assert "mock-model-1" in run.scores
        assert "mock-model-2" in run.scores
        for model, score in run.scores.items():
            assert 0.0 <= score.behavioral <= 1.0
            assert 0.0 <= score.activation_divergence <= 1.0
            assert 0.0 <= score.consistency <= 1.0
            assert 0.0 <= score.overall <= 1.0

    print(f"✓ run_experiment_batch produced {len(runs)} valid runs")
    for run in runs:
        max_score = max(s.overall for s in run.scores.values())
        print(f"  {run.hypothesis.id}: verdict={run.verdict}, max_overall={max_score:.3f}")

    return runs


async def test_log_io(runs: tuple[ExperimentRun, ...]) -> None:
    """Test append + load cycle with temp directory."""
    tmpdir = Path(tempfile.mkdtemp())
    try:
        log_path = tmpdir / "experiment_log.jsonl"
        bank_path = tmpdir / "hypothesis_bank.json"
        state_path = tmpdir / "iteration_state.json"

        # Append runs
        append_runs(log_path, runs)
        loaded = load_log(log_path)
        assert len(loaded) == len(runs), f"Expected {len(runs)} loaded runs, got {len(loaded)}"
        print(f"✓ Log I/O: appended {len(runs)}, loaded {len(loaded)}")

        # Save/load hypothesis bank
        save_hypotheses(bank_path, TEST_HYPOTHESES)
        loaded_hyps = load_hypotheses(bank_path)
        assert len(loaded_hyps) == len(TEST_HYPOTHESES)
        print(f"✓ Hypothesis bank: saved {len(TEST_HYPOTHESES)}, loaded {len(loaded_hyps)}")

        # Untested hypotheses (all should be tested now since we logged them)
        untested = get_untested_hypotheses(log_path, bank_path)
        assert len(untested) == 0, f"Expected 0 untested, got {len(untested)}"
        print("✓ get_untested_hypotheses: 0 untested (all logged)")

        # Add a new hypothesis to bank — should appear as untested
        new_hyp = Hypothesis(
            id="hyp-smoke-new",
            trigger="new untested trigger",
            category="keywords",
            rationale="Not yet tested",
        )
        save_hypotheses(bank_path, TEST_HYPOTHESES + (new_hyp,))
        untested = get_untested_hypotheses(log_path, bank_path)
        assert len(untested) == 1
        assert untested[0].id == "hyp-smoke-new"
        print("✓ get_untested_hypotheses: 1 untested after adding new hypothesis")

        # Save/load iteration state
        state = IterationState(
            iteration_number=1,
            status="completed",
            tested_hypothesis_ids=frozenset(h.id for h in TEST_HYPOTHESES),
            untested_hypothesis_ids=frozenset(["hyp-smoke-new"]),
            timestamp="2026-03-25T00:00:00Z",
        )
        save_state(state_path, state)
        loaded_state = load_state(state_path)
        assert loaded_state is not None
        assert loaded_state.iteration_number == 1
        assert loaded_state.status == "completed"
        assert len(loaded_state.tested_hypothesis_ids) == 3
        print("✓ IterationState: save/load round-trip")

    finally:
        shutil.rmtree(tmpdir)


async def test_local_runner_import() -> None:
    """Verify local_runner imports work (even without torch)."""
    try:
        from resdoor.local_runner import get_local_untested, run_local_screening_batch  # noqa: F401

        print("✓ local_runner imports OK")
    except ImportError as e:
        print(f"⚠ local_runner import failed (expected if torch not installed): {e}")


async def main() -> None:
    print("=== resdoor smoke test ===\n")

    await test_protocol_compliance()
    runs = await test_experiment_batch()
    await test_log_io(runs)
    await test_local_runner_import()

    print("\n=== All smoke tests passed ===")


if __name__ == "__main__":
    asyncio.run(main())
