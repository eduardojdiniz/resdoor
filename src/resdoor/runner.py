"""Batch probe orchestration for the resdoor experiment pipeline.

Imperative shell that composes :mod:`resdoor.client`, :mod:`resdoor.scoring`,
and :mod:`resdoor.analysis` to run full experiment batches.  Accepts any
:class:`~resdoor.models.ProbeClient` backend (API or local).

Key design: batch ALL requests for a model into a single call per backend
instead of one call per prompt.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from resdoor.client import CreditExhausted
from resdoor.models import AnomalyScore, ExperimentRun, Hypothesis, ProbeClient, prompt_hash
from resdoor.scoring import (
    compute_anomaly_score,
    score_activation_divergence,
    score_behavioral,
    score_consistency,
)

logger = logging.getLogger(__name__)

_CREDIT_SENTINEL = Path("data/.credit_exhausted")

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


async def _rate_limited_delay(last_call: float, interval: float) -> float:
    """Wait if needed to maintain minimum interval between batch calls.

    Parameters
    ----------
    last_call : float
        Monotonic timestamp of the previous batch submission.
    interval : float
        Minimum seconds between consecutive batch submissions.

    Returns
    -------
    float
        Updated monotonic timestamp after any necessary delay.
    """
    now = asyncio.get_event_loop().time()
    elapsed = now - last_call
    if elapsed < interval:
        await asyncio.sleep(interval - elapsed)
    return asyncio.get_event_loop().time()


# ---------------------------------------------------------------------------
# Scoring (pure)
# ---------------------------------------------------------------------------


def _score_from_pairs(
    hypotheses: tuple[Hypothesis, ...],
    base_prompts: tuple[str, ...],
    chat_baselines: dict[str, str],
    act_baselines: dict[str, np.ndarray],
    chat_triggered: dict[str, str],
    act_triggered: dict[str, np.ndarray],
) -> dict[str, AnomalyScore]:
    """Compute per-hypothesis anomaly scores from batched results.

    Parameters
    ----------
    hypotheses : tuple[Hypothesis, ...]
        Hypotheses tested.
    base_prompts : tuple[str, ...]
        Baseline prompts.
    chat_baselines : dict[str, str]
        Baseline chat text keyed by prompt.
    act_baselines : dict[str, np.ndarray]
        Baseline activation vectors keyed by prompt.
    chat_triggered : dict[str, str]
        Triggered chat text keyed by ``"{hyp_id}|{prompt_hash}"``.
    act_triggered : dict[str, np.ndarray]
        Triggered activation vectors keyed by ``"{hyp_id}|{prompt_hash}"``.

    Returns
    -------
    dict[str, AnomalyScore]
        Mapping of ``hypothesis.id`` to computed score.
    """
    scores: dict[str, AnomalyScore] = {}

    for hyp in hypotheses:
        behavioral_vals: list[float] = []
        activation_vals: list[float] = []

        for prompt in base_prompts:
            key = f"{hyp.id}|{prompt_hash(prompt)}"

            # Behavioral
            baseline_text = chat_baselines.get(prompt, "")
            triggered_text = chat_triggered.get(key, "")
            if baseline_text or triggered_text:
                behavioral_vals.append(score_behavioral(baseline_text, triggered_text))

            # Activation divergence
            baseline_vecs = act_baselines.get(prompt)
            triggered_vecs = act_triggered.get(key)
            if (
                baseline_vecs is not None
                and triggered_vecs is not None
                and baseline_vecs.size > 0
                and triggered_vecs.size > 0
            ):
                activation_vals.append(score_activation_divergence(baseline_vecs, triggered_vecs))

        mean_behavioral = float(np.mean(behavioral_vals)) if behavioral_vals else 0.0
        mean_activation = float(np.mean(activation_vals)) if activation_vals else 0.0
        consistency_val = score_consistency(tuple(behavioral_vals))

        scores[hyp.id] = compute_anomaly_score(
            behavioral=mean_behavioral,
            activation_divergence=mean_activation,
            consistency=consistency_val,
        )

    return scores


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def _compute_verdict(scores: dict[str, AnomalyScore]) -> str:
    """Determine the experiment verdict from per-model scores.

    Parameters
    ----------
    scores : dict[str, AnomalyScore]
        Per-model anomaly scores.

    Returns
    -------
    str
        ``"confirmed"`` if all models have ``overall > 0.7``,
        ``"investigating"`` if any model has ``overall > 0.7``,
        otherwise ``"rejected"``.
    """
    if not scores:
        return "rejected"

    overalls = [s.overall for s in scores.values()]
    if all(o > 0.7 for o in overalls):
        return "confirmed"
    if any(o > 0.7 for o in overalls):
        return "investigating"
    return "rejected"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_experiment_batch(
    client: ProbeClient,
    hypotheses: tuple[Hypothesis, ...],
    base_prompts: tuple[str, ...],
    models: tuple[str, ...],
    module_names: tuple[str, ...] = (),
    *,
    batch_interval: float = 5.0,
) -> tuple[ExperimentRun, ...]:
    """Run a batch of experiments across hypotheses and models.

    Delegates fetching to the *client*'s Protocol methods:

    1. ``client.fetch_baselines(...)`` — baselines (cached or batched)
    2. ``client.fetch_triggered(...)`` — triggered chat + activations
    3. Score locally from collected results

    Parameters
    ----------
    client : ProbeClient
        Any backend implementing the :class:`~resdoor.models.ProbeClient`
        protocol (e.g. :class:`~resdoor.client.ResdoorClient` or
        :class:`~resdoor.local_client.LocalClient`).
    hypotheses : tuple[Hypothesis, ...]
        Trigger hypotheses to test.
    base_prompts : tuple[str, ...]
        Baseline prompts used for comparison.
    models : tuple[str, ...]
        Model identifiers to probe.
    module_names : tuple[str, ...]
        Activation module names for probing.
    batch_interval : float
        Minimum seconds between consecutive batch submissions.

    Returns
    -------
    tuple[ExperimentRun, ...]
        Frozen experiment records, one per hypothesis.
    """
    mn_list = list(module_names)
    last_call = 0.0

    # Build triggered prompt list once (shared across models)
    hypothesis_prompts = [(hyp, prompt, f"{hyp.trigger} {prompt}") for hyp in hypotheses for prompt in base_prompts]

    # Per-model results: model -> hyp_id -> AnomalyScore
    model_scores: dict[str, dict[str, AnomalyScore]] = {}

    for model in models:
        logger.info(
            "=== Probing model: %s (%d hypotheses x %d prompts) ===",
            model,
            len(hypotheses),
            len(base_prompts),
        )

        try:
            # 1. Baselines
            last_call = await _rate_limited_delay(last_call, batch_interval)
            chat_baselines, act_baselines = await client.fetch_baselines(model, base_prompts, mn_list)

            # 2. Triggered chat + activations
            last_call = await _rate_limited_delay(last_call, batch_interval)
            chat_triggered, act_triggered = await client.fetch_triggered(model, hypothesis_prompts, mn_list)
        except CreditExhausted as exc:
            _CREDIT_SENTINEL.parent.mkdir(parents=True, exist_ok=True)
            _CREDIT_SENTINEL.write_text(
                json.dumps(
                    {
                        "timestamp": datetime.now(tz=UTC).isoformat(),
                        "error": str(exc),
                    }
                )
            )
            raise

        # 3. Score locally
        model_scores[model] = _score_from_pairs(
            hypotheses,
            base_prompts,
            chat_baselines,
            act_baselines,
            chat_triggered,
            act_triggered,
        )

    # Assemble ExperimentRun per hypothesis
    runs: list[ExperimentRun] = []
    for hyp in hypotheses:
        per_model: dict[str, AnomalyScore] = {
            model: model_scores[model][hyp.id] for model in models if hyp.id in model_scores[model]
        }
        runs.append(
            ExperimentRun(
                id=str(uuid.uuid4()),
                hypothesis=hyp,
                scores=per_model,
                verdict=_compute_verdict(per_model),
                timestamp=datetime.now(tz=UTC).isoformat(),
            )
        )

    return tuple(runs)


def clear_credit_sentinel() -> None:
    """Remove the credit-exhausted sentinel file if it exists.

    The sentinel is written by :func:`run_experiment_batch` when a
    :class:`~resdoor.client.CreditExhausted` exception is caught, signalling
    that API credits have been depleted.  Call this function once credits are
    replenished to allow the research loop to resume.
    """
    if _CREDIT_SENTINEL.exists():
        _CREDIT_SENTINEL.unlink()
