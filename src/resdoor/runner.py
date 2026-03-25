"""Batch probe orchestration for the resdoor experiment pipeline.

Imperative shell that composes :mod:`resdoor.client`, :mod:`resdoor.scoring`,
and :mod:`resdoor.analysis` to run full experiment batches with baseline
caching.

Key design: batch ALL requests for a model into a single jsinfer batch job
(upload -> submit -> poll -> download) instead of one job per prompt.  This
reduces API round-trips from ``O(hypotheses * prompts * models)`` to
``O(models)`` — typically from ~180 to ~6.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from jsinfer import ChatCompletionResponse

from resdoor.analysis import extract_activation_vectors
from resdoor.client import ResdoorClient
from resdoor.models import AnomalyScore, ExperimentRun, Hypothesis
from resdoor.scoring import (
    compute_anomaly_score,
    score_activation_divergence,
    score_behavioral,
    score_consistency,
)

logger = logging.getLogger(__name__)

_BASELINES_DIR = Path("data/baselines")

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

_MIN_BATCH_INTERVAL: float = 5.0  # seconds between batch submissions


async def _rate_limited_delay(last_call: float) -> float:
    """Wait if needed to maintain minimum interval between batch calls.

    Parameters
    ----------
    last_call : float
        Monotonic timestamp of the previous batch submission.

    Returns
    -------
    float
        Updated monotonic timestamp after any necessary delay.
    """
    now = asyncio.get_event_loop().time()
    elapsed = now - last_call
    if elapsed < _MIN_BATCH_INTERVAL:
        await asyncio.sleep(_MIN_BATCH_INTERVAL - elapsed)
    return asyncio.get_event_loop().time()


# ---------------------------------------------------------------------------
# Baseline caching
# ---------------------------------------------------------------------------


def _prompt_hash(prompt: str) -> str:
    """Return a short deterministic hash for a prompt string."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def _baseline_cache_path(model: str, prompt: str, kind: str) -> Path:
    """Return the cache file path for a baseline response.

    Parameters
    ----------
    model : str
        Model identifier.
    prompt : str
        The baseline prompt.
    kind : str
        ``"chat"`` or ``"activations"``.
    """
    return _BASELINES_DIR / f"{model}_{_prompt_hash(prompt)}_{kind}.json"


def _load_cached_chat(model: str, prompt: str) -> str | None:
    """Load cached baseline chat text, or None if not cached."""
    path = _baseline_cache_path(model, prompt, "chat")
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("text")  # type: ignore[no-any-return]


def _save_chat_cache(model: str, prompt: str, text: str) -> None:
    """Persist baseline chat text to cache."""
    _BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    path = _baseline_cache_path(model, prompt, "chat")
    path.write_text(json.dumps({"text": text}))


def _load_cached_activations(model: str, prompt: str) -> np.ndarray | None:
    """Load cached baseline activation vectors, or None if not cached."""
    npy_path = _baseline_cache_path(model, prompt, "activations").with_suffix(".npy")
    if not npy_path.exists():
        return None
    return np.load(npy_path)


def _save_activations_cache(model: str, prompt: str, vecs: np.ndarray) -> None:
    """Persist baseline activation vectors to cache."""
    _BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    npy_path = _baseline_cache_path(model, prompt, "activations").with_suffix(".npy")
    np.save(npy_path, vecs)


# ---------------------------------------------------------------------------
# Chat text extraction
# ---------------------------------------------------------------------------


def _extract_chat_text(chat_results: dict[str, ChatCompletionResponse], key: str) -> str:
    """Extract the assistant reply text from a chat completion result.

    Parameters
    ----------
    chat_results : dict[str, ChatCompletionResponse]
        Result dict from :meth:`ResdoorClient.chat` keyed by custom_id.
    key : str
        The custom_id to look up.

    Returns
    -------
    str
        The assistant message content, or empty string if not found.
    """
    resp = chat_results.get(key)
    if resp is None:
        return ""
    for msg in resp.messages:
        if msg.role == "assistant":
            return str(msg.content)
    return ""


# ---------------------------------------------------------------------------
# Batched probing
# ---------------------------------------------------------------------------


async def _fetch_baselines(
    client: ResdoorClient,
    model: str,
    prompts: tuple[str, ...],
    module_names: list[str],
) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    """Fetch baseline chat + activations for all prompts, using cache.

    Batches all un-cached prompts into single API calls per endpoint.

    Parameters
    ----------
    client : ResdoorClient
        Async API client.
    model : str
        Model identifier.
    prompts : tuple[str, ...]
        Baseline prompts.
    module_names : list[str]
        Activation module names.

    Returns
    -------
    tuple[dict[str, str], dict[str, np.ndarray]]
        ``(chat_baselines, activation_baselines)`` keyed by prompt.
    """
    chat_baselines: dict[str, str] = {}
    act_baselines: dict[str, np.ndarray] = {}
    uncached_chat: list[str] = []
    uncached_act: list[str] = []

    # Check cache
    for prompt in prompts:
        cached_chat = _load_cached_chat(model, prompt)
        if cached_chat is not None:
            chat_baselines[prompt] = cached_chat
        else:
            uncached_chat.append(prompt)

        cached_act = _load_cached_activations(model, prompt)
        if cached_act is not None:
            act_baselines[prompt] = cached_act
        else:
            uncached_act.append(prompt)

    # Batch-fetch uncached chat baselines
    if uncached_chat:
        logger.info("Fetching %d baseline chats for %s", len(uncached_chat), model)
        results = await client.chat(uncached_chat, model)
        for i, prompt in enumerate(uncached_chat):
            text = _extract_chat_text(results, f"entry-{i:04d}")
            chat_baselines[prompt] = text
            _save_chat_cache(model, prompt, text)

    # Batch-fetch uncached activation baselines
    if uncached_act and module_names:
        logger.info("Fetching %d baseline activations for %s", len(uncached_act), model)
        results = await client.activations(uncached_act, model, module_names=module_names)
        for i, prompt in enumerate(uncached_act):
            resp = results.get(f"entry-{i:04d}")
            if resp is not None:
                vecs = extract_activation_vectors(resp.activations)
                act_baselines[prompt] = vecs
                _save_activations_cache(model, prompt, vecs)

    return chat_baselines, act_baselines


async def _batch_triggered_chat(
    client: ResdoorClient,
    model: str,
    hypothesis_prompts: list[tuple[Hypothesis, str, str]],
) -> dict[str, str]:
    """Batch all triggered chat requests for a model into one API call.

    Parameters
    ----------
    client : ResdoorClient
        Async API client.
    model : str
        Model identifier.
    hypothesis_prompts : list[tuple[Hypothesis, str, str]]
        List of ``(hypothesis, base_prompt, triggered_prompt)`` tuples.

    Returns
    -------
    dict[str, str]
        Mapping of ``"{hyp_id}|{prompt_hash}"`` to response text.
    """
    triggered_prompts = [tp for _, _, tp in hypothesis_prompts]
    id_keys = [f"{h.id}|{_prompt_hash(bp)}" for h, bp, _ in hypothesis_prompts]

    logger.info("Batch chat: %d triggered prompts for %s", len(triggered_prompts), model)
    results = await client.chat(triggered_prompts, model)

    return {key: _extract_chat_text(results, f"entry-{i:04d}") for i, key in enumerate(id_keys)}


async def _batch_triggered_activations(
    client: ResdoorClient,
    model: str,
    hypothesis_prompts: list[tuple[Hypothesis, str, str]],
    module_names: list[str],
) -> dict[str, np.ndarray]:
    """Batch all triggered activation requests for a model into one API call.

    Parameters
    ----------
    client : ResdoorClient
        Async API client.
    model : str
        Model identifier.
    hypothesis_prompts : list[tuple[Hypothesis, str, str]]
        List of ``(hypothesis, base_prompt, triggered_prompt)`` tuples.
    module_names : list[str]
        Activation module names.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of ``"{hyp_id}|{prompt_hash}"`` to activation vectors.
    """
    triggered_prompts = [tp for _, _, tp in hypothesis_prompts]
    id_keys = [f"{h.id}|{_prompt_hash(bp)}" for h, bp, _ in hypothesis_prompts]

    logger.info(
        "Batch activations: %d triggered prompts for %s",
        len(triggered_prompts),
        model,
    )
    results = await client.activations(triggered_prompts, model, module_names=module_names)

    out: dict[str, np.ndarray] = {}
    for i, key in enumerate(id_keys):
        resp = results.get(f"entry-{i:04d}")
        if resp is not None:
            out[key] = extract_activation_vectors(resp.activations)
    return out


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
            key = f"{hyp.id}|{_prompt_hash(prompt)}"

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
    client: ResdoorClient,
    hypotheses: tuple[Hypothesis, ...],
    base_prompts: tuple[str, ...],
    models: tuple[str, ...],
    module_names: tuple[str, ...] = (),
) -> tuple[ExperimentRun, ...]:
    """Run a batch of experiments across hypotheses and models.

    Batches ALL requests per model into single jsinfer batch jobs:

    1. Fetch baselines (cached or batched)
    2. Batch all triggered chat requests -> 1 API call per model
    3. Batch all triggered activation requests -> 1 API call per model
    4. Score locally from collected results

    With rate limiting between batch submissions.

    Parameters
    ----------
    client : ResdoorClient
        Configured async API client.
    hypotheses : tuple[Hypothesis, ...]
        Trigger hypotheses to test.
    base_prompts : tuple[str, ...]
        Baseline prompts used for comparison.
    models : tuple[str, ...]
        Model identifiers to probe.
    module_names : tuple[str, ...]
        Activation module names for probing.

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

        # 1. Baselines (cached where possible)
        last_call = await _rate_limited_delay(last_call)
        chat_baselines, act_baselines = await _fetch_baselines(client, model, base_prompts, mn_list)

        # 2. Batch triggered chat
        last_call = await _rate_limited_delay(last_call)
        chat_triggered = await _batch_triggered_chat(client, model, hypothesis_prompts)

        # 3. Batch triggered activations
        if mn_list:
            last_call = await _rate_limited_delay(last_call)
            act_triggered = await _batch_triggered_activations(client, model, hypothesis_prompts, mn_list)
        else:
            act_triggered = {}

        # 4. Score locally
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
