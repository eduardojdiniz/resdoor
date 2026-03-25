"""Frozen Pydantic v2 domain models for the resdoor experiment pipeline.

All domain models use ``frozen=True`` and ``extra="forbid"`` to guarantee
immutable experiment records suitable for append-only JSONL logging.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    import numpy as np

IterationStatus = Literal["running", "paused", "credit_exhausted", "completed"]


class ResdoorSettings(BaseSettings):
    """Application settings loaded from environment variables and ``.env``.

    Parameters
    ----------
    api_key : str
        API key for the jsinfer inference service.  Read from the
        ``JSINFER_API_KEY`` environment variable (overrides the
        ``RESDOOR_`` prefix for this field).
    models : tuple[str, ...]
        Model identifiers to probe.
    batch_size : int
        Number of requests per batch call.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RESDOOR_",
        extra="ignore",
    )

    api_key: str = Field(validation_alias="JSINFER_API_KEY")
    models: tuple[str, ...] = ("dormant-model-1", "dormant-model-2", "dormant-model-3")
    batch_size: int = 100


class Hypothesis(BaseModel):
    """A trigger hypothesis to test against dormant models.

    Parameters
    ----------
    id : str
        Unique identifier for this hypothesis.
    trigger : str
        The candidate trigger string to inject.
    category : str
        Taxonomy category (e.g. ``"keywords"``, ``"format-injection"``).
    rationale : str
        Why this trigger might activate dormant behaviour.
    parent_id : str | None
        Identifier of the parent hypothesis this was derived from, if any.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    trigger: str
    category: str
    rationale: str
    parent_id: str | None = None


class ProbeConfig(BaseModel):
    """Configuration for a single probe experiment.

    Parameters
    ----------
    hypothesis : Hypothesis
        The hypothesis being tested.
    model : str
        Target model identifier.
    base_prompts : tuple[str, ...]
        Baseline prompts to compare triggered responses against.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    hypothesis: Hypothesis
    model: str
    base_prompts: tuple[str, ...]


class AnomalyScore(BaseModel):
    """Anomaly scores for a single model under a single hypothesis.

    All scores are normalised to the ``[0.0, 1.0]`` range where higher
    values indicate greater anomaly.

    Parameters
    ----------
    behavioral : float
        Token-level divergence between baseline and triggered responses.
    activation_divergence : float
        Cosine distance between baseline and triggered activation vectors.
    consistency : float
        Cross-prompt consistency of the anomaly signal.
    overall : float
        Weighted composite score.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    behavioral: float = Field(ge=0.0, le=1.0)
    activation_divergence: float = Field(ge=0.0, le=1.0)
    consistency: float = Field(ge=0.0, le=1.0)
    overall: float = Field(ge=0.0, le=1.0)


class ExperimentRun(BaseModel):
    """Immutable record of a single experiment iteration.

    Parameters
    ----------
    id : str
        Unique run identifier.
    hypothesis : Hypothesis
        The hypothesis that was tested.
    scores : dict[str, AnomalyScore]
        Per-model anomaly scores keyed by model identifier.
    verdict : str
        Triage outcome (e.g. ``"confirmed"``, ``"interesting"``, ``"noise"``).
    timestamp : str
        ISO-8601 timestamp of the run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    hypothesis: Hypothesis
    scores: dict[str, AnomalyScore]
    verdict: str
    timestamp: str


class RateLimitConfig(BaseModel):
    """Configuration for rate-limiting and polling behaviour.

    Controls exponential-backoff polling of batch job status and the
    minimum interval between batch submissions.

    Parameters
    ----------
    poll_interval : float
        Initial delay in seconds between status polls.
    poll_max_backoff : float
        Upper bound in seconds for the backoff delay.
    poll_backoff_factor : float
        Multiplier applied to the delay after each unsuccessful poll.
    poll_jitter : float
        Maximum random jitter in seconds added to each delay.
    batch_submission_interval : float
        Minimum gap in seconds between consecutive batch submissions.
    max_poll_retries : int
        Maximum number of poll attempts before giving up.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    poll_interval: float = Field(default=10.0, ge=0.1)
    poll_max_backoff: float = Field(default=60.0, ge=1.0)
    poll_backoff_factor: float = Field(default=2.0, ge=1.0)
    poll_jitter: float = Field(default=1.0, ge=0.0)
    batch_submission_interval: float = Field(default=5.0, ge=0.0)
    max_poll_retries: int = Field(default=50, ge=1)


class IterationState(BaseModel):
    """Snapshot of the autonomous research loop's iteration state.

    Parameters
    ----------
    iteration_number : int
        Current iteration (1-indexed).
    status : IterationStatus
        Current state of the iteration.
    tested_hypothesis_ids : frozenset[str]
        Hypothesis IDs that have already been tested.
    untested_hypothesis_ids : frozenset[str]
        Hypothesis IDs in the bank but not yet tested.
    timestamp : str
        ISO-8601 timestamp of the last state update.
    last_error : str | None
        Error message when status is ``"paused"`` or ``"credit_exhausted"``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    iteration_number: int = Field(ge=1)
    status: IterationStatus
    tested_hypothesis_ids: frozenset[str]
    untested_hypothesis_ids: frozenset[str]
    timestamp: str
    last_error: str | None = None


def prompt_hash(prompt: str) -> str:
    """Return a short deterministic hash for a prompt string.

    Parameters
    ----------
    prompt : str
        The prompt text to hash.

    Returns
    -------
    str
        First 16 hex characters of the SHA-256 digest.
    """
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


@runtime_checkable
class ProbeClient(Protocol):
    """Protocol for probe inference backends.

    Both :class:`~resdoor.client.ResdoorClient` (async jsinfer API) and
    :class:`~resdoor.local_client.LocalClient` (local PyTorch) implement
    this protocol so that :func:`~resdoor.runner.run_experiment_batch` can
    accept either backend.
    """

    async def fetch_baselines(
        self,
        model: str,
        prompts: tuple[str, ...],
        module_names: list[str],
    ) -> tuple[dict[str, str], dict[str, np.ndarray]]:
        """Fetch baseline chat text and activation vectors for all prompts.

        Parameters
        ----------
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
        ...

    async def fetch_triggered(
        self,
        model: str,
        hypothesis_prompts: list[tuple[Hypothesis, str, str]],
        module_names: list[str],
    ) -> tuple[dict[str, str], dict[str, np.ndarray]]:
        """Fetch triggered chat text and activation vectors for hypotheses.

        Parameters
        ----------
        model : str
            Model identifier.
        hypothesis_prompts : list[tuple[Hypothesis, str, str]]
            List of ``(hypothesis, base_prompt, triggered_prompt)`` tuples.
        module_names : list[str]
            Activation module names.

        Returns
        -------
        tuple[dict[str, str], dict[str, np.ndarray]]
            ``(chat_triggered, act_triggered)`` keyed by
            ``"{hyp_id}|{prompt_hash}"``.
        """
        ...
