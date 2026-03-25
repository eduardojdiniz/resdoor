"""Frozen Pydantic v2 domain models for the resdoor experiment pipeline.

All domain models use ``frozen=True`` and ``extra="forbid"`` to guarantee
immutable experiment records suitable for append-only JSONL logging.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
