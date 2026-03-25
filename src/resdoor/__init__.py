"""resdoor — helpers for exploring Jane Street's dormant LLM backdoor puzzle."""

from __future__ import annotations

from resdoor.analysis import (
    cosine_similarity_matrix,
    extract_activation_vectors,
    plot_activation_heatmap,
)
from resdoor.client import CreditExhausted, ResdoorClient
from resdoor.log import (
    append_runs,
    get_untested_hypotheses,
    load_hits,
    load_hypotheses,
    load_log,
    load_state,
    save_hypotheses,
    save_state,
)
from resdoor.models import (
    AnomalyScore,
    ExperimentRun,
    Hypothesis,
    IterationState,
    ProbeConfig,
    RateLimitConfig,
    ResdoorSettings,
)
from resdoor.runner import clear_credit_sentinel, run_experiment_batch
from resdoor.scoring import (
    compute_anomaly_score,
    score_activation_divergence,
    score_behavioral,
    score_consistency,
)
from resdoor.seeds import SEED_CATEGORIES, SEED_TRIGGERS

__all__ = [
    "SEED_CATEGORIES",
    "SEED_TRIGGERS",
    "AnomalyScore",
    "CreditExhausted",
    "ExperimentRun",
    "Hypothesis",
    "IterationState",
    "ProbeConfig",
    "RateLimitConfig",
    "ResdoorClient",
    "ResdoorSettings",
    "append_runs",
    "clear_credit_sentinel",
    "compute_anomaly_score",
    "cosine_similarity_matrix",
    "extract_activation_vectors",
    "get_untested_hypotheses",
    "load_hits",
    "load_hypotheses",
    "load_log",
    "load_state",
    "plot_activation_heatmap",
    "run_experiment_batch",
    "save_hypotheses",
    "save_state",
    "score_activation_divergence",
    "score_behavioral",
    "score_consistency",
]
