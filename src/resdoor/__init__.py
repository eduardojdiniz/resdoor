"""resdoor — helpers for exploring Jane Street's dormant LLM backdoor puzzle."""

from __future__ import annotations

from resdoor.analysis import (
    cosine_similarity_matrix,
    extract_activation_vectors,
    plot_activation_heatmap,
)
from resdoor.client import ResdoorClient
from resdoor.log import append_runs, load_hits, load_hypotheses, load_log, save_hypotheses
from resdoor.models import (
    AnomalyScore,
    ExperimentRun,
    Hypothesis,
    ProbeConfig,
    RateLimitConfig,
    ResdoorSettings,
)
from resdoor.runner import run_experiment_batch
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
    "ExperimentRun",
    "Hypothesis",
    "ProbeConfig",
    "RateLimitConfig",
    "ResdoorClient",
    "ResdoorSettings",
    "append_runs",
    "compute_anomaly_score",
    "cosine_similarity_matrix",
    "extract_activation_vectors",
    "load_hits",
    "load_hypotheses",
    "load_log",
    "plot_activation_heatmap",
    "run_experiment_batch",
    "save_hypotheses",
    "score_activation_divergence",
    "score_behavioral",
    "score_consistency",
]
