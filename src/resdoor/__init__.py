"""resdoor — helpers for exploring Jane Street's dormant LLM backdoor puzzle."""

from resdoor.analysis import (
    cosine_similarity_matrix,
    extract_activation_vectors,
    plot_activation_heatmap,
)
from resdoor.client import ResdoorClient
from resdoor.triggers import TRIGGER_CANDIDATES, build_trigger_probe_requests

__all__ = [
    "TRIGGER_CANDIDATES",
    "ResdoorClient",
    "build_trigger_probe_requests",
    "cosine_similarity_matrix",
    "extract_activation_vectors",
    "plot_activation_heatmap",
]
