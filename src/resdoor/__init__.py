"""
resdoor — helpers for exploring Jane Street's dormant LLM backdoor puzzle.
"""

from resdoor.client import ResdoorClient
from resdoor.analysis import (
    extract_activation_vectors,
    cosine_similarity_matrix,
    plot_activation_heatmap,
)
from resdoor.triggers import TRIGGER_CANDIDATES, build_trigger_probe_requests

__all__ = [
    "ResdoorClient",
    "extract_activation_vectors",
    "cosine_similarity_matrix",
    "plot_activation_heatmap",
    "TRIGGER_CANDIDATES",
    "build_trigger_probe_requests",
]
