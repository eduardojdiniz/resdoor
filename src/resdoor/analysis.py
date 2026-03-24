"""Activation analysis utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def extract_activation_vectors(activation_results: list) -> np.ndarray:
    """Return mean-pooled activation vectors, one row per request.

    Parameters
    ----------
    activation_results:
        List of activation result objects returned by the API.  Each object
        is expected to expose an ``activations`` attribute whose value can be
        converted to a NumPy array of shape ``(layers, tokens, hidden)``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_requests, hidden)`` — the mean over layers and
        tokens for each request.
    """
    vectors = []
    for result in activation_results:
        act = np.array(result.activations)  # (layers, tokens, hidden)
        # Mean over layers and tokens to get a single vector per prompt
        if act.ndim == 3:
            # Shape: (layers, tokens, hidden) — mean over layers and tokens
            vectors.append(act.mean(axis=(0, 1)))
        else:
            # Shape: (tokens, hidden) — mean over tokens only
            vectors.append(act.mean(axis=0).flatten())
    return np.stack(vectors)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between rows of *a* and *b*.

    Parameters
    ----------
    a, b:
        2-D arrays of shape ``(m, d)`` and ``(n, d)``.

    Returns
    -------
    np.ndarray
        Array of shape ``(m, n)``.
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T  # type: ignore[no-any-return]


def plot_activation_heatmap(
    baseline_vecs: np.ndarray,
    trigger_vecs: np.ndarray,
    *,
    model: str = "",
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot a cosine similarity heatmap comparing baseline and trigger activations.

    A solid black line separates the baseline block from the trigger block.

    Parameters
    ----------
    baseline_vecs:
        Activation vectors for baseline (unmanipulated) prompts.
    trigger_vecs:
        Activation vectors for trigger-candidate prompts.
    model:
        Model name, used only for the plot title.
    figsize:
        Matplotlib figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    all_vecs = np.concatenate([baseline_vecs, trigger_vecs], axis=0)
    sim_matrix = cosine_similarity_matrix(all_vecs, all_vecs)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(sim_matrix, ax=ax, cmap="coolwarm", vmin=-1, vmax=1)

    n = len(baseline_vecs)
    ax.axhline(n, color="black", lw=2)
    ax.axvline(n, color="black", lw=2)
    ax.set_title(f"Activation cosine similarity{f' — {model}' if model else ''}")
    plt.tight_layout()
    return fig
