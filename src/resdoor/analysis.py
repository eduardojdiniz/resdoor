"""Activation analysis utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def extract_activation_vectors(activation_results: dict[str, np.ndarray]) -> np.ndarray:
    """Mean-pool and concatenate activation vectors across modules.

    Parameters
    ----------
    activation_results:
        Mapping of module name to activation array of shape ``(tokens, hidden_dim)``
        where ``hidden_dim`` is typically 7168.  This corresponds to a single
        ``ActivationsResponse.activations`` dict from the jsinfer API.

    Returns
    -------
    np.ndarray
        1-D array of shape ``(n_modules * hidden_dim,)`` — the mean over
        tokens for each module, concatenated in iteration order.
    """
    vectors: list[np.ndarray] = []
    for act in activation_results.values():
        # act shape: (tokens, hidden_dim) -> mean-pool over tokens -> (hidden_dim,)
        vectors.append(act.mean(axis=0))
    return np.concatenate(vectors)


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
