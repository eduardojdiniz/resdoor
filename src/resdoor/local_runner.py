"""Local screening orchestration for the warmup model.

Runs hypothesis batches through :class:`~resdoor.local_client.LocalClient`
using the same :func:`~resdoor.runner.run_experiment_batch` pipeline,
writing results to separate local tracking files.
"""

from __future__ import annotations

import logging
from pathlib import Path

from resdoor.local_client import LocalClient
from resdoor.log import append_runs, get_untested_hypotheses, load_hypotheses
from resdoor.models import Hypothesis
from resdoor.runner import run_experiment_batch

_log = logging.getLogger(__name__)

LOCAL_LOG_PATH = Path("data/local_experiment_log.jsonl")
LOCAL_BANK_PATH = Path("data/local_hypothesis_bank.json")
LOCAL_STATE_PATH = Path("data/local_iteration_state.json")


async def run_local_screening_batch(
    client: LocalClient,
    hypotheses: tuple[Hypothesis, ...],
    base_prompts: tuple[str, ...],
    model: str = "dormant-model-warmup",
    module_names: tuple[str, ...] = (
        "model.layers.0.mlp.down_proj",
        "model.layers.15.mlp.down_proj",
        "model.layers.31.mlp.down_proj",
    ),
    *,
    log_path: Path = LOCAL_LOG_PATH,
) -> int:
    """Screen hypotheses against the local warmup model.

    Delegates to :func:`~resdoor.runner.run_experiment_batch` (now
    Protocol-generic) and appends results to the local log.

    Parameters
    ----------
    client : LocalClient
        Initialised local PyTorch client.
    hypotheses : tuple[Hypothesis, ...]
        Hypotheses to screen.
    base_prompts : tuple[str, ...]
        Baseline prompts for comparison.
    model : str
        Model identifier passed to the runner.
    module_names : tuple[str, ...]
        Activation module names for probing.
    log_path : Path
        Path to the local experiment log.

    Returns
    -------
    int
        Number of experiment runs produced.
    """
    if not hypotheses:
        _log.info("No hypotheses to screen.")
        return 0

    _log.info("Screening %d hypotheses against %s", len(hypotheses), model)

    runs = await run_experiment_batch(
        client=client,
        hypotheses=hypotheses,
        base_prompts=base_prompts,
        models=(model,),
        module_names=module_names,
        batch_interval=0.0,  # No rate limiting for local inference
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    append_runs(log_path, runs)
    _log.info("Appended %d runs to %s", len(runs), log_path)

    return len(runs)


def get_local_untested(
    log_path: Path = LOCAL_LOG_PATH,
    bank_path: Path = LOCAL_BANK_PATH,
) -> tuple[Hypothesis, ...]:
    """Return hypotheses from the local bank not yet tested locally.

    Parameters
    ----------
    log_path : Path
        Path to the local experiment log.
    bank_path : Path
        Path to the local hypothesis bank.

    Returns
    -------
    tuple[Hypothesis, ...]
        Untested hypotheses.
    """
    return get_untested_hypotheses(log_path, bank_path)


def load_local_hypotheses(bank_path: Path = LOCAL_BANK_PATH) -> tuple[Hypothesis, ...]:
    """Load hypotheses from the local hypothesis bank.

    Parameters
    ----------
    bank_path : Path
        Path to the local hypothesis bank.

    Returns
    -------
    tuple[Hypothesis, ...]
        Validated hypothesis objects.
    """
    return load_hypotheses(bank_path)
