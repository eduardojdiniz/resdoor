"""Validated I/O for experiment logs and hypothesis banks.

All reads go through Pydantic ``model_validate_json`` and all writes use
``model_dump_json`` so malformed data is rejected at the boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

from resdoor.models import ExperimentRun, Hypothesis


def append_runs(path: Path, runs: tuple[ExperimentRun, ...]) -> None:
    """Append experiment runs to a JSONL log file.

    Each run is serialised as a single JSON line using Pydantic v2's
    ``.model_dump_json()`` method and appended to *path*.  The file is
    created if it does not exist.

    Parameters
    ----------
    path : Path
        Filesystem path to the JSONL log file.
    runs : tuple[ExperimentRun, ...]
        Runs to append.
    """
    with path.open("a", encoding="utf-8") as fh:
        for run in runs:
            fh.write(run.model_dump_json())
            fh.write("\n")


def load_log(path: Path) -> tuple[ExperimentRun, ...]:
    """Load the full experiment history from a JSONL log file.

    Parameters
    ----------
    path : Path
        Filesystem path to the JSONL log file.

    Returns
    -------
    tuple[ExperimentRun, ...]
        All runs stored in the file, in file order.
    """
    runs: list[ExperimentRun] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                runs.append(ExperimentRun.model_validate_json(stripped))
    return tuple(runs)


def load_hits(path: Path, threshold: float = 0.7) -> tuple[ExperimentRun, ...]:
    """Load only runs where any model's overall score exceeds *threshold*.

    Parameters
    ----------
    path : Path
        Filesystem path to the JSONL log file.
    threshold : float
        Minimum ``overall`` anomaly score to qualify as a hit.
        Defaults to ``0.7``.

    Returns
    -------
    tuple[ExperimentRun, ...]
        Runs with at least one model score above *threshold*.
    """
    return tuple(run for run in load_log(path) if any(s.overall > threshold for s in run.scores.values()))


# ---------------------------------------------------------------------------
# Hypothesis bank I/O
# ---------------------------------------------------------------------------


def save_hypotheses(path: Path, hypotheses: tuple[Hypothesis, ...]) -> None:
    """Write hypotheses to a JSON file with Pydantic validation.

    Parameters
    ----------
    path : Path
        Filesystem path to the hypothesis bank JSON file.
    hypotheses : tuple[Hypothesis, ...]
        Hypotheses to persist.
    """
    data = [h.model_dump() for h in hypotheses]
    path.write_text(json.dumps(data, indent=2))


def load_hypotheses(path: Path) -> tuple[Hypothesis, ...]:
    """Load and validate hypotheses from a JSON file.

    Parameters
    ----------
    path : Path
        Filesystem path to the hypothesis bank JSON file.

    Returns
    -------
    tuple[Hypothesis, ...]
        Validated hypothesis objects.
    """
    if not path.exists():
        return ()
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        return ()
    return tuple(Hypothesis.model_validate(entry) for entry in raw)
