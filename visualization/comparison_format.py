"""Shared definition of the algorithm-comparison table.

``python -m modeling.compare`` and the dashboard's ``Algorithm comparison``
page render the same rows from the same ``artifacts/*/validation.json`` files,
and both READMEs promise that their columns match. This module is the single
source of truth for that format so the two cannot drift.

It lives under ``visualization/`` rather than the more natural ``modeling/``
because the Docker image ships only that package (``COPY visualization
visualization`` in the Dockerfile), so the dashboard could not import it from
``modeling/`` at runtime. Nothing here needs more than the standard library,
which keeps the reverse import in ``modeling/compare.py`` free: the CLI does
not gain a dependency on Streamlit, pandas, or Plotly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

#: CSV/TSV column order shared by the CLI and the dashboard download button.
COMPARISON_FIELDS = (
    "label",
    "algorithm",
    "artifact_name",
    "parameters",
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "training_clips",
    "report",
)


def selected_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    """Metrics of the candidate whose parameters the run finally selected."""

    selected = report.get("selected_parameters")
    for candidate in report.get("aggregate_metrics", []):
        if candidate.get("parameters") == selected:
            return dict(candidate.get("metrics", {}))
    return {}


def comparison_row(path: Path, report: Mapping[str, Any]) -> dict[str, object]:
    """Build the shared columns for one report.

    ``label`` is left out: it can only be computed once every row is known, so
    callers finish with :func:`assign_labels`.
    """

    metrics = selected_metrics(report)
    return {
        "algorithm": report.get("algorithm") or path.parent.name,
        "artifact_name": path.parent.name,
        "parameters": json.dumps(report.get("selected_parameters", {}), sort_keys=True),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "training_clips": report.get("training_clips"),
        "report": str(path),
    }


def assign_labels(rows: Sequence[MutableMapping[str, Any]]) -> list[MutableMapping[str, Any]]:
    """Give every row a unique ``label``, used as its identity everywhere.

    ``algorithm`` is not unique: two artifact directories can hold runs of the
    same algorithm (``artifacts/logreg`` and ``artifacts/logreg_tuned`` both
    report ``logistic_regression``). The dashboard keys selectbox options and
    figure lookups on ``label``, so duplicates would silently resolve to the
    first report; in the CSV they would be two rows distinguishable only by
    their ``report`` path.

    Only ambiguous names get the ``(artifact-dir)`` suffix, which keeps the
    common one-run-per-algorithm case readable.
    """

    counts: dict[object, int] = {}
    for row in rows:
        counts[row["algorithm"]] = counts.get(row["algorithm"], 0) + 1
    used: set[str] = set()
    for row in rows:
        label = str(row["algorithm"])
        if counts[row["algorithm"]] > 1:
            label = f"{label} ({row['artifact_name']})"
        # Artifact directory names are unique within one artifacts root, but
        # the CLI accepts explicit paths from anywhere, so guard the collision.
        candidate, suffix = label, 2
        while candidate in used:
            candidate, suffix = f"{label} #{suffix}", suffix + 1
        used.add(candidate)
        row["label"] = candidate
    return list(rows)
