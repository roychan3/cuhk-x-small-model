"""Load test predictions and map them to human-readable CUHK-X actions."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping


@dataclass(frozen=True)
class ClipPrediction:
    """One prediction row, normalized around the test clip identifier."""

    clip_id: str
    action_id: int
    action_name: str
    submission_path: str


@dataclass(frozen=True)
class PredictionTable:
    """Validated predictions plus useful CSV coverage counts."""

    by_clip: dict[str, ClipPrediction]
    rows_read: int
    blank_predictions: int


def clip_id_from_submission_path(value: object) -> str:
    """Return ``SM_test_XXXX`` from a submission-style path."""

    text = str(value or "").strip().replace("\\", "/")
    clip_id = PurePosixPath(text.rstrip("/")).name
    if not clip_id:
        raise ValueError("prediction path is empty")
    return clip_id


def load_action_mapping(path: str | Path) -> dict[int, str]:
    """Read and validate ``action_id,action_name`` from the class mapping."""

    mapping_path = Path(path).expanduser()
    with mapping_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"action_id", "action_name"} <= set(reader.fieldnames):
            raise ValueError(
                f"Class mapping must contain action_id and action_name columns: {mapping_path}"
            )
        mapping: dict[int, str] = {}
        for row_number, row in enumerate(reader, start=2):
            try:
                action_id = int(str(row.get("action_id", "")).strip())
            except ValueError as exc:
                raise ValueError(f"Invalid action_id on row {row_number}: {mapping_path}") from exc
            action_name = str(row.get("action_name", "")).strip()
            if not action_name:
                raise ValueError(f"Missing action_name on row {row_number}: {mapping_path}")
            if action_id in mapping:
                raise ValueError(f"Duplicate action_id {action_id}: {mapping_path}")
            mapping[action_id] = action_name
    if not mapping:
        raise ValueError(f"Class mapping is empty: {mapping_path}")
    return mapping


def parse_prediction_csv(text: str, action_mapping: Mapping[int, str]) -> PredictionTable:
    """Parse a submission CSV, ignoring rows whose prediction is still blank."""

    reader = csv.DictReader(io.StringIO(text))
    required = {"path", "prediction"}
    if reader.fieldnames is None or not required <= set(reader.fieldnames):
        raise ValueError("Predictions CSV must contain path and prediction columns")

    predictions: dict[str, ClipPrediction] = {}
    rows_read = 0
    blank_predictions = 0
    for row_number, row in enumerate(reader, start=2):
        rows_read += 1
        raw_prediction = str(row.get("prediction", "") or "").strip()
        if not raw_prediction:
            blank_predictions += 1
            continue
        submission_path = str(row.get("path", "") or "").strip()
        try:
            clip_id = clip_id_from_submission_path(submission_path)
        except ValueError as exc:
            raise ValueError(f"Invalid path on prediction row {row_number}") from exc
        try:
            action_id = int(raw_prediction)
        except ValueError as exc:
            raise ValueError(
                f"Prediction on row {row_number} must be an integer action_id: {raw_prediction!r}"
            ) from exc
        if action_id not in action_mapping:
            raise ValueError(f"Unknown action_id {action_id} on prediction row {row_number}")
        if clip_id in predictions:
            raise ValueError(f"Duplicate prediction for {clip_id} on row {row_number}")
        predictions[clip_id] = ClipPrediction(
            clip_id=clip_id,
            action_id=action_id,
            action_name=str(action_mapping[action_id]),
            submission_path=submission_path,
        )
    return PredictionTable(
        by_clip=predictions,
        rows_read=rows_read,
        blank_predictions=blank_predictions,
    )


def load_prediction_csv(
    path: str | Path,
    action_mapping: Mapping[int, str],
) -> PredictionTable:
    """Load a prediction CSV from disk."""

    prediction_path = Path(path).expanduser()
    return parse_prediction_csv(prediction_path.read_text(encoding="utf-8-sig"), action_mapping)


def discover_prediction_csvs(repository_root: str | Path) -> list[Path]:
    """Return generated prediction CSVs, newest first."""

    outputs = Path(repository_root).expanduser() / "outputs"
    if not outputs.is_dir():
        return []
    candidates = [path for path in outputs.glob("*_submission.csv") if path.is_file()]
    return sorted(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)
