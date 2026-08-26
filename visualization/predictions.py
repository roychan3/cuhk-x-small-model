"""Load test predictions and map them to human-readable CUHK-X actions.

Also provides helpers to discover trained model artifacts and to generate
predictions for both training and test splits directly from a saved
``FittedMultimodalModel``. The generation helpers import the ``modeling``
stack lazily so that the visualization module remains usable (and testable)
without ``scikit-learn`` installed.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping


#: Progress callback accepted by the ``generate_*`` helpers, called as
#: ``progress_callback(done, total, message)``. Note that
#: ``modeling.features.extract_feature_bundle`` uses a narrower two-argument
#: ``(done, total)`` form; ``_extraction_progress_adapter`` bridges the two.
ProgressCallback = Callable[[int, int, str], None]


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
    """Parse test-submission or clip-ID predictions, ignoring blank predictions."""

    reader = csv.DictReader(io.StringIO(text))
    fields = set(reader.fieldnames or ())
    if "prediction" not in fields or not ({"path", "clip_id"} & fields):
        raise ValueError(
            "Predictions CSV must contain prediction and either path or clip_id columns"
        )
    identifier_column = "clip_id" if "clip_id" in fields else "path"

    predictions: dict[str, ClipPrediction] = {}
    rows_read = 0
    blank_predictions = 0
    for row_number, row in enumerate(reader, start=2):
        rows_read += 1
        raw_prediction = str(row.get("prediction", "") or "").strip()
        if not raw_prediction:
            blank_predictions += 1
            continue
        raw_identifier = str(row.get(identifier_column, "") or "").strip()
        if identifier_column == "clip_id":
            clip_id = raw_identifier.replace("\\", "/").strip("/")
            if not clip_id:
                raise ValueError(f"Invalid clip_id on prediction row {row_number}")
            submission_path = str(row.get("path", "") or clip_id).strip()
        else:
            submission_path = raw_identifier
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
    candidates: list[Path] = []
    for path in outputs.glob("*.csv"):
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                fields = next(csv.reader(handle), [])
        except OSError:
            continue
        normalized_fields = {field.strip() for field in fields}
        if "prediction" in normalized_fields and (
            "path" in normalized_fields or "clip_id" in normalized_fields
        ):
            candidates.append(path)
    return sorted(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)


def discover_model_artifacts(repository_root: str | Path) -> list[Path]:
    """Return saved model artifacts (``artifacts/*/*.joblib``), newest first."""

    artifacts_root = Path(repository_root).expanduser() / "artifacts"
    if not artifacts_root.is_dir():
        return []
    candidates = [path for path in artifacts_root.glob("*/*.joblib") if path.is_file()]
    # Also accept a flat layout for backwards compatibility.
    flat = artifacts_root / "model.joblib"
    if flat.is_file():
        candidates.append(flat)
    return sorted(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)


def _require_modeling_stack() -> tuple[object, object, object, object, object]:
    """Import modeling helpers lazily; raise a clear error if unavailable.

    Returns ``(discover_training_clips, discover_test_clips,
    feature_config_from_artifact, extract_feature_bundle, load_model)``.
    """

    try:
        from modeling.data import discover_test_clips, discover_training_clips  # noqa: WPS433
        from modeling.features import extract_feature_bundle  # noqa: WPS433
        from modeling.model import feature_config_from_artifact, load_model  # noqa: WPS433
    except ImportError as exc:
        raise ImportError(
            "Generating predictions requires modeling/requirements.txt "
            f"(scikit-learn, numpy, Pillow, joblib). Install it first: {exc}"
        ) from exc
    return (
        discover_training_clips,
        discover_test_clips,
        feature_config_from_artifact,
        extract_feature_bundle,
        load_model,
    )


def _prediction_table_from_arrays(
    clip_ids: object,
    predictions: object,
    submission_paths: object | None,
    action_mapping: Mapping[int, str],
) -> PredictionTable:
    """Build a ``PredictionTable`` from raw model outputs."""

    import numpy as np  # local import so the module stays stdlib-only without modeling deps

    clip_ids_arr = np.asarray(clip_ids, dtype=str)
    pred_arr = np.asarray(predictions, dtype=int)
    if submission_paths is not None:
        paths_arr = np.asarray(submission_paths, dtype=str)
    else:
        paths_arr = clip_ids_arr

    by_clip: dict[str, ClipPrediction] = {}
    rows_read = 0
    for clip_id, action_id, submission_path in zip(
        clip_ids_arr, pred_arr, paths_arr, strict=True
    ):
        rows_read += 1
        clip_id_str = str(clip_id)
        action_id_int = int(action_id)
        if action_id_int not in action_mapping:
            raise ValueError(f"Model predicted unknown action_id {action_id_int} for clip {clip_id_str}")
        if clip_id_str in by_clip:
            raise ValueError(f"Duplicate prediction for {clip_id_str}")
        by_clip[clip_id_str] = ClipPrediction(
            clip_id=clip_id_str,
            action_id=action_id_int,
            action_name=str(action_mapping[action_id_int]),
            submission_path=str(submission_path),
        )
    return PredictionTable(by_clip=by_clip, rows_read=rows_read, blank_predictions=0)


def _extraction_progress_adapter(
    progress_callback: ProgressCallback | None,
    split: str,
) -> Callable[[int, int], None] | None:
    """Bridge the public 3-argument callback to the 2-argument extractor one.

    ``modeling.features.extract_feature_bundle`` calls ``callback(done, total)``.
    The ``generate_*`` helpers expose ``callback(done, total, message)`` so a
    caller driving a progress bar can label the phase without tracking it.
    """

    if progress_callback is None:
        return None
    label = "training" if split == "train" else "test"

    def forward(done: int, total: int) -> None:
        progress_callback(done, total, f"Extracting {label} features: {done:,}/{total:,} clips")

    return forward


def generate_predictions_from_model(
    model_path: str | Path,
    dataset_root: str | Path | None = None,
    test_csv: str | Path | None = None,
    action_mapping: Mapping[int, str] | None = None,
    *,
    split: str = "test",
    n_jobs: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PredictionTable:
    """Generate predictions for one split using a saved model artifact.

    Parameters
    ----------
    model_path:
        Path to ``artifacts/<algorithm>/model.joblib``.
    dataset_root:
        Dataset root containing ``Training/`` and ``Testing/``. When ``None``,
        resolves via ``visualization.dataset.resolve_dataset_root``.
    test_csv:
        Optional override for the test CSV (only used when ``split == "test"``).
    action_mapping:
        Optional ``action_id -> action_name`` mapping. When ``None``, loads
        ``Training/class_mapping.csv`` from the repository or dataset root.
    split:
        Either ``"train"`` or ``"test"``. ``"train"`` predicts on the filtered
        training clips (those with all four modalities and five IMU devices);
        ``"test"`` predicts in submission order.
    n_jobs:
        Parallelism for feature extraction. Defaults to ``min(8, cpu_count)``.
    progress_callback:
        Optional ``callback(done, total, message)`` invoked during feature
        extraction. Exceptions raised by the callback propagate.

    Returns a ``PredictionTable`` whose ``by_clip`` keys are the logical
    clip IDs (``SM_test_XXXX`` for test, ``<action>/<user>/<trial>`` for train)
    and whose values carry the predicted ``action_id`` and ``action_name``.
    """

    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    (
        discover_training_clips,
        discover_test_clips,
        feature_config_from_artifact,
        extract_feature_bundle,
        load_model,
    ) = _require_modeling_stack()

    import os

    from visualization.dataset import resolve_dataset_root

    model = load_model(Path(model_path).expanduser())
    config = feature_config_from_artifact(model)

    if action_mapping is None:
        # Resolve mapping from repository or dataset root.
        repository_root = Path(__file__).resolve().parents[1]
        candidates = [
            repository_root / "Training" / "class_mapping.csv",
            resolve_dataset_root(dataset_root) / "Training" / "class_mapping.csv",
        ]
        mapping_path = next((p for p in candidates if p.is_file()), None)
        if mapping_path is None:
            raise FileNotFoundError("Training/class_mapping.csv not found in repository or dataset root")
        action_mapping = load_action_mapping(mapping_path)

    dataset_root_resolved = resolve_dataset_root(dataset_root)
    effective_n_jobs = int(n_jobs) if n_jobs is not None else min(8, os.cpu_count() or 1)

    extraction_progress = _extraction_progress_adapter(progress_callback, split)

    if split == "train":
        records, _ = discover_training_clips(dataset_root_resolved)
        bundle = extract_feature_bundle(
            records, config, n_jobs=effective_n_jobs, progress_callback=extraction_progress
        )
        predictions = model.predict(bundle)
        return _prediction_table_from_arrays(
            bundle.clip_ids,
            predictions,
            None,
            action_mapping,
        )
    # test split
    # ``discover_test_clips`` accepts an optional test_csv override.
    if test_csv is not None:
        records_test = discover_test_clips(dataset_root_resolved, Path(test_csv).expanduser())
    else:
        records_test = discover_test_clips(dataset_root_resolved)
    bundle_test = extract_feature_bundle(
        records_test, config, n_jobs=effective_n_jobs, progress_callback=extraction_progress
    )
    predictions_test = model.predict(bundle_test)
    return _prediction_table_from_arrays(
        bundle_test.clip_ids,
        predictions_test,
        bundle_test.submission_paths,
        action_mapping,
    )


def generate_all_split_predictions(
    model_path: str | Path,
    dataset_root: str | Path | None = None,
    test_csv: str | Path | None = None,
    action_mapping: Mapping[int, str] | None = None,
    *,
    n_jobs: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, PredictionTable]:
    """Generate predictions for both training and test splits.

    Returns ``{"train": PredictionTable, "test": PredictionTable}``. This is a
    convenience wrapper around ``generate_predictions_from_model`` that reuses
    the resolved ``action_mapping`` and ``dataset_root`` for both calls.

    ``progress_callback(done, total, message)`` is rescaled onto a single
    monotonic 0-100 range across both splits: setup 0-5, training extraction
    5-70, test extraction 70-95, finalization 95-100.
    """

    # Resolve mapping once so both splits share the same object and error
    # handling (avoids loading the CSV twice and keeps the two tables
    # consistent when the caller passes an explicit mapping).
    if action_mapping is None:
        from visualization.dataset import resolve_dataset_root

        repository_root = Path(__file__).resolve().parents[1]
        candidates = [
            repository_root / "Training" / "class_mapping.csv",
            resolve_dataset_root(dataset_root) / "Training" / "class_mapping.csv",
        ]
        mapping_path = next((p for p in candidates if p.is_file()), None)
        if mapping_path is None:
            raise FileNotFoundError("Training/class_mapping.csv not found in repository or dataset root")
        action_mapping = load_action_mapping(mapping_path)

    def rescaled(start: int, span: int) -> ProgressCallback | None:
        """Map one split's 0-100% extraction progress onto ``start..start+span``."""

        if progress_callback is None:
            return None

        def forward(done: int, total: int, message: str) -> None:
            progress_callback(start + int(span * done / max(1, total)), 100, message)

        return forward

    train_table = generate_predictions_from_model(
        model_path,
        dataset_root,
        test_csv,
        action_mapping,
        split="train",
        n_jobs=n_jobs,
        progress_callback=rescaled(5, 65),
    )
    if progress_callback is not None:
        # Training extraction ends at 5 + 65 = 70, which is exactly where test
        # extraction starts, so the bar never moves backwards.
        progress_callback(70, 100, "Running model prediction on training features...")
    test_table = generate_predictions_from_model(
        model_path,
        dataset_root,
        test_csv,
        action_mapping,
        split="test",
        n_jobs=n_jobs,
        progress_callback=rescaled(70, 25),
    )
    if progress_callback is not None:
        progress_callback(95, 100, "Finalizing predictions...")
    return {"train": train_table, "test": test_table}


def combine_prediction_tables(
    tables: Mapping[str, PredictionTable],
) -> PredictionTable:
    """Combine split-specific prediction tables without losing full clip IDs."""

    combined: dict[str, ClipPrediction] = {}
    for table in tables.values():
        overlap = set(combined) & set(table.by_clip)
        if overlap:
            duplicate = sorted(overlap)[0]
            raise ValueError(f"Duplicate prediction for {duplicate}")
        combined.update(table.by_clip)
    return PredictionTable(
        by_clip=combined,
        rows_read=sum(table.rows_read for table in tables.values()),
        blank_predictions=sum(table.blank_predictions for table in tables.values()),
    )


def prediction_csv_text(
    table: PredictionTable,
    *,
    identifier: str = "path",
) -> str:
    """Render a ``PredictionTable`` as a reloadable prediction CSV.

    ``identifier="path"`` preserves the competition submission format and is
    appropriate for test-only predictions. ``identifier="clip_id"`` preserves
    hierarchical training IDs and supports train-only or combined files.
    """

    if identifier not in {"path", "clip_id"}:
        raise ValueError("identifier must be 'path' or 'clip_id'")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=(identifier, "prediction"))
    writer.writeheader()
    for prediction in table.by_clip.values():
        writer.writerow(
            {
                identifier: (
                    prediction.submission_path
                    if identifier == "path"
                    else prediction.clip_id
                ),
                "prediction": int(prediction.action_id),
            }
        )
    return buffer.getvalue()


def save_prediction_csv(
    table: PredictionTable,
    output_path: str | Path,
    *,
    identifier: str = "path",
) -> Path:
    """Write a prediction table using a submission path or exact clip ID."""

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        prediction_csv_text(table, identifier=identifier),
        encoding="utf-8",
        newline="",
    )
    return path
