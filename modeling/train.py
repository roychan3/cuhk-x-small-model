#!/usr/bin/env python3
"""Extract features, compare parameters, train an algorithm, and submit."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from modeling.algorithms import available_algorithms, get_algorithm
from modeling.cache import load_feature_cache, save_feature_cache
from modeling.data import class_counts, discover_test_clips, discover_training_clips
from modeling.features import FeatureConfig, RawFeatureBundle, extract_feature_bundle
from modeling.model import (
    cross_validate_detailed,
    fit_final_model,
    library_versions,
    save_model,
    save_validation_outputs,
)
from visualization.dataset import resolve_dataset_root
from visualization.progress import COMPLETE, RUNNING, mark_error, write_progress


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_CACHE = REPOSITORY_ROOT / "artifacts" / "features" / "four_sensor_v3.npz"


def parse_args(default_algorithm: str = "logistic_regression") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        choices=available_algorithms(),
        default=default_algorithm,
        help="Registered estimator to train after shared multimodal preprocessing.",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=REPOSITORY_ROOT / "Testing" / "test.csv")
    parser.add_argument(
        "--features-cache",
        type=Path,
        default=DEFAULT_FEATURE_CACHE,
        help="Algorithm-independent raw feature cache shared by comparisons.",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--cv-repeats",
        type=int,
        default=1,
        help="Repeat participant-grouped cross-validation with consecutive seeds.",
    )
    parser.add_argument(
        "--search-space",
        default=None,
        help=(
            "JSON object whose values are candidate lists. Defaults to the "
            "selected algorithm's search space."
        ),
    )
    parser.add_argument(
        "--parameters",
        default=None,
        help="JSON parameters used with --skip-validation.",
    )
    parser.add_argument("--selection-metric", default="macro_f1")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help="Optional JSON status file for dashboard and automation progress.",
    )
    arguments = parser.parse_args()
    # cross_validate_detailed() rejects these too, but --skip-validation never
    # reaches it, so a nonsense value would otherwise be accepted in silence.
    if arguments.folds < 2:
        parser.error("--folds must be at least 2")
    if arguments.cv_repeats < 1:
        parser.error("--cv-repeats must be at least 1")
    return arguments


def _parse_json_mapping(value: str | None, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _search_space(value: str | None) -> Mapping[str, Sequence[Any]] | None:
    parsed = _parse_json_mapping(value, "--search-space")
    if parsed is None:
        return None
    for name, candidates in parsed.items():
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Search-space value for {name!r} must be a nonempty JSON list")
    return parsed


def _cache_matches(
    train: RawFeatureBundle,
    test: RawFeatureBundle,
    training_ids: list[str],
    test_ids: list[str],
) -> bool:
    return np.array_equal(train.clip_ids, np.asarray(training_ids)) and np.array_equal(
        test.clip_ids, np.asarray(test_ids)
    )


def _write_submission(
    path: Path,
    submission_paths: np.ndarray,
    predictions: np.ndarray,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "prediction"))
        writer.writeheader()
        for source_path, prediction in zip(submission_paths, predictions, strict=True):
            writer.writerow({"path": str(source_path), "prediction": int(prediction)})
    return path


def _feature_health(
    train: RawFeatureBundle,
    test: RawFeatureBundle,
) -> dict[str, dict[str, int]]:
    health: dict[str, dict[str, int]] = {}
    for name, values in train.modality_arrays().items():
        if not np.isfinite(values).all():
            bad_rows = int((~np.isfinite(values).all(axis=1)).sum())
            raise ValueError(f"Strict training features contain {bad_rows} invalid {name} rows")
    for name, values in test.modality_arrays().items():
        finite = np.isfinite(values)
        all_missing = ~finite.any(axis=1)
        partial = ~finite.all(axis=1) & ~all_missing
        health[name] = {
            "complete_rows": int(finite.all(axis=1).sum()),
            "partial_rows": int(partial.sum()),
            "all_missing_rows": int(all_missing.sum()),
        }
        if partial.any() or all_missing.any():
            print(
                f"Test {name} fallback: {int(partial.sum())} partial and "
                f"{int(all_missing.sum())} all-missing feature rows will be mean-imputed."
            )
    return health


def main(default_algorithm: str = "logistic_regression") -> int:
    args = parse_args(default_algorithm)
    progress_path = args.progress_file.expanduser() if args.progress_file is not None else None
    try:
        return _run(args, progress_path)
    except Exception as exc:
        # A reader polling only the progress file cannot see the exit status,
        # so the failure has to be recorded before the traceback propagates.
        # A hard kill still leaves the file mid-stage; `updated_at` is what
        # reveals that.
        mark_error(progress_path, f"{type(exc).__name__}: {exc}")
        raise


def _run(args: argparse.Namespace, progress_path: Path | None) -> int:
    algorithm = get_algorithm(args.algorithm)
    dataset_root = resolve_dataset_root(args.dataset_root)
    write_progress(
        progress_path,
        status=RUNNING,
        stage="discovery",
        message="Discovering training and test clips",
    )
    config = FeatureConfig()
    artifacts_dir = (
        args.artifacts_dir.expanduser()
        if args.artifacts_dir is not None
        else REPOSITORY_ROOT / "artifacts" / algorithm.artifact_name
    )
    output_path = (
        args.output.expanduser()
        if args.output is not None
        else REPOSITORY_ROOT / "outputs" / f"{algorithm.artifact_name}_submission.csv"
    )
    cache_path = args.features_cache.expanduser()
    model_path = artifacts_dir / "model.joblib"
    report_path = artifacts_dir / "validation.json"
    oof_path = artifacts_dir / "oof_predictions.npz"

    print(f"Algorithm: {algorithm.display_name} ({algorithm.name})")
    print(f"Dataset root: {dataset_root}")
    training_records, discovery = discover_training_clips(dataset_root)
    test_records = discover_test_clips(dataset_root, args.test_csv)
    counts = class_counts(training_records)
    print(
        "Training filter: "
        f"{discovery.logical_clips:,} logical -> "
        f"{discovery.modality_complete:,} four-modality -> "
        f"{discovery.nonempty_imu_files:,} nonempty IMU files -> "
        f"{len(training_records):,} all five IMU devices"
    )
    print(
        f"Retained {len(counts)} classes; class counts range "
        f"from {min(counts.values())} to {max(counts.values())}."
    )
    print(f"Test clips: {len(test_records):,}")
    total_feature_clips = len(training_records) + len(test_records)
    write_progress(
        progress_path,
        status=RUNNING,
        stage="features",
        message="Preparing the shared feature cache",
        current=0,
        total=total_feature_clips,
    )

    train_bundle: RawFeatureBundle
    test_bundle: RawFeatureBundle
    use_cache = cache_path.is_file() and not args.rebuild_features
    if use_cache:
        try:
            train_bundle, test_bundle = load_feature_cache(cache_path, config)
            if not _cache_matches(
                train_bundle,
                test_bundle,
                [record.clip_id for record in training_records],
                [record.clip_id for record in test_records],
            ):
                raise ValueError("Feature cache clip order does not match discovered clips")
            print(f"Loaded shared feature cache: {cache_path}")
            write_progress(
                progress_path,
                status=RUNNING,
                stage="features",
                message="Loaded the compatible shared feature cache",
                current=total_feature_clips,
                total=total_feature_clips,
            )
        except (KeyError, OSError, ValueError) as exc:
            print(f"Ignoring incompatible feature cache: {exc}")
            use_cache = False

    if not use_cache:
        print("Extracting training features...")
        train_bundle = extract_feature_bundle(
            training_records,
            config,
            n_jobs=args.n_jobs,
            progress_callback=lambda done, total: write_progress(
                progress_path,
                status=RUNNING,
                stage="features",
                message=f"Extracting training features: {done:,}/{total:,}",
                current=done,
                total=total_feature_clips,
            ),
        )
        print("Extracting test features...")
        test_bundle = extract_feature_bundle(
            test_records,
            config,
            n_jobs=args.n_jobs,
            progress_callback=lambda done, total: write_progress(
                progress_path,
                status=RUNNING,
                stage="features",
                message=f"Extracting test features: {done:,}/{total:,}",
                current=len(training_records) + done,
                total=total_feature_clips,
            ),
        )
        save_feature_cache(cache_path, train_bundle, test_bundle, config)
        print(f"Saved shared feature cache: {cache_path}")
        write_progress(
            progress_path,
            status=RUNNING,
            stage="features",
            message="Saved the shared feature cache",
            current=total_feature_clips,
            total=total_feature_clips,
        )

    test_feature_health = _feature_health(train_bundle, test_bundle)
    if args.extract_only:
        write_progress(
            progress_path,
            status=COMPLETE,
            stage="complete",
            message="Feature extraction complete",
            outputs={"feature_cache": str(cache_path)},
        )
        return 0

    if args.skip_validation:
        write_progress(
            progress_path,
            status=RUNNING,
            stage="validation",
            message="Skipping cross-validation",
        )
        selected_parameters = algorithm.resolved_parameters(
            _parse_json_mapping(args.parameters, "--parameters")
        )
        validation_outputs = None
        report: dict[str, object] = {
            "algorithm": algorithm.name,
            "selection_metric": None,
            "selected_parameters": selected_parameters,
            "validation_skipped": True,
        }
    else:
        candidates = algorithm.parameter_candidates(_search_space(args.search_space))
        validation_steps = args.folds * args.cv_repeats
        write_progress(
            progress_path,
            status=RUNNING,
            stage="validation",
            message="Running participant-grouped cross-validation",
            current=0,
            total=validation_steps,
        )
        selected_parameters, report, validation_outputs = cross_validate_detailed(
            train_bundle,
            algorithm,
            candidates,
            n_splits=args.folds,
            n_repeats=args.cv_repeats,
            random_state=args.random_state,
            selection_metric=args.selection_metric,
            progress_callback=lambda done, total: write_progress(
                progress_path,
                status=RUNNING,
                stage="validation",
                message=f"Completed validation fold {done:,}/{total:,}",
                current=done,
                total=total,
            ),
        )
        print(
            f"Selected parameters by {args.selection_metric}: "
            f"{json.dumps(selected_parameters, sort_keys=True)}"
        )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if validation_outputs is not None:
        save_validation_outputs(validation_outputs, oof_path)
        report["oof_predictions"] = str(oof_path)
    report.update(
        {
            "algorithm": algorithm.name,
            "algorithm_display_name": algorithm.display_name,
            "dataset_root": str(dataset_root),
            "training_clips": len(training_records),
            "test_clips": len(test_records),
            "class_counts": counts,
            "raw_feature_dimensions": {
                name: int(values.shape[1])
                for name, values in train_bundle.modality_arrays().items()
            },
            "test_feature_health": test_feature_health,
            "library_versions": library_versions(),
        }
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("Fitting final model...")
    write_progress(
        progress_path,
        status=RUNNING,
        stage="training",
        message="Fitting the final model",
    )
    model = fit_final_model(
        train_bundle,
        algorithm,
        parameters=selected_parameters,
        feature_config=config,
        random_state=args.random_state,
    )
    save_model(model, model_path)
    write_progress(
        progress_path,
        status=RUNNING,
        stage="saving",
        message="Generating and saving test predictions",
    )
    predictions = model.predict(test_bundle)
    assert test_bundle.submission_paths is not None
    _write_submission(output_path, test_bundle.submission_paths, predictions)
    print(f"Saved model: {model_path}")
    print(f"Saved validation report: {report_path}")
    if validation_outputs is not None:
        print(f"Saved out-of-fold predictions: {oof_path}")
    print(f"Saved submission: {output_path}")
    outputs = {
        "feature_cache": str(cache_path),
        "model": str(model_path),
        "validation_report": str(report_path),
        "submission": str(output_path),
    }
    if validation_outputs is not None:
        outputs["oof_predictions"] = str(oof_path)
    write_progress(
        progress_path,
        status=COMPLETE,
        stage="complete",
        message="Training pipeline complete",
        outputs=outputs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
