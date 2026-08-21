#!/usr/bin/env python3
"""Generate a submission from any saved registered-algorithm model."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from modeling.algorithms import available_algorithms, get_algorithm
from modeling.data import discover_test_clips
from modeling.features import FeatureConfig, extract_feature_bundle
from modeling.model import load_model
from visualization.dataset import resolve_dataset_root


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=REPOSITORY_ROOT / "Testing" / "test.csv")
    parser.add_argument(
        "--algorithm",
        choices=available_algorithms(),
        default="logistic_regression",
        help="Algorithm whose saved artifacts to load. Ignored when --model is given.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Model artifact. Defaults to artifacts/<algorithm>/model.joblib.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Submission path. Defaults to outputs/<algorithm>_submission.csv, "
            "named after the algorithm recorded in the loaded model."
        ),
    )
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def _artifact_name(algorithm_name: str) -> str:
    """Return the artifact name for an algorithm, tolerating unregistered names."""

    try:
        return get_algorithm(algorithm_name).artifact_name
    except ValueError:
        return algorithm_name


def main() -> int:
    args = parse_args()
    model_path = (
        args.model
        if args.model is not None
        else REPOSITORY_ROOT / "artifacts" / _artifact_name(args.algorithm) / "model.joblib"
    )
    model = load_model(model_path)
    output_path = (
        args.output
        if args.output is not None
        else REPOSITORY_ROOT
        / "outputs"
        / f"{_artifact_name(model.algorithm_name)}_submission.csv"
    )
    config_values = dict(model.feature_config)
    if "include_legacy_ir" not in config_values:
        config_values["include_legacy_ir"] = "ir" in model.reducers
    config = FeatureConfig(**config_values)
    records = discover_test_clips(resolve_dataset_root(args.dataset_root), args.test_csv)
    bundle = extract_feature_bundle(records, config, n_jobs=args.n_jobs)
    predictions = model.predict(bundle)
    assert bundle.submission_paths is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "prediction"))
        writer.writeheader()
        for path, prediction in zip(bundle.submission_paths, predictions, strict=True):
            writer.writerow({"path": str(path), "prediction": int(prediction)})
    print(
        f"Saved {len(predictions):,} {model.algorithm_name} predictions "
        f"to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
