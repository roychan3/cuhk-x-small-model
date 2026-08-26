#!/usr/bin/env python3
"""Create a compact CUHK-X dataset for exercising the dashboard and trainer."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path.home() / "AAI" / "opt-scratch" / "small-model"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "artifacts" / "sample_dataset"

TRAINING_CLIPS = (
    "7_Eat_food/user1/1-2-1",
    "7_Eat_food/user1/1-2-2",
    "7_Eat_food/user2/1-2-1",
    "7_Eat_food/user2/1-2-2",
    "8_Take_and_use_tableware/user1/1-2-1",
    "8_Take_and_use_tableware/user1/1-2-2",
    "8_Take_and_use_tableware/user2/1-2-1",
    "8_Take_and_use_tableware/user2/1-2-2",
)
TEST_CLIPS = ("SM_test_0001", "SM_test_0002")
FILE_LIMITS: dict[str, int | None] = {
    "Depth_Color": 6,
    "IR": 6,
    "IMU": None,
    "Skeleton": 8,
    "Thermal": 6,
    "Radar": None,
}
REQUIRED_TRAINING_MODALITIES = {"Depth_Color", "IR", "IMU", "Skeleton"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def evenly_spaced(paths: list[Path], limit: int | None) -> list[Path]:
    """Keep representative files across a clip rather than only its beginning."""

    if limit is None or len(paths) <= limit:
        return paths
    if limit == 1:
        return [paths[len(paths) // 2]]
    indices = {round(index * (len(paths) - 1) / (limit - 1)) for index in range(limit)}
    return [paths[index] for index in sorted(indices)]


def copy_clip(source: Path, destination: Path, limit: int | None) -> int:
    files = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    )
    selected = evenly_spaced(files, limit)
    for path in selected:
        output = destination / path.relative_to(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, output)
    return len(selected)


def main() -> int:
    args = parse_args()
    source_root = args.source.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source_root}")
    if output_root.exists():
        raise FileExistsError(
            f"Sample dataset already exists: {output_root}. Remove or rename it first."
        )

    staging = output_root.with_name(f".{output_root.name}.building")
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")

    copied_files: dict[str, int] = {}
    try:
        training_source = source_root / "Training" / "data" / "HAR" / "data"
        training_output = staging / "Training" / "data" / "HAR" / "data"
        for modality, limit in FILE_LIMITS.items():
            for clip_id in TRAINING_CLIPS:
                source = training_source / modality / clip_id
                if not source.is_dir():
                    if modality in REQUIRED_TRAINING_MODALITIES:
                        raise FileNotFoundError(
                            f"Missing required sample source: {source}"
                        )
                    continue
                copied_files[f"train/{modality}/{clip_id}"] = copy_clip(
                    source, training_output / modality / clip_id, limit
                )

        testing_source = source_root / "Testing" / "data" / "small_model_track_test"
        testing_output = staging / "Testing" / "data" / "small_model_track_test"
        for clip_id in TEST_CLIPS:
            for modality, limit in FILE_LIMITS.items():
                source = testing_source / clip_id / modality
                if not source.is_dir():
                    continue
                copied_files[f"test/{clip_id}/{modality}"] = copy_clip(
                    source, testing_output / clip_id / modality, limit
                )

        mapping_source = REPOSITORY_ROOT / "Training" / "class_mapping.csv"
        mapping_output = staging / "Training" / "class_mapping.csv"
        mapping_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mapping_source, mapping_output)

        test_csv = staging / "Testing" / "test.csv"
        test_csv.parent.mkdir(parents=True, exist_ok=True)
        with test_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("path", "prediction"))
            writer.writeheader()
            for clip_id in TEST_CLIPS:
                writer.writerow(
                    {"path": f"small_model_track_test/{clip_id}/", "prediction": ""}
                )

        metadata = {
            "source": str(source_root),
            "training_clips": list(TRAINING_CLIPS),
            "test_clips": list(TEST_CLIPS),
            "file_limits_per_clip": FILE_LIMITS,
            "copied_files": copied_files,
        }
        (staging / "sample.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        (staging / "README.md").write_text(
            "# CUHK-X UI sample\n\n"
            "A compact two-class, two-user fixture generated by "
            "`scripts/prepare_sample_dataset.py`. It supports the dataset explorer, "
            "feature extraction, and two-fold grouped training.\n",
            encoding="utf-8",
        )
        output_root.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Created sample dataset at {output_root}")
    print(f"Training clips: {len(TRAINING_CLIPS)}")
    print(f"Test clips: {len(TEST_CLIPS)}")
    print(f"Files copied: {sum(copied_files.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
