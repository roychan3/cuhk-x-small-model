"""Dataset discovery for the multimodal logistic-regression baseline."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from visualization.dataset import MODALITIES, resolve_dataset_root


EXPECTED_IMU_DEVICES = ("WTLA", "WTRA", "WTC", "WTLL", "WTRL")
REQUIRED_MODALITIES = ("Depth_Color", "IR", "IMU", "Skeleton")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ClipRecord:
    """One logical clip and the directories containing its modalities."""

    split: str
    clip_id: str
    depth_dir: Path
    ir_dir: Path
    imu_dir: Path
    skeleton_dir: Path
    label: int | None = None
    user: str | None = None
    submission_path: str | None = None


@dataclass(frozen=True)
class DiscoveryReport:
    """Counts describing complete-case filtering."""

    logical_clips: int
    modality_complete: int
    nonempty_imu_files: int
    all_imu_devices: int


def _clip_directories(modality_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    if not modality_root.is_dir():
        return result
    for path in modality_root.glob("*/*/*"):
        if path.is_dir():
            result[path.relative_to(modality_root).as_posix()] = path
    return result


def _has_nonempty_file(directory: Path, suffixes: set[str] | None = None) -> bool:
    if not directory.is_dir():
        return False
    for path in directory.rglob("*"):
        if not path.is_file() or path.name.startswith(".") or path.stat().st_size == 0:
            continue
        if suffixes is None or path.suffix.lower() in suffixes:
            return True
    return False


def inspect_imu_directory(directory: Path) -> tuple[set[str], int, int]:
    """Return device prefixes, total data rows, and nonempty file count."""

    devices: set[str] = set()
    row_count = 0
    nonempty_files = 0
    if not directory.is_dir():
        return devices, row_count, nonempty_files

    for path in sorted(directory.glob("*.csv")):
        rows_in_file = 0
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                rows_in_file += 1
                devices.add(row[1].split("(", 1)[0].strip())
        row_count += rows_in_file
        nonempty_files += int(rows_in_file > 0)
    return devices, row_count, nonempty_files


def discover_training_clips(
    dataset_root: str | Path | None = None,
) -> tuple[list[ClipRecord], DiscoveryReport]:
    """Return clips with all four modalities and all five IMU devices.

    The strict filter removes clips with a missing modality, an empty upper- or
    lower-body IMU file, or an absent expected IMU device.
    """

    root = resolve_dataset_root(dataset_root)
    data_root = root / "Training" / "data" / "HAR" / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Extracted training data not found at {data_root}. "
            "Run scripts/prepare_training_data.sh first."
        )

    directories = {name: _clip_directories(data_root / name) for name in MODALITIES}
    logical_keys = set().union(*(set(values) for values in directories.values()))
    complete_keys = set.intersection(*(set(directories[name]) for name in REQUIRED_MODALITIES))

    records: list[ClipRecord] = []
    nonempty_imu_count = 0
    all_device_count = 0
    expected = set(EXPECTED_IMU_DEVICES)

    for key in sorted(complete_keys):
        depth_dir = directories["Depth_Color"][key]
        ir_dir = directories["IR"][key]
        imu_dir = directories["IMU"][key]
        skeleton_dir = directories["Skeleton"][key]
        if not _has_nonempty_file(depth_dir, IMAGE_SUFFIXES):
            continue
        if not _has_nonempty_file(ir_dir, IMAGE_SUFFIXES):
            continue
        if not _has_nonempty_file(skeleton_dir, {".json"}):
            continue

        devices, _, nonempty_files = inspect_imu_directory(imu_dir)
        if nonempty_files >= 2:
            nonempty_imu_count += 1
        else:
            continue
        if not expected.issubset(devices):
            continue
        all_device_count += 1

        action, user, _trial = key.split("/", 2)
        records.append(
            ClipRecord(
                split="train",
                clip_id=key,
                depth_dir=depth_dir,
                ir_dir=ir_dir,
                imu_dir=imu_dir,
                skeleton_dir=skeleton_dir,
                label=int(action.split("_", 1)[0]),
                user=user,
            )
        )

    records.sort(key=lambda row: (row.label if row.label is not None else 10_000, row.clip_id))
    report = DiscoveryReport(
        logical_clips=len(logical_keys),
        modality_complete=len(complete_keys),
        nonempty_imu_files=nonempty_imu_count,
        all_imu_devices=all_device_count,
    )
    return records, report


def _read_submission_rows(test_csv: Path) -> list[dict[str, str]]:
    with test_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "path" not in rows[0]:
        raise ValueError(f"Test CSV must contain a path column: {test_csv}")
    return rows


def discover_test_clips(
    dataset_root: str | Path | None = None,
    test_csv: str | Path | None = None,
) -> list[ClipRecord]:
    """Return test clips in submission order.

    Missing test modalities are retained so the trained preprocessing pipeline
    can mean-impute their raw feature blocks.
    """

    root = resolve_dataset_root(dataset_root)
    data_root = root / "Testing" / "data" / "small_model_track_test"
    if not data_root.is_dir():
        raise FileNotFoundError(f"Extracted test data not found at {data_root}")

    if test_csv is None:
        repository_csv = Path(__file__).resolve().parents[1] / "Testing" / "test.csv"
        dataset_csv = root / "Testing" / "test.csv"
        test_path = repository_csv if repository_csv.is_file() else dataset_csv
    else:
        test_path = Path(test_csv).expanduser()

    records: list[ClipRecord] = []
    for row in _read_submission_rows(test_path):
        submission_path = row["path"]
        clip_id = PurePosixPath(submission_path.rstrip("/")).name
        clip_root = data_root / clip_id
        if not clip_root.is_dir():
            raise FileNotFoundError(f"Test clip directory not found: {clip_root}")
        records.append(
            ClipRecord(
                split="test",
                clip_id=clip_id,
                depth_dir=clip_root / "Depth_Color",
                ir_dir=clip_root / "IR",
                imu_dir=clip_root / "IMU",
                skeleton_dir=clip_root / "Skeleton",
                submission_path=submission_path,
            )
        )
    return records


def class_counts(records: Iterable[ClipRecord]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for record in records:
        if record.label is not None:
            counts[record.label] = counts.get(record.label, 0) + 1
    return dict(sorted(counts.items()))
