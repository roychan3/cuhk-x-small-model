"""Fixed-length feature extraction for four CUHK-X modalities."""

from __future__ import annotations

import csv
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
from PIL import Image

from modeling.data import ClipRecord, EXPECTED_IMU_DEVICES


_TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3})")
_FRAME_RE = re.compile(r"(?:frame_|_)(\d+)(?:_Color)?\.(?:jpg|jpeg|png|json)$", re.I)


@dataclass(frozen=True)
class FeatureConfig:
    image_width: int = 64
    image_height: int = 48
    max_image_frames: int = 64
    skeleton_frames: int = 32
    temporal_bins: int = 4
    hog_orientations: int = 9
    hog_pixels_per_cell: tuple[int, int] = (8, 8)
    hog_cells_per_block: tuple[int, int] = (2, 2)


@dataclass(frozen=True)
class RawFeatureBundle:
    clip_ids: np.ndarray
    depth: np.ndarray
    ir: np.ndarray
    imu: np.ndarray
    skeleton: np.ndarray
    labels: np.ndarray | None = None
    groups: np.ndarray | None = None
    submission_paths: np.ndarray | None = None

    def modality_arrays(self) -> dict[str, np.ndarray]:
        return {
            "depth": self.depth,
            "ir": self.ir,
            "imu": self.imu,
            "skeleton": self.skeleton,
        }


def _path_sort_key(path: Path) -> tuple[str, int, str]:
    timestamp = _TIMESTAMP_RE.search(path.name)
    frame = _FRAME_RE.search(path.name)
    return (
        timestamp.group(1) if timestamp else "",
        int(frame.group(1)) if frame else -1,
        path.name,
    )


def _evenly_sample(paths: Sequence[Path], maximum: int) -> list[Path]:
    if len(paths) <= maximum:
        return list(paths)
    indices = np.linspace(0, len(paths) - 1, num=maximum).round().astype(int)
    return [paths[index] for index in indices]


def _load_image_frames(directory: Path, config: FeatureConfig) -> np.ndarray | None:
    paths = sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ),
        key=_path_sort_key,
    )
    paths = _evenly_sample(paths, config.max_image_frames)
    frames: list[np.ndarray] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image = image.convert("L").resize(
                    (config.image_width, config.image_height),
                    Image.Resampling.BILINEAR,
                )
                frames.append(np.asarray(image, dtype=np.float32) / 255.0)
        except (OSError, ValueError):
            continue
    return np.stack(frames) if frames else None


def _image_summaries(frames: np.ndarray) -> np.ndarray:
    if len(frames) > 1:
        motion = np.mean(np.abs(np.diff(frames, axis=0)), axis=0)
    else:
        motion = np.zeros_like(frames[0])
    return np.stack(
        (
            frames[0],
            frames[-1],
            np.mean(frames, axis=0),
            np.std(frames, axis=0),
            motion,
        )
    ).astype(np.float32)


def _coarse_pool(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    if height % 12 != 0 or width % 16 != 0:
        raise ValueError("Image dimensions must be divisible into a 12x16 coarse grid")
    return image.reshape(12, height // 12, 16, width // 16).mean(axis=(1, 3))


def _hog(image: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """Compute unsigned-gradient HOG with L2-Hys block normalization."""

    cell_height, cell_width = config.hog_pixels_per_cell
    block_height, block_width = config.hog_cells_per_block
    height, width = image.shape
    if height % cell_height or width % cell_width:
        raise ValueError("Image dimensions must be divisible by the HOG cell size")

    gradient_x = np.zeros_like(image, dtype=np.float32)
    gradient_y = np.zeros_like(image, dtype=np.float32)
    gradient_x[:, 1:-1] = image[:, 2:] - image[:, :-2]
    gradient_x[:, 0] = image[:, 1] - image[:, 0]
    gradient_x[:, -1] = image[:, -1] - image[:, -2]
    gradient_y[1:-1] = image[2:] - image[:-2]
    gradient_y[0] = image[1] - image[0]
    gradient_y[-1] = image[-1] - image[-2]

    magnitude = np.hypot(gradient_x, gradient_y)
    angle = np.mod(np.arctan2(gradient_y, gradient_x), np.pi)
    bin_position = angle * (config.hog_orientations / np.pi)
    lower_bin = np.floor(bin_position).astype(np.int32) % config.hog_orientations
    upper_bin = (lower_bin + 1) % config.hog_orientations
    upper_weight = bin_position - np.floor(bin_position)
    lower_weight = 1.0 - upper_weight

    cell_rows = height // cell_height
    cell_columns = width // cell_width
    row_cells = np.repeat(np.arange(cell_rows), cell_height)[:, None]
    column_cells = np.repeat(np.arange(cell_columns), cell_width)[None, :]
    cell_indices = row_cells * cell_columns + column_cells
    histogram = np.zeros((cell_rows * cell_columns, config.hog_orientations), dtype=np.float32)
    np.add.at(
        histogram,
        (cell_indices.ravel(), lower_bin.ravel()),
        (magnitude * lower_weight).ravel(),
    )
    np.add.at(
        histogram,
        (cell_indices.ravel(), upper_bin.ravel()),
        (magnitude * upper_weight).ravel(),
    )
    histogram = histogram.reshape(cell_rows, cell_columns, config.hog_orientations)

    blocks: list[np.ndarray] = []
    epsilon = 1e-5
    for row in range(cell_rows - block_height + 1):
        for column in range(cell_columns - block_width + 1):
            block = histogram[
                row : row + block_height,
                column : column + block_width,
            ].ravel()
            block = block / np.sqrt(np.dot(block, block) + epsilon * epsilon)
            block = np.minimum(block, 0.2)
            block = block / np.sqrt(np.dot(block, block) + epsilon * epsilon)
            blocks.append(block)
    return np.concatenate(blocks).astype(np.float32)


def image_feature_size(config: FeatureConfig) -> int:
    dummy = np.zeros((config.image_height, config.image_width), dtype=np.float32)
    hog_size = _hog(dummy, config).size
    return 5 * (hog_size + 12 * 16 + 5)


def extract_image_features(directory: Path, config: FeatureConfig) -> np.ndarray:
    frames = _load_image_frames(directory, config)
    if frames is None:
        return np.full(image_feature_size(config), np.nan, dtype=np.float32)

    features: list[np.ndarray] = []
    for summary in _image_summaries(frames):
        hog_features = _hog(summary, config)
        coarse = _coarse_pool(summary).ravel().astype(np.float32)
        percentiles = np.percentile(summary, (0, 25, 50, 75, 100)).astype(np.float32)
        features.extend((hog_features, coarse, percentiles))
    return np.concatenate(features).astype(np.float32)


def _parse_time(value: str, fallback: int) -> float:
    text = value.strip()
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return float(fallback)


def _safe_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_imu(directory: Path) -> dict[str, np.ndarray]:
    rows_by_device: dict[str, list[tuple[float, np.ndarray]]] = {
        device: [] for device in EXPECTED_IMU_DEVICES
    }
    sequence = 0
    for path in sorted(directory.glob("*.csv")):
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) < 8:
                    continue
                device = row[1].split("(", 1)[0].strip()
                if device not in rows_by_device:
                    continue
                values = [_safe_float(value) for value in row[2:8]]
                if any(value is None for value in values):
                    continue
                vector = np.asarray(values, dtype=np.float32)
                acc_magnitude = float(np.linalg.norm(vector[:3]))
                gyro_magnitude = float(np.linalg.norm(vector[3:]))
                enriched = np.concatenate(
                    (vector, np.asarray((acc_magnitude, gyro_magnitude), dtype=np.float32))
                )
                rows_by_device[device].append((_parse_time(row[0], sequence), enriched))
                sequence += 1

    result: dict[str, np.ndarray] = {}
    for device, rows in rows_by_device.items():
        rows.sort(key=lambda item: item[0])
        result[device] = (
            np.stack([values for _, values in rows])
            if rows
            else np.empty((0, 8), dtype=np.float32)
        )
    return result


def _signal_features(values: np.ndarray, temporal_bins: int) -> np.ndarray:
    if values.size == 0:
        return np.full(10 + temporal_bins * 2, np.nan, dtype=np.float32)
    difference = np.diff(values)
    if len(values) > 1:
        slope = float(np.polyfit(np.linspace(0.0, 1.0, len(values)), values, 1)[0])
    else:
        slope = 0.0
    global_features = np.asarray(
        (
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.percentile(values, 25),
            np.median(values),
            np.percentile(values, 75),
            np.sqrt(np.mean(np.square(values))),
            np.mean(np.abs(difference)) if difference.size else 0.0,
            slope,
        ),
        dtype=np.float32,
    )
    temporal: list[float] = []
    temporal_values = values
    if len(values) < temporal_bins:
        temporal_values = np.interp(
            np.linspace(0, max(0, len(values) - 1), temporal_bins),
            np.arange(len(values)),
            values,
        )
    for chunk in np.array_split(temporal_values, temporal_bins):
        temporal.extend((float(np.mean(chunk)), float(np.std(chunk))))
    return np.concatenate((global_features, np.asarray(temporal, dtype=np.float32)))


def imu_feature_size(config: FeatureConfig) -> int:
    return len(EXPECTED_IMU_DEVICES) * 8 * (10 + config.temporal_bins * 2)


def extract_imu_features(directory: Path, config: FeatureConfig) -> np.ndarray:
    devices = _read_imu(directory)
    features = [
        _signal_features(devices[device][:, channel], config.temporal_bins)
        for device in EXPECTED_IMU_DEVICES
        for channel in range(8)
    ]
    return np.concatenate(features).astype(np.float32)


def _valid_people(payload: object) -> list[tuple[np.ndarray, np.ndarray]]:
    if not isinstance(payload, list):
        return []
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for person in payload:
        if not isinstance(person, dict):
            continue
        points = np.asarray(person.get("keypoints", []), dtype=np.float32)
        if points.shape != (17, 3) or not np.isfinite(points).all():
            continue
        scores = np.asarray(person.get("keypoint_scores", [1.0] * 17), dtype=np.float32)
        if scores.shape != (17,):
            scores = np.ones(17, dtype=np.float32)
        result.append((points, scores))
    return result


def _pose_extent(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))


def _select_person(
    people: list[tuple[np.ndarray, np.ndarray]],
    previous_center: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not people:
        return None
    if previous_center is None:
        return max(people, key=lambda item: (float(np.mean(item[1])), _pose_extent(item[0])))
    return min(
        people,
        key=lambda item: float(np.linalg.norm(np.mean(item[0], axis=0) - previous_center)),
    )


def _interpolate_columns(values: np.ndarray) -> np.ndarray | None:
    result = values.copy()
    index = np.arange(len(values), dtype=np.float32)
    for column in range(values.shape[1]):
        valid = np.isfinite(values[:, column])
        if not valid.any():
            return None
        result[:, column] = np.interp(index, index[valid], values[valid, column])
    return result


def _resample(values: np.ndarray, length: int) -> np.ndarray:
    old = np.linspace(0.0, 1.0, len(values))
    new = np.linspace(0.0, 1.0, length)
    result = np.empty((length, values.shape[1]), dtype=np.float32)
    for column in range(values.shape[1]):
        result[:, column] = np.interp(new, old, values[:, column])
    return result


def skeleton_feature_size(config: FeatureConfig) -> int:
    return config.skeleton_frames * (17 * 3 * 2 + 17)


def extract_skeleton_features(directory: Path, config: FeatureConfig) -> np.ndarray:
    paths = sorted(directory.rglob("*.json"), key=_path_sort_key)
    if not paths:
        return np.full(skeleton_feature_size(config), np.nan, dtype=np.float32)

    poses: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    previous_center: np.ndarray | None = None
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            selected = None
        else:
            selected = _select_person(_valid_people(payload), previous_center)
        if selected is None:
            poses.append(np.full((17, 3), np.nan, dtype=np.float32))
            scores.append(np.full(17, np.nan, dtype=np.float32))
            continue
        points, confidence = selected
        previous_center = np.mean(points, axis=0)
        poses.append(points)
        scores.append(confidence)

    pose_array = np.asarray(poses, dtype=np.float32)
    score_array = np.asarray(scores, dtype=np.float32)
    flat_pose = _interpolate_columns(pose_array.reshape(len(pose_array), -1))
    flat_scores = _interpolate_columns(score_array)
    if flat_pose is None or flat_scores is None:
        return np.full(skeleton_feature_size(config), np.nan, dtype=np.float32)

    pose_array = flat_pose.reshape(-1, 17, 3)
    roots = (pose_array[:, 11] + pose_array[:, 12]) / 2.0
    shoulder_centers = (pose_array[:, 5] + pose_array[:, 6]) / 2.0
    torso_lengths = np.linalg.norm(shoulder_centers - roots, axis=1)
    valid_scale = torso_lengths[np.isfinite(torso_lengths) & (torso_lengths > 1e-6)]
    scale = float(np.median(valid_scale)) if valid_scale.size else 1.0
    normalized = (pose_array - roots[:, None, :]) / max(scale, 1e-6)

    resampled_pose = _resample(normalized.reshape(len(normalized), -1), config.skeleton_frames)
    resampled_scores = _resample(flat_scores, config.skeleton_frames)
    velocity = np.diff(resampled_pose, axis=0, prepend=resampled_pose[:1])
    return np.concatenate(
        (resampled_pose.ravel(), velocity.ravel(), resampled_scores.ravel())
    ).astype(np.float32)


def extract_clip_features(
    record: ClipRecord,
    config: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        extract_image_features(record.depth_dir, config),
        extract_image_features(record.ir_dir, config),
        extract_imu_features(record.imu_dir, config),
        extract_skeleton_features(record.skeleton_dir, config),
    )


def _extract_job(args: tuple[ClipRecord, FeatureConfig]) -> tuple[np.ndarray, ...]:
    return extract_clip_features(*args)


def _progressive_results(
    records: Sequence[ClipRecord],
    config: FeatureConfig,
    n_jobs: int,
) -> Iterator[tuple[np.ndarray, ...]]:
    jobs = ((record, config) for record in records)
    if n_jobs == 1:
        for job in jobs:
            yield _extract_job(job)
        return
    workers = None if n_jobs < 1 else n_jobs
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(_extract_job, jobs, chunksize=4)


def extract_feature_bundle(
    records: Sequence[ClipRecord],
    config: FeatureConfig | None = None,
    *,
    n_jobs: int = 1,
    progress_every: int = 50,
) -> RawFeatureBundle:
    """Extract raw modality matrices while preserving record order."""

    if not records:
        raise ValueError("Cannot extract features from an empty record list")
    active_config = config or FeatureConfig()
    depth: list[np.ndarray] = []
    infrared: list[np.ndarray] = []
    imu: list[np.ndarray] = []
    skeleton: list[np.ndarray] = []
    for index, values in enumerate(
        _progressive_results(records, active_config, n_jobs), start=1
    ):
        depth_value, ir_value, imu_value, skeleton_value = values
        depth.append(depth_value)
        infrared.append(ir_value)
        imu.append(imu_value)
        skeleton.append(skeleton_value)
        if progress_every > 0 and (index % progress_every == 0 or index == len(records)):
            print(f"Extracted {index:,}/{len(records):,} clips", flush=True)

    is_train = records[0].split == "train"
    return RawFeatureBundle(
        clip_ids=np.asarray([record.clip_id for record in records]),
        depth=np.stack(depth),
        ir=np.stack(infrared),
        imu=np.stack(imu),
        skeleton=np.stack(skeleton),
        labels=(
            np.asarray([record.label for record in records], dtype=np.int64)
            if is_train
            else None
        ),
        groups=(
            np.asarray([record.user for record in records])
            if is_train
            else None
        ),
        submission_paths=(
            np.asarray([record.submission_path for record in records])
            if not is_train
            else None
        ),
    )


def feature_config_dict(config: FeatureConfig) -> dict[str, object]:
    return asdict(config)
