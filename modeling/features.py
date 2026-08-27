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
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image

from modeling.data import ClipRecord, EXPECTED_IMU_DEVICES
from visualization.feature_blocks import (
    ALL_FEATURE_BLOCKS,
    DEFAULT_FEATURE_BLOCKS,
    FEATURE_BLOCK_GROUPS,
    normalize_feature_blocks,
)


_TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3})")
_FRAME_RE = re.compile(r"(?:frame_|_)(\d+)(?:_Color)?\.(?:jpg|jpeg|png|json)$", re.I)
_IMU_BASE_CHANNELS = 8
_IMU_EXTENDED_CHANNELS = 18
_DYNAMIC_FEATURES = 9

# Skeleton predictions use the Human3.6M 17-joint order, NOT COCO. Verified
# against the dataset over 1,314 frames from 80 clips: joint 0 sits at exactly
# x == y == 0 in every frame (the root-relative pelvis), joint 10 is the
# highest joint, and joints 3 and 6 are the lowest. Under COCO these indices
# mean nose, right wrist, right ankle and left ankle instead, which turns
# "wrist to head" into a near-rigid trunk length and inverts the torso vector.
# Index these by name so the two orders cannot be confused again.
PELVIS = 0
R_HIP, R_KNEE, R_ANKLE = 1, 2, 3
L_HIP, L_KNEE, L_ANKLE = 4, 5, 6
SPINE, THORAX, NECK, HEAD = 7, 8, 9, 10
L_SHOULDER, L_ELBOW, L_WRIST = 11, 12, 13
R_SHOULDER, R_ELBOW, R_WRIST = 14, 15, 16
SKELETON_JOINTS = 17


@dataclass(frozen=True)
class FeatureConfig:
    image_width: int = 64
    image_height: int = 48
    max_image_frames: int = 64
    image_engineered_grid: tuple[int, int] = (4, 4)
    include_legacy_ir: bool = False
    skeleton_frames: int = 32
    temporal_bins: int = 4
    spectral_samples: int = 64
    hog_orientations: int = 9
    hog_pixels_per_cell: tuple[int, int] = (8, 8)
    hog_cells_per_block: tuple[int, int] = (2, 2)
    # Recorded so an artifact says which joint order its skeleton features were
    # built from. Artifacts written before the COCO-to-H36M correction have no
    # value here, and their skeleton blocks mean something different, so the
    # prediction entry points refuse them rather than silently mixing the two.
    skeleton_layout: str = "h36m"


def filter_bundle_to_blocks(
    bundle: RawFeatureBundle,
    blocks: Sequence[str] | None,
) -> RawFeatureBundle:
    """Return a copy of ``bundle`` carrying only ``blocks``.

    A selected block the bundle does not have is simply absent from the result;
    callers that need every one present should compare against
    ``modality_arrays()`` first.
    """

    if blocks is None:
        return bundle
    selected = set(normalize_feature_blocks(blocks))
    arrays = bundle.modality_arrays()
    return RawFeatureBundle(
        clip_ids=bundle.clip_ids,
        labels=bundle.labels,
        groups=bundle.groups,
        submission_paths=bundle.submission_paths,
        **{
            name: (arrays.get(name) if name in selected else None)
            for name in ALL_FEATURE_BLOCKS
        },
    )


@dataclass(frozen=True)
class RawFeatureBundle:
    clip_ids: np.ndarray
    depth: np.ndarray | None = None
    ir: np.ndarray | None = None
    imu: np.ndarray | None = None
    skeleton: np.ndarray | None = None
    depth_engineered: np.ndarray | None = None
    ir_engineered: np.ndarray | None = None
    imu_engineered: np.ndarray | None = None
    skeleton_engineered: np.ndarray | None = None
    labels: np.ndarray | None = None
    groups: np.ndarray | None = None
    submission_paths: np.ndarray | None = None

    def modality_arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray | None] = {
            "depth": self.depth,
            "ir": self.ir,
            "imu": self.imu,
            "skeleton": self.skeleton,
            "depth_engineered": self.depth_engineered,
            "ir_engineered": self.ir_engineered,
            "imu_engineered": self.imu_engineered,
            "skeleton_engineered": self.skeleton_engineered,
        }
        return {name: values for name, values in arrays.items() if values is not None}


@dataclass(frozen=True)
class _IMUDeviceSeries:
    timestamps: np.ndarray
    values: np.ndarray


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


def _load_image_frame_data(
    directory: Path,
    config: FeatureConfig,
    *,
    include_color: bool,
) -> tuple[np.ndarray, np.ndarray | None] | None:
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
    color_frames: list[np.ndarray] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                grayscale = image.convert("L").resize(
                    (config.image_width, config.image_height),
                    Image.Resampling.BILINEAR,
                )
                grayscale_array = np.asarray(grayscale, dtype=np.float32) / 255.0
                if include_color:
                    hsv = image.convert("HSV").resize(
                        (config.image_width, config.image_height),
                        Image.Resampling.BILINEAR,
                    )
                    color_array = np.asarray(hsv, dtype=np.float32) / 255.0
                frames.append(grayscale_array)
                if include_color:
                    color_frames.append(color_array)
        except (OSError, ValueError):
            continue
    if not frames:
        return None
    return np.stack(frames), np.stack(color_frames) if include_color else None


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


def _grid_pool(image: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    rows, columns = grid
    height, width = image.shape
    if height % rows != 0 or width % columns != 0:
        raise ValueError(f"Image dimensions must be divisible into a {rows}x{columns} grid")
    return image.reshape(rows, height // rows, columns, width // columns).mean(axis=(1, 3))


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


def image_engineered_feature_size(config: FeatureConfig, *, include_color: bool) -> int:
    grid_size = config.image_engineered_grid[0] * config.image_engineered_grid[1]
    grayscale_size = config.temporal_bins * (3 * grid_size + 9)
    color_size = config.temporal_bins * 3 * grid_size + 3 * 12 if include_color else 0
    return grayscale_size + color_size


def _motion_center(motion: np.ndarray) -> tuple[float, float]:
    weights = np.asarray(motion, dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return 0.5, 0.5
    rows = np.linspace(0.0, 1.0, weights.shape[0])[:, None]
    columns = np.linspace(0.0, 1.0, weights.shape[1])[None, :]
    return float(np.sum(weights * columns) / total), float(np.sum(weights * rows) / total)


def _ensure_temporal_length(values: np.ndarray, minimum: int) -> np.ndarray:
    if len(values) >= minimum:
        return values
    indices = np.linspace(0, len(values) - 1, minimum).round().astype(int)
    return values[indices]


def _image_engineered_features(
    frames: np.ndarray,
    color_frames: np.ndarray | None,
    config: FeatureConfig,
) -> np.ndarray:
    grid = config.image_engineered_grid
    temporal_frames = _ensure_temporal_length(frames, config.temporal_bins)
    features: list[np.ndarray] = []
    for chunk in np.array_split(temporal_frames, config.temporal_bins):
        mean_image = np.mean(chunk, axis=0)
        signed_motion = chunk[-1] - chunk[0]
        absolute_motion = (
            np.mean(np.abs(np.diff(chunk, axis=0)), axis=0)
            if len(chunk) > 1
            else np.zeros_like(chunk[0])
        )
        center_x, center_y = _motion_center(absolute_motion)
        scalar = np.asarray(
            (
                np.mean(chunk),
                np.std(chunk),
                np.percentile(chunk, 25),
                np.median(chunk),
                np.percentile(chunk, 75),
                np.mean(absolute_motion),
                np.std(absolute_motion),
                center_x,
                center_y,
            ),
            dtype=np.float32,
        )
        features.extend(
            (
                _grid_pool(mean_image, grid).ravel().astype(np.float32),
                _grid_pool(absolute_motion, grid).ravel().astype(np.float32),
                _grid_pool(signed_motion, grid).ravel().astype(np.float32),
                scalar,
            )
        )

    if color_frames is not None:
        temporal_color = _ensure_temporal_length(color_frames, config.temporal_bins)
        for chunk in np.array_split(temporal_color, config.temporal_bins):
            mean_color = np.mean(chunk, axis=0)
            for channel in range(3):
                features.append(
                    _grid_pool(mean_color[:, :, channel], grid).ravel().astype(np.float32)
                )
        for channel in range(3):
            histogram, _ = np.histogram(
                color_frames[:, :, :, channel],
                bins=12,
                range=(0.0, 1.0),
            )
            total = max(1, int(np.sum(histogram)))
            features.append((histogram / total).astype(np.float32))
    return np.concatenate(features).astype(np.float32)


def extract_image_feature_pair(
    directory: Path,
    config: FeatureConfig,
    *,
    include_color: bool,
    include_base: bool = True,
) -> tuple[np.ndarray | None, np.ndarray]:
    loaded = _load_image_frame_data(directory, config, include_color=include_color)
    if loaded is None:
        return (
            (
                np.full(image_feature_size(config), np.nan, dtype=np.float32)
                if include_base
                else None
            ),
            np.full(
                image_engineered_feature_size(config, include_color=include_color),
                np.nan,
                dtype=np.float32,
            ),
        )
    frames, color_frames = loaded
    base: np.ndarray | None = None
    if include_base:
        summaries: list[np.ndarray] = []
        for summary in _image_summaries(frames):
            summaries.extend(
                (
                    _hog(summary, config),
                    _coarse_pool(summary).ravel().astype(np.float32),
                    np.percentile(summary, (0, 25, 50, 75, 100)).astype(np.float32),
                )
            )
        base = np.concatenate(summaries).astype(np.float32)
    return (
        base,
        _image_engineered_features(frames, color_frames, config),
    )


def extract_image_features(directory: Path, config: FeatureConfig) -> np.ndarray:
    base, _ = extract_image_feature_pair(directory, config, include_color=False)
    assert base is not None
    return base


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


def _read_imu(directory: Path) -> dict[str, _IMUDeviceSeries]:
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
                base_values = [_safe_float(value) for value in row[2:8]]
                if any(value is None for value in base_values):
                    continue
                vector = np.asarray(base_values, dtype=np.float32)
                acc_magnitude = float(np.linalg.norm(vector[:3]))
                gyro_magnitude = float(np.linalg.norm(vector[3:]))
                optional = [_safe_float(value) for value in row[8:18]]
                optional.extend([None] * (10 - len(optional)))
                enriched = np.concatenate(
                    (
                        vector,
                        np.asarray((acc_magnitude, gyro_magnitude), dtype=np.float32),
                        np.asarray(
                            [value if value is not None else np.nan for value in optional[:10]],
                            dtype=np.float32,
                        ),
                    )
                )
                rows_by_device[device].append((_parse_time(row[0], sequence), enriched))
                sequence += 1

    result: dict[str, _IMUDeviceSeries] = {}
    for device, rows in rows_by_device.items():
        rows.sort(key=lambda item: item[0])
        result[device] = _IMUDeviceSeries(
            timestamps=(
                np.asarray([timestamp for timestamp, _ in rows], dtype=np.float64)
                if rows
                else np.empty(0, dtype=np.float64)
            ),
            values=(
                np.stack([values for _, values in rows])
                if rows
                else np.empty((0, _IMU_EXTENDED_CHANNELS), dtype=np.float32)
            ),
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


def _interpolate_signal(values: np.ndarray) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(signal)
    if not valid.any():
        return np.empty(0, dtype=np.float64)
    if valid.all():
        return signal
    index = np.arange(len(signal), dtype=np.float64)
    return np.interp(index, index[valid], signal[valid])


def _resample_signal(values: np.ndarray, length: int) -> np.ndarray:
    signal = _interpolate_signal(values)
    if signal.size == 0:
        return signal
    if len(signal) == 1:
        return np.full(length, signal[0], dtype=np.float64)
    return np.interp(
        np.linspace(0.0, 1.0, length),
        np.linspace(0.0, 1.0, len(signal)),
        signal,
    )


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = float(
        np.sqrt(np.sum(first_centered**2) * np.sum(second_centered**2))
    )
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(first_centered * second_centered) / denominator)


def _dynamic_signal_features(values: np.ndarray, samples: int) -> np.ndarray:
    signal = _resample_signal(values, samples)
    if signal.size == 0:
        return np.full(_DYNAMIC_FEATURES, np.nan, dtype=np.float32)
    centered = signal - np.mean(signal)
    difference = np.diff(signal)
    second_difference = np.diff(signal, n=2)
    standard_deviation = float(np.std(centered))
    if standard_deviation > 1e-12:
        normalized = centered / standard_deviation
        skewness = float(np.mean(normalized**3))
        kurtosis = float(np.mean(normalized**4) - 3.0)
        autocorrelation = _safe_correlation(centered[:-1], centered[1:])
    else:
        skewness = 0.0
        kurtosis = 0.0
        autocorrelation = 0.0

    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    spectrum = spectrum[1:]
    total_power = float(np.sum(spectrum))
    if total_power > 1e-12:
        probabilities = spectrum / total_power
        dominant_frequency = float((np.argmax(spectrum) + 1) / max(1, len(spectrum)))
        spectral_entropy = float(
            -np.sum(probabilities * np.log(probabilities + 1e-12))
            / math.log(max(2, len(probabilities)))
        )
        high_frequency_fraction = float(np.sum(spectrum[len(spectrum) // 2 :]) / total_power)
    else:
        dominant_frequency = 0.0
        spectral_entropy = 0.0
        high_frequency_fraction = 0.0
    return np.asarray(
        (
            np.sqrt(np.mean(difference**2)) if difference.size else 0.0,
            np.sqrt(np.mean(second_difference**2)) if second_difference.size else 0.0,
            np.mean(np.signbit(centered[:-1]) != np.signbit(centered[1:]))
            if len(centered) > 1
            else 0.0,
            skewness,
            kurtosis,
            autocorrelation,
            dominant_frequency,
            spectral_entropy,
            high_frequency_fraction,
        ),
        dtype=np.float32,
    )


def _circular_features(values: np.ndarray, temporal_bins: int) -> np.ndarray:
    signal = _interpolate_signal(values)
    size = 5 + temporal_bins * 2
    if signal.size == 0:
        return np.full(size, np.nan, dtype=np.float32)
    radians = np.deg2rad(signal)
    sine = np.sin(radians)
    cosine = np.cos(radians)
    resultant = float(np.hypot(np.mean(sine), np.mean(cosine)))
    unwrapped = np.unwrap(radians)
    features = [
        float(np.mean(sine)),
        float(np.mean(cosine)),
        resultant,
        float(np.std(unwrapped) / np.pi),
        float(np.ptp(unwrapped) / (2.0 * np.pi)),
    ]
    temporal_sine = _ensure_temporal_length(sine, temporal_bins)
    temporal_cosine = _ensure_temporal_length(cosine, temporal_bins)
    for sine_chunk, cosine_chunk in zip(
        np.array_split(temporal_sine, temporal_bins),
        np.array_split(temporal_cosine, temporal_bins),
        strict=True,
    ):
        features.extend((float(np.mean(sine_chunk)), float(np.mean(cosine_chunk))))
    return np.asarray(features, dtype=np.float32)


def _quaternion_features(values: np.ndarray, config: FeatureConfig) -> np.ndarray:
    size = 5 * (10 + config.temporal_bins * 2)
    interpolated = _interpolate_columns(values)
    if interpolated is None:
        return np.full(size, np.nan, dtype=np.float32)
    norms = np.linalg.norm(interpolated, axis=1)
    valid = norms > 1e-8
    if not valid.any():
        return np.full(size, np.nan, dtype=np.float32)
    quaternions = interpolated / np.maximum(norms[:, None], 1e-8)
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0:
            quaternions[index] *= -1.0
    initial_conjugate = quaternions[0] * np.asarray((1.0, -1.0, -1.0, -1.0))
    w1, x1, y1, z1 = np.moveaxis(quaternions, 1, 0)
    w2, x2, y2, z2 = initial_conjugate
    relative = np.column_stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )
    orientation_step = np.zeros(len(quaternions), dtype=np.float64)
    if len(quaternions) > 1:
        dots = np.clip(np.abs(np.sum(quaternions[1:] * quaternions[:-1], axis=1)), 0.0, 1.0)
        orientation_step[1:] = 2.0 * np.arccos(dots)
    return np.concatenate(
        [
            *(
                _signal_features(relative[:, channel], config.temporal_bins)
                for channel in range(4)
            ),
            _signal_features(orientation_step, config.temporal_bins),
        ]
    ).astype(np.float32)


def _vector_shape_features(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.full(6, np.nan, dtype=np.float32)
    columns = np.column_stack([_interpolate_signal(values[:, index]) for index in range(3)])
    if columns.shape[0] < 2:
        return np.zeros(6, dtype=np.float32)
    covariance = np.cov(columns, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    pairwise = np.asarray(
        (
            _safe_correlation(columns[:, 0], columns[:, 1]),
            _safe_correlation(columns[:, 0], columns[:, 2]),
            _safe_correlation(columns[:, 1], columns[:, 2]),
        )
    )
    return np.concatenate((eigenvalues, pairwise)).astype(np.float32)


def _device_metadata(series: _IMUDeviceSeries) -> np.ndarray:
    if len(series.timestamps) == 0:
        return np.full(3, np.nan, dtype=np.float32)
    duration = max(0.0, float(series.timestamps[-1] - series.timestamps[0]))
    sample_rate = (len(series.timestamps) - 1) / duration if duration > 1e-9 else 0.0
    return np.asarray((np.log1p(len(series.timestamps)), duration, sample_rate), dtype=np.float32)


def _pairwise_device_features(
    left: _IMUDeviceSeries,
    right: _IMUDeviceSeries,
    config: FeatureConfig,
) -> np.ndarray:
    features: list[float] = []
    for channel in range(_IMU_BASE_CHANNELS):
        left_signal = _resample_signal(left.values[:, channel], config.spectral_samples)
        right_signal = _resample_signal(right.values[:, channel], config.spectral_samples)
        if left_signal.size == 0 or right_signal.size == 0:
            features.extend((np.nan, np.nan, np.nan))
            continue
        difference = left_signal - right_signal
        correlation = _safe_correlation(left_signal, right_signal)
        features.extend(
            (
                correlation,
                float(np.sqrt(np.mean(difference**2))),
                float(np.mean(np.abs(difference))),
            )
        )
    return np.asarray(features, dtype=np.float32)


def imu_feature_size(config: FeatureConfig) -> int:
    return len(EXPECTED_IMU_DEVICES) * 8 * (10 + config.temporal_bins * 2)


def imu_engineered_feature_size(config: FeatureConfig) -> int:
    signal_size = 10 + config.temporal_bins * 2
    per_device = (
        _IMU_BASE_CHANNELS * _DYNAMIC_FEATURES
        + 3 * (5 + config.temporal_bins * 2)
        + 5 * signal_size
        + 3
        + 12
    )
    device_pairs = 6
    return len(EXPECTED_IMU_DEVICES) * per_device + device_pairs * _IMU_BASE_CHANNELS * 3


def extract_imu_feature_pair(
    directory: Path,
    config: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    devices = _read_imu(directory)
    base = [
        _signal_features(devices[device].values[:, channel], config.temporal_bins)
        for device in EXPECTED_IMU_DEVICES
        for channel in range(_IMU_BASE_CHANNELS)
    ]
    engineered: list[np.ndarray] = []
    for device in EXPECTED_IMU_DEVICES:
        series = devices[device]
        engineered.extend(
            _dynamic_signal_features(series.values[:, channel], config.spectral_samples)
            for channel in range(_IMU_BASE_CHANNELS)
        )
        engineered.extend(
            _circular_features(series.values[:, channel], config.temporal_bins)
            for channel in range(8, 11)
        )
        engineered.append(_quaternion_features(series.values[:, 14:18], config))
        engineered.append(_device_metadata(series))
        engineered.append(_vector_shape_features(series.values[:, :3]))
        engineered.append(_vector_shape_features(series.values[:, 3:6]))

    pair_names = (
        ("WTLA", "WTRA"),
        ("WTLL", "WTRL"),
        ("WTLA", "WTC"),
        ("WTRA", "WTC"),
        ("WTLL", "WTC"),
        ("WTRL", "WTC"),
    )
    engineered.extend(
        _pairwise_device_features(devices[left], devices[right], config)
        for left, right in pair_names
    )
    return (
        np.concatenate(base).astype(np.float32),
        np.concatenate(engineered).astype(np.float32),
    )


def extract_imu_features(directory: Path, config: FeatureConfig) -> np.ndarray:
    return extract_imu_feature_pair(directory, config)[0]


def _valid_people(payload: object) -> list[tuple[np.ndarray, np.ndarray]]:
    if not isinstance(payload, list):
        return []
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for person in payload:
        if not isinstance(person, dict):
            continue
        points = np.asarray(person.get("keypoints", []), dtype=np.float32)
        if points.shape != (SKELETON_JOINTS, 3) or not np.isfinite(points).all():
            continue
        scores = np.asarray(
            person.get("keypoint_scores", [1.0] * SKELETON_JOINTS), dtype=np.float32
        )
        if scores.shape != (SKELETON_JOINTS,):
            scores = np.ones(SKELETON_JOINTS, dtype=np.float32)
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
    return config.skeleton_frames * (SKELETON_JOINTS * 3 * 2 + SKELETON_JOINTS)


def skeleton_engineered_feature_size(config: FeatureConfig) -> int:
    signal_size = 10 + config.temporal_bins * 2
    geometry_signals = 24
    joint_motion = SKELETON_JOINTS * 4
    selected_acceleration = 5 * 4
    confidence_and_count = 22
    periodic_signals = 7 * _DYNAMIC_FEATURES
    return (
        geometry_signals * signal_size
        + joint_motion
        + selected_acceleration
        + confidence_and_count
        + periodic_signals
    )


def _joint_angle_cosine(
    points: np.ndarray,
    first: int,
    center: int,
    last: int,
) -> np.ndarray:
    first_vector = points[:, first] - points[:, center]
    last_vector = points[:, last] - points[:, center]
    denominator = np.linalg.norm(first_vector, axis=1) * np.linalg.norm(last_vector, axis=1)
    cosine = np.sum(first_vector * last_vector, axis=1) / np.maximum(denominator, 1e-8)
    return np.clip(cosine, -1.0, 1.0)


def _skeleton_engineered_features(
    pose_array: np.ndarray,
    resampled_pose: np.ndarray,
    resampled_scores: np.ndarray,
    scale: float,
    config: FeatureConfig,
) -> np.ndarray:
    points = resampled_pose.reshape(config.skeleton_frames, SKELETON_JOINTS, 3)
    # Elbow, shoulder, hip and knee flexion, left and right.
    angle_triplets = (
        (L_SHOULDER, L_ELBOW, L_WRIST),
        (R_SHOULDER, R_ELBOW, R_WRIST),
        (L_ELBOW, L_SHOULDER, L_HIP),
        (R_ELBOW, R_SHOULDER, R_HIP),
        (L_SHOULDER, L_HIP, L_KNEE),
        (R_SHOULDER, R_HIP, R_KNEE),
        (L_HIP, L_KNEE, L_ANKLE),
        (R_HIP, R_KNEE, R_ANKLE),
    )
    angles = np.column_stack(
        [_joint_angle_cosine(points, *triplet) for triplet in angle_triplets]
    )
    # Hand-to-head separates the face-directed actions (Wash_face, Brush_teeth,
    # Drink_water); the rest carry stance width and arm extension.
    distance_pairs = (
        (L_WRIST, HEAD),
        (R_WRIST, HEAD),
        (L_WRIST, R_WRIST),
        (L_ANKLE, R_ANKLE),
        (L_WRIST, L_HIP),
        (R_WRIST, R_HIP),
        (L_SHOULDER, R_SHOULDER),
    )
    distances = np.column_stack(
        [
            np.linalg.norm(points[:, first] - points[:, second], axis=1)
            for first, second in distance_pairs
        ]
    )
    roots = points[:, PELVIS]
    shoulder_centers = (points[:, L_SHOULDER] + points[:, R_SHOULDER]) / 2.0
    torso = shoulder_centers - roots
    torso_unit = torso / np.maximum(np.linalg.norm(torso, axis=1)[:, None], 1e-8)
    extents = np.max(points, axis=1) - np.min(points, axis=1)

    original_roots = pose_array[:, PELVIS]
    resampled_roots = _resample(original_roots, config.skeleton_frames)
    root_displacement = (resampled_roots - resampled_roots[:1]) / max(scale, 1e-6)
    geometry = np.concatenate(
        (angles, distances, torso_unit, extents, root_displacement),
        axis=1,
    )
    features: list[np.ndarray] = [
        _signal_features(geometry[:, channel], config.temporal_bins)
        for channel in range(geometry.shape[1])
    ]

    velocity = np.diff(points, axis=0, prepend=points[:1])
    speed = np.linalg.norm(velocity, axis=2)
    motion_summary = np.column_stack(
        (
            np.mean(speed, axis=0),
            np.std(speed, axis=0),
            np.max(speed, axis=0),
            np.sum(speed, axis=0),
        )
    ).ravel()
    features.append(motion_summary.astype(np.float32))

    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
    acceleration_norm = np.linalg.norm(acceleration, axis=2)
    selected_joints = (HEAD, L_WRIST, R_WRIST, L_ANKLE, R_ANKLE)
    acceleration_summary = np.column_stack(
        (
            np.mean(acceleration_norm[:, selected_joints], axis=0),
            np.std(acceleration_norm[:, selected_joints], axis=0),
            np.max(acceleration_norm[:, selected_joints], axis=0),
            np.sqrt(np.mean(acceleration_norm[:, selected_joints] ** 2, axis=0)),
        )
    ).ravel()
    features.append(acceleration_summary.astype(np.float32))

    confidence = np.asarray(
        (
            *np.mean(resampled_scores, axis=0),
            np.mean(resampled_scores),
            np.std(resampled_scores),
            np.min(resampled_scores),
            np.mean(resampled_scores < 0.5),
            np.log1p(len(pose_array)),
        ),
        dtype=np.float32,
    )
    features.append(confidence)

    periodic = [
        root_displacement[:, 0],
        root_displacement[:, 1],
        root_displacement[:, 2],
        speed[:, L_WRIST],
        speed[:, R_WRIST],
        speed[:, L_ANKLE],
        speed[:, R_ANKLE],
    ]
    features.extend(
        _dynamic_signal_features(signal, config.spectral_samples) for signal in periodic
    )
    return np.concatenate(features).astype(np.float32)


def extract_skeleton_feature_pair(
    directory: Path,
    config: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted(directory.rglob("*.json"), key=_path_sort_key)
    if not paths:
        return (
            np.full(skeleton_feature_size(config), np.nan, dtype=np.float32),
            np.full(skeleton_engineered_feature_size(config), np.nan, dtype=np.float32),
        )

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
            poses.append(np.full((SKELETON_JOINTS, 3), np.nan, dtype=np.float32))
            scores.append(np.full(SKELETON_JOINTS, np.nan, dtype=np.float32))
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
        return (
            np.full(skeleton_feature_size(config), np.nan, dtype=np.float32),
            np.full(skeleton_engineered_feature_size(config), np.nan, dtype=np.float32),
        )

    pose_array = flat_pose.reshape(-1, SKELETON_JOINTS, 3)
    # The pelvis is the pose root, so centring on it removes camera placement
    # without the wobble that a limb midpoint would introduce.
    roots = pose_array[:, PELVIS]
    shoulder_centers = (pose_array[:, L_SHOULDER] + pose_array[:, R_SHOULDER]) / 2.0
    torso_lengths = np.linalg.norm(shoulder_centers - roots, axis=1)
    valid_scale = torso_lengths[np.isfinite(torso_lengths) & (torso_lengths > 1e-6)]
    scale = float(np.median(valid_scale)) if valid_scale.size else 1.0
    normalized = (pose_array - roots[:, None, :]) / max(scale, 1e-6)

    resampled_pose = _resample(normalized.reshape(len(normalized), -1), config.skeleton_frames)
    resampled_scores = _resample(flat_scores, config.skeleton_frames)
    velocity = np.diff(resampled_pose, axis=0, prepend=resampled_pose[:1])
    base = np.concatenate(
        (resampled_pose.ravel(), velocity.ravel(), resampled_scores.ravel())
    ).astype(np.float32)
    return (
        base,
        _skeleton_engineered_features(
            pose_array,
            resampled_pose,
            resampled_scores,
            scale,
            config,
        ),
    )


def extract_skeleton_features(directory: Path, config: FeatureConfig) -> np.ndarray:
    return extract_skeleton_feature_pair(directory, config)[0]


def extract_clip_features(
    record: ClipRecord,
    config: FeatureConfig,
    selected_blocks: Sequence[str] | None = None,
) -> tuple[np.ndarray | None, ...]:
    """Extract one clip, returning ``None`` for every unselected block.

    Blocks come in base/engineered pairs that share one read of the source
    files, so a pair is extracted whenever either half is wanted and the unused
    half is discarded.
    """

    needed = set(normalize_feature_blocks(selected_blocks))
    values: dict[str, np.ndarray | None] = dict.fromkeys(ALL_FEATURE_BLOCKS)

    def wanted(*names: str) -> bool:
        return any(name in needed for name in names)

    def keep(name: str, value: np.ndarray | None) -> None:
        values[name] = value if name in needed else None

    if wanted("depth", "depth_engineered"):
        base, engineered = extract_image_feature_pair(
            record.depth_dir, config, include_color=True
        )
        keep("depth", base)
        keep("depth_engineered", engineered)
    if wanted("ir", "ir_engineered"):
        base, engineered = extract_image_feature_pair(
            record.ir_dir,
            config,
            include_color=False,
            # The legacy flag only decides whether the base is produced by
            # default; asking for the block explicitly always produces it.
            include_base=config.include_legacy_ir or "ir" in needed,
        )
        keep("ir", base)
        keep("ir_engineered", engineered)
    if wanted("imu", "imu_engineered"):
        base, engineered = extract_imu_feature_pair(record.imu_dir, config)
        keep("imu", base)
        keep("imu_engineered", engineered)
    if wanted("skeleton", "skeleton_engineered"):
        base, engineered = extract_skeleton_feature_pair(record.skeleton_dir, config)
        keep("skeleton", base)
        keep("skeleton_engineered", engineered)

    return tuple(values[name] for name in ALL_FEATURE_BLOCKS)


def _extract_job(
    args: tuple[ClipRecord, FeatureConfig, tuple[str, ...]],
) -> tuple[np.ndarray | None, ...]:
    record, config, selected = args
    return extract_clip_features(record, config, selected)


def _progressive_results(
    records: Sequence[ClipRecord],
    config: FeatureConfig,
    n_jobs: int,
    selected_blocks: tuple[str, ...],
) -> Iterator[tuple[np.ndarray | None, ...]]:
    jobs = ((record, config, selected_blocks) for record in records)
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
    progress_callback: Callable[[int, int], None] | None = None,
    selected_blocks: Sequence[str] | None = None,
) -> RawFeatureBundle:
    """Extract raw modality matrices while preserving record order.

    ``selected_blocks`` optionally restricts extraction to a subset of
    :data:`ALL_FEATURE_BLOCKS`.  When ``None`` the default set
    :data:`DEFAULT_FEATURE_BLOCKS` is used for backward compatibility — the
    legacy IR base block remains excluded unless explicitly requested.

    When ``progress_callback`` is provided it is called as
    ``progress_callback(done, total)`` each time ``progress_every`` clips
    have been processed (and once at the end). Exceptions raised by the
    callback propagate rather than being swallowed, so a wrong-arity callback
    fails loudly instead of silently never updating.
    """

    if not records:
        raise ValueError("Cannot extract features from an empty record list")
    active_config = config or FeatureConfig()
    selected = normalize_feature_blocks(selected_blocks)
    columns: dict[str, list[np.ndarray]] = {name: [] for name in selected}

    for index, values in enumerate(
        _progressive_results(records, active_config, n_jobs, selected), start=1
    ):
        for name, value in zip(ALL_FEATURE_BLOCKS, values, strict=True):
            if name not in columns:
                continue
            if value is None:
                # Every selected block must yield a row for every clip. A gap
                # would shorten one column and silently pair the remaining
                # clips' features with the wrong labels.
                raise ValueError(
                    f"Block {name!r} produced no features for clip "
                    f"{records[index - 1].clip_id!r}"
                )
            columns[name].append(value)
        if progress_every > 0 and (index % progress_every == 0 or index == len(records)):
            print(f"Extracted {index:,}/{len(records):,} clips", flush=True)
        if progress_callback is not None and (index % progress_every == 0 or index == len(records)):
            progress_callback(index, len(records))

    is_train = records[0].split == "train"
    return RawFeatureBundle(
        clip_ids=np.asarray([record.clip_id for record in records]),
        **{
            name: (np.stack(columns[name]) if name in columns else None)
            for name in ALL_FEATURE_BLOCKS
        },
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
