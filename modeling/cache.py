"""On-disk cache for expensive raw multimodal features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from modeling.features import FeatureConfig, RawFeatureBundle, feature_config_dict


CACHE_VERSION = 3


def save_feature_cache(
    path: str | Path,
    train: RawFeatureBundle,
    test: RawFeatureBundle,
    config: FeatureConfig,
) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if train.labels is None or train.groups is None:
        raise ValueError("Training bundle is missing labels or groups")
    if test.submission_paths is None:
        raise ValueError("Test bundle is missing submission paths")
    train_arrays = train.modality_arrays()
    test_arrays = test.modality_arrays()
    if tuple(train_arrays) != tuple(test_arrays):
        raise ValueError("Training and test bundles have different feature blocks")
    payload: dict[str, np.ndarray] = {
        "cache_version": np.asarray(CACHE_VERSION),
        "feature_config": np.asarray(json.dumps(feature_config_dict(config), sort_keys=True)),
        "feature_blocks": np.asarray(json.dumps(list(train_arrays))),
        "train_clip_ids": train.clip_ids,
        "train_labels": train.labels,
        "train_groups": train.groups,
        "test_clip_ids": test.clip_ids,
        "test_submission_paths": test.submission_paths,
    }
    for name in train_arrays:
        payload[f"train_{name}"] = train_arrays[name]
        payload[f"test_{name}"] = test_arrays[name]
    np.savez(output, **payload)
    return output


def load_feature_cache(
    path: str | Path,
    config: FeatureConfig,
) -> tuple[RawFeatureBundle, RawFeatureBundle]:
    source = Path(path).expanduser()
    expected_config = json.dumps(feature_config_dict(config), sort_keys=True)
    with np.load(source, allow_pickle=False) as data:
        version = int(data["cache_version"])
        cached_config = str(data["feature_config"])
        if version != CACHE_VERSION:
            raise ValueError(f"Feature cache version {version} is not supported")
        if cached_config != expected_config:
            raise ValueError("Feature cache configuration does not match the active configuration")
        feature_blocks = json.loads(str(data["feature_blocks"]))
        if not isinstance(feature_blocks, list) or not all(
            isinstance(name, str) for name in feature_blocks
        ):
            raise ValueError("Feature cache has an invalid block list")
        required = {"depth", "imu", "skeleton"}
        supported = required | {
            "ir",
            "depth_engineered",
            "ir_engineered",
            "imu_engineered",
            "skeleton_engineered",
        }
        if not required.issubset(feature_blocks):
            raise ValueError("Feature cache is missing a required base block")
        if len(feature_blocks) != len(set(feature_blocks)) or not set(feature_blocks) <= supported:
            raise ValueError("Feature cache contains unsupported or duplicate blocks")
        train_arrays = {name: data[f"train_{name}"].copy() for name in feature_blocks}
        test_arrays = {name: data[f"test_{name}"].copy() for name in feature_blocks}
        train = RawFeatureBundle(
            clip_ids=data["train_clip_ids"].copy(),
            ir=train_arrays.pop("ir", None),
            labels=data["train_labels"].copy(),
            groups=data["train_groups"].copy(),
            **train_arrays,
        )
        test = RawFeatureBundle(
            clip_ids=data["test_clip_ids"].copy(),
            ir=test_arrays.pop("ir", None),
            submission_paths=data["test_submission_paths"].copy(),
            **test_arrays,
        )
    return train, test
