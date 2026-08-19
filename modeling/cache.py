"""On-disk cache for expensive raw multimodal features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from modeling.features import FeatureConfig, RawFeatureBundle, feature_config_dict


CACHE_VERSION = 1


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
    np.savez(
        output,
        cache_version=np.asarray(CACHE_VERSION),
        feature_config=np.asarray(json.dumps(feature_config_dict(config), sort_keys=True)),
        train_clip_ids=train.clip_ids,
        train_depth=train.depth,
        train_ir=train.ir,
        train_imu=train.imu,
        train_skeleton=train.skeleton,
        train_labels=train.labels,
        train_groups=train.groups,
        test_clip_ids=test.clip_ids,
        test_depth=test.depth,
        test_ir=test.ir,
        test_imu=test.imu,
        test_skeleton=test.skeleton,
        test_submission_paths=test.submission_paths,
    )
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
        train = RawFeatureBundle(
            clip_ids=data["train_clip_ids"].copy(),
            depth=data["train_depth"].copy(),
            ir=data["train_ir"].copy(),
            imu=data["train_imu"].copy(),
            skeleton=data["train_skeleton"].copy(),
            labels=data["train_labels"].copy(),
            groups=data["train_groups"].copy(),
        )
        test = RawFeatureBundle(
            clip_ids=data["test_clip_ids"].copy(),
            depth=data["test_depth"].copy(),
            ir=data["test_ir"].copy(),
            imu=data["test_imu"].copy(),
            skeleton=data["test_skeleton"].copy(),
            submission_paths=data["test_submission_paths"].copy(),
        )
    return train, test
