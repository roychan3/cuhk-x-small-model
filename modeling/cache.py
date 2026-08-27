"""On-disk cache for expensive raw multimodal features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from visualization.feature_blocks import normalize_feature_blocks
from modeling.features import (
    ALL_FEATURE_BLOCKS,
    FeatureConfig,
    RawFeatureBundle,
    feature_config_dict,
)


# 4: skeleton features were reindexed from COCO to the Human3.6M joint order
# the dataset actually uses, so every cached skeleton block from 3 is stale.
CACHE_VERSION = 4



def cached_feature_blocks(path: str | Path) -> tuple[str, ...]:
    """Return the blocks a cache holds, or empty when it cannot be read.

    Callers use this to decide whether a cache can serve a run without paying
    to load its matrices, so an unreadable or foreign file is reported as
    "nothing" rather than raising.
    """

    try:
        with np.load(Path(path).expanduser(), allow_pickle=False) as data:
            blocks = json.loads(str(data["feature_blocks"]))
    except (KeyError, OSError, ValueError):
        return ()
    if not isinstance(blocks, list):
        return ()
    try:
        return normalize_feature_blocks([str(name) for name in blocks])
    except ValueError:
        return ()


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
        supported = set(ALL_FEATURE_BLOCKS)
        if not feature_blocks:
            raise ValueError("Feature cache is empty")
        if len(feature_blocks) != len(set(feature_blocks)) or not set(feature_blocks) <= supported:
            raise ValueError("Feature cache contains unsupported or duplicate blocks")
        # Version 4 caches written before block selection always carried
        # depth/imu/skeleton; newer ones may hold any non-empty subset. The
        # payload layout is unchanged, so the version still describes the
        # format and older files stay readable. Which blocks a cache actually
        # holds is recorded in "feature_blocks" and checked by the caller.
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
