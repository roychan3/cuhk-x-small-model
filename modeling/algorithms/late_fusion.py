"""Late fusion of classifiers trained on the existing modality blocks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from modeling.algorithms.base import TrainingAlgorithm, register_algorithm


WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "equal": {"depth": 1.0, "ir": 1.0, "imu": 1.0, "skeleton": 1.0},
    "motion": {"depth": 0.75, "ir": 0.5, "imu": 1.25, "skeleton": 1.25},
    "wearables": {"depth": 0.75, "ir": 0.5, "imu": 1.75, "skeleton": 1.0},
    "stacked": {
        "combined": 1.25,
        "depth": 0.75,
        "ir": 0.5,
        "imu": 1.25,
        "skeleton": 1.25,
    },
}


class LateFusionClassifier:
    """Blend per-modality and optional combined-model probabilities."""

    def __init__(
        self,
        feature_groups: Mapping[str, np.ndarray],
        *,
        C: float,
        combined_C: float,
        weighting: str,
        random_state: int,
    ) -> None:
        self.feature_groups = {
            name: np.asarray(indices, dtype=np.int64)
            for name, indices in feature_groups.items()
        }
        self.C = C
        self.combined_C = combined_C
        self.weighting = weighting
        self.random_state = random_state
        self.estimators: dict[str, LogisticRegression] = {}
        self.estimator_columns: dict[str, np.ndarray] = {}
        self.classes_: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LateFusionClassifier":
        if self.weighting not in WEIGHT_PROFILES:
            raise ValueError(f"Unknown late-fusion weighting profile: {self.weighting}")
        if not self.feature_groups:
            raise ValueError("Late fusion requires at least one named feature group")
        profile = WEIGHT_PROFILES[self.weighting]
        # Weighting is the whole point of this algorithm, so an unrecognised
        # group must fail loudly rather than silently blend at weight 1.0 --
        # and it must fail before spending a fold's worth of fitting.
        unweighted = sorted(
            name for name, indices in self.feature_groups.items()
            if indices.size and name not in profile
        )
        if unweighted:
            raise ValueError(
                f"Weighting profile {self.weighting!r} has no weight for: {unweighted}"
            )
        # Refitting must not accumulate state from an earlier call, and the
        # groups handed in by the caller stay untouched: the combined model is
        # recorded in estimator_columns only.
        self.estimators = {}
        self.estimator_columns = {}
        for offset, (name, indices) in enumerate(self.feature_groups.items()):
            if indices.size == 0:
                continue
            estimator = LogisticRegression(
                C=self.C,
                solver="lbfgs",
                class_weight="balanced",
                max_iter=5_000,
                random_state=self.random_state + offset,
            )
            estimator.fit(features[:, indices], labels)
            self.estimators[name] = estimator
            self.estimator_columns[name] = indices
        if profile.get("combined", 0.0) > 0:
            combined = LogisticRegression(
                C=self.combined_C,
                solver="lbfgs",
                class_weight="balanced",
                max_iter=5_000,
                random_state=self.random_state + len(self.estimators),
            )
            combined.fit(features, labels)
            self.estimator_columns["combined"] = np.arange(features.shape[1], dtype=np.int64)
            self.estimators["combined"] = combined
        if not self.estimators:
            raise ValueError("Late fusion received only empty feature groups")
        self.classes_ = np.asarray(next(iter(self.estimators.values())).classes_)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("LateFusionClassifier has not been fitted")
        profile = WEIGHT_PROFILES[self.weighting]
        probabilities = np.zeros((len(features), len(self.classes_)), dtype=np.float64)
        total_weight = 0.0
        for name, estimator in self.estimators.items():
            weight = float(profile[name])
            probabilities += weight * estimator.predict_proba(
                features[:, self.estimator_columns[name]]
            )
            total_weight += weight
        return probabilities / total_weight

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("LateFusionClassifier has not been fitted")
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]


class LateFusionAlgorithm(TrainingAlgorithm):
    name = "late_fusion"
    display_name = "Modality-level late fusion"
    artifact_name = "late_fusion"
    default_parameters = {"C": 0.03, "combined_C": 0.1, "weighting": "stacked"}
    default_search_space = {
        "C": (0.03, 0.1),
        "combined_C": (0.1,),
        "weighting": ("motion", "stacked"),
    }

    def _resolved(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        resolved = self.resolved_parameters(parameters)
        unknown = set(resolved) - {"C", "combined_C", "weighting"}
        if unknown:
            raise ValueError(f"Unsupported late-fusion parameters: {sorted(unknown)}")
        weighting = str(resolved["weighting"])
        if weighting not in WEIGHT_PROFILES:
            raise ValueError(f"Unknown late-fusion weighting profile: {weighting}")
        return {
            "C": float(resolved["C"]),
            "combined_C": float(resolved["combined_C"]),
            "weighting": weighting,
        }

    def build_estimator(
        self,
        parameters: Mapping[str, Any],
        random_state: int,
    ) -> LateFusionClassifier:
        # The base contract is "return something fit()-able", and late fusion
        # cannot honour it: the per-modality column indices are only known once
        # the fold's reducers have run. Refuse here instead of handing back an
        # estimator whose fit() fails with an unrelated message.
        self._resolved(parameters)
        raise NotImplementedError(
            "Late fusion needs the per-modality column groups that only "
            "fit_estimator() receives; call fit_estimator() instead."
        )

    def fit_estimator(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        parameters: Mapping[str, Any],
        random_state: int,
        feature_groups: Mapping[str, np.ndarray] | None = None,
    ) -> LateFusionClassifier:
        if feature_groups is None:
            raise ValueError("Late fusion requires named feature groups")
        resolved = self._resolved(parameters)
        estimator = LateFusionClassifier(
            feature_groups,
            C=resolved["C"],
            combined_C=resolved["combined_C"],
            weighting=resolved["weighting"],
            random_state=random_state,
        )
        return estimator.fit(features, labels)


ALGORITHM = register_algorithm(LateFusionAlgorithm())
