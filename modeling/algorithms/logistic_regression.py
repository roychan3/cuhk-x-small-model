"""Multinomial logistic-regression algorithm definition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sklearn.linear_model import LogisticRegression

from modeling.algorithms.base import TrainingAlgorithm, register_algorithm


class LogisticRegressionAlgorithm(TrainingAlgorithm):
    name = "logistic_regression"
    display_name = "Multinomial logistic regression"
    artifact_name = "logreg"
    default_parameters = {"C": 1.0}
    default_search_space = {"C": (0.01, 0.1, 1.0, 10.0)}

    def build_estimator(
        self,
        parameters: Mapping[str, Any],
        random_state: int,
    ) -> LogisticRegression:
        resolved = self.resolved_parameters(parameters)
        unknown = set(resolved) - {"C"}
        if unknown:
            raise ValueError(f"Unsupported logistic-regression parameters: {sorted(unknown)}")
        return LogisticRegression(
            C=float(resolved["C"]),
            solver="lbfgs",
            class_weight="balanced",
            max_iter=5_000,
            random_state=random_state,
        )


ALGORITHM = register_algorithm(LogisticRegressionAlgorithm())
