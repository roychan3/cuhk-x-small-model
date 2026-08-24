"""Registered training algorithms."""

from modeling.algorithms.base import (
    TrainingAlgorithm,
    available_algorithms,
    get_algorithm,
    register_algorithm,
)
from modeling.algorithms.late_fusion import ALGORITHM as LATE_FUSION
from modeling.algorithms.logistic_regression import ALGORITHM as LOGISTIC_REGRESSION

__all__ = [
    "LATE_FUSION",
    "LOGISTIC_REGRESSION",
    "TrainingAlgorithm",
    "available_algorithms",
    "get_algorithm",
    "register_algorithm",
]
