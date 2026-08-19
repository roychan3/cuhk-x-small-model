"""Registered training algorithms."""

from modeling.algorithms.base import (
    TrainingAlgorithm,
    available_algorithms,
    get_algorithm,
    register_algorithm,
)
from modeling.algorithms.logistic_regression import ALGORITHM as LOGISTIC_REGRESSION

__all__ = [
    "LOGISTIC_REGRESSION",
    "TrainingAlgorithm",
    "available_algorithms",
    "get_algorithm",
    "register_algorithm",
]
