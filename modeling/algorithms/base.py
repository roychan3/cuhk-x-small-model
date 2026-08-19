"""Interface and registry for comparable training algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from sklearn.model_selection import ParameterGrid


ParameterSet = dict[str, Any]


class TrainingAlgorithm(ABC):
    """Build estimators and define their default comparison search space."""

    name: str
    display_name: str
    artifact_name: str
    default_parameters: Mapping[str, Any]
    default_search_space: Mapping[str, Sequence[Any]]

    @abstractmethod
    def build_estimator(self, parameters: Mapping[str, Any], random_state: int) -> Any:
        """Return an unfitted estimator implementing fit() and predict()."""

    def parameter_candidates(
        self,
        search_space: Mapping[str, Sequence[Any]] | None = None,
    ) -> list[ParameterSet]:
        active_space = self.default_search_space if search_space is None else search_space
        candidates = [dict(parameters) for parameters in ParameterGrid(dict(active_space))]
        if not candidates:
            raise ValueError(f"Algorithm {self.name} produced no parameter candidates")
        return candidates

    def resolved_parameters(self, parameters: Mapping[str, Any] | None = None) -> ParameterSet:
        resolved = dict(self.default_parameters)
        if parameters is not None:
            resolved.update(parameters)
        return resolved


_ALGORITHMS: dict[str, TrainingAlgorithm] = {}


def register_algorithm(algorithm: TrainingAlgorithm) -> TrainingAlgorithm:
    if not algorithm.name:
        raise ValueError("An algorithm must define a nonempty name")
    if algorithm.name in _ALGORITHMS:
        raise ValueError(f"Algorithm already registered: {algorithm.name}")
    _ALGORITHMS[algorithm.name] = algorithm
    return algorithm


def get_algorithm(name: str) -> TrainingAlgorithm:
    try:
        return _ALGORITHMS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_ALGORITHMS))
        raise ValueError(f"Unknown algorithm {name!r}. Available: {choices}") from exc


def available_algorithms() -> tuple[str, ...]:
    return tuple(sorted(_ALGORITHMS))
