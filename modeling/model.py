"""Leakage-safe multimodal preprocessing and algorithm evaluation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modeling.algorithms.base import TrainingAlgorithm
from modeling.features import FeatureConfig, RawFeatureBundle, feature_config_dict


PCA_COMPONENTS = {
    "depth": 64,
    "ir": 64,
    "imu": 128,
    "skeleton": 128,
}


def library_versions() -> dict[str, str]:
    """Return the library versions a model artifact depends on for unpickling."""

    return {"scikit-learn": sklearn.__version__, "numpy": np.__version__}


@dataclass
class FittedMultimodalModel:
    reducers: dict[str, Pipeline]
    combined_scaler: StandardScaler
    classifier: Any
    feature_config: dict[str, object]
    pca_components: dict[str, int]
    algorithm_name: str = "logistic_regression"
    algorithm_parameters: dict[str, object] = field(default_factory=dict)
    library_versions: dict[str, str] = field(default_factory=dict)

    def transform(self, bundle: RawFeatureBundle) -> np.ndarray:
        arrays = bundle.modality_arrays()
        missing = set(self.reducers) - set(arrays)
        if missing:
            raise ValueError(f"Feature bundle is missing model blocks: {sorted(missing)}")
        blocks = [
            reducer.transform(arrays[name])
            for name, reducer in self.reducers.items()
        ]
        return self.combined_scaler.transform(np.concatenate(blocks, axis=1))

    def predict(self, bundle: RawFeatureBundle) -> np.ndarray:
        return self.classifier.predict(self.transform(bundle))

    def predict_proba(self, bundle: RawFeatureBundle) -> np.ndarray:
        if not hasattr(self.classifier, "predict_proba"):
            raise AttributeError(f"Algorithm {self.algorithm_name} does not provide predict_proba()")
        return self.classifier.predict_proba(self.transform(bundle))


def _make_reducer(name: str, random_state: int) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ]
    components = PCA_COMPONENTS.get(name)
    if components is not None:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=components,
                    svd_solver="randomized",
                    random_state=random_state,
                ),
            )
        )
    return Pipeline(steps)


def _fit_reducers(
    arrays: Mapping[str, np.ndarray],
    train_indices: np.ndarray,
    random_state: int,
) -> dict[str, Pipeline]:
    reducers: dict[str, Pipeline] = {}
    for name in arrays:
        reducer = _make_reducer(name, random_state)
        reducer.fit(arrays[name][train_indices])
        reducers[name] = reducer
    return reducers


def _transform(
    reducers: Mapping[str, Pipeline],
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [reducer.transform(arrays[name][indices]) for name, reducer in reducers.items()],
        axis=1,
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    observed_classes = np.unique(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(
            recall_score(
                y_true,
                y_pred,
                labels=observed_classes,
                average="macro",
                zero_division=0,
            )
        ),
    }


def cross_validate(
    bundle: RawFeatureBundle,
    algorithm: TrainingAlgorithm,
    parameter_candidates: Sequence[Mapping[str, Any]],
    *,
    n_splits: int = 5,
    random_state: int = 42,
    selection_metric: str = "macro_f1",
) -> tuple[dict[str, Any], dict[str, object]]:
    """Evaluate estimator parameters with preprocessing fitted per fold."""

    if bundle.labels is None or bundle.groups is None:
        raise ValueError("Cross-validation requires labels and participant groups")
    if selection_metric not in {"accuracy", "macro_f1", "balanced_accuracy"}:
        raise ValueError(f"Unsupported selection metric: {selection_metric}")

    arrays = bundle.modality_arrays()
    labels = bundle.labels
    groups = bundle.groups
    candidates = [algorithm.resolved_parameters(parameters) for parameters in parameter_candidates]
    if not candidates:
        raise ValueError("Cross-validation requires at least one parameter candidate")
    predictions = [np.full(labels.shape, -1, dtype=np.int64) for _ in candidates]
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_summaries: list[dict[str, object]] = []
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels, groups), start=1
    ):
        reducers = _fit_reducers(arrays, train_indices, random_state + fold)
        train_reduced = _transform(reducers, arrays, train_indices)
        validation_reduced = _transform(reducers, arrays, validation_indices)
        scaler = StandardScaler().fit(train_reduced)
        train_scaled = scaler.transform(train_reduced)
        validation_scaled = scaler.transform(validation_reduced)

        fold_metrics: dict[str, object] = {
            "fold": fold,
            "training_clips": int(len(train_indices)),
            "validation_clips": int(len(validation_indices)),
            "validation_users": sorted({str(groups[index]) for index in validation_indices}),
            "candidates": [],
        }
        for candidate_index, parameters in enumerate(candidates):
            classifier = algorithm.build_estimator(parameters, random_state + fold)
            classifier.fit(train_scaled, labels[train_indices])
            predicted = classifier.predict(validation_scaled)
            predictions[candidate_index][validation_indices] = predicted
            fold_metrics["candidates"].append(
                {
                    "parameters": parameters,
                    "metrics": _metrics(labels[validation_indices], predicted),
                }
            )
        fold_summaries.append(fold_metrics)
        print(
            f"Completed grouped fold {fold}/{n_splits} "
            f"({len(validation_indices):,} validation clips)",
            flush=True,
        )

    aggregate = [
        {"parameters": parameters, "metrics": _metrics(labels, predicted)}
        for parameters, predicted in zip(candidates, predictions, strict=True)
    ]
    best_index = max(
        range(len(candidates)),
        key=lambda index: aggregate[index]["metrics"][selection_metric],
    )
    selected_parameters = candidates[best_index]
    classes = np.arange(int(np.max(labels)) + 1)
    report: dict[str, object] = {
        "algorithm": algorithm.name,
        "selection_metric": selection_metric,
        "selected_parameters": selected_parameters,
        "aggregate_metrics": aggregate,
        "folds": fold_summaries,
        "confusion_matrix": confusion_matrix(
            labels,
            predictions[best_index],
            labels=classes,
        ).tolist(),
    }
    return selected_parameters, report


def fit_final_model(
    bundle: RawFeatureBundle,
    algorithm: TrainingAlgorithm,
    *,
    parameters: Mapping[str, Any],
    feature_config: FeatureConfig,
    random_state: int = 42,
) -> FittedMultimodalModel:
    if bundle.labels is None:
        raise ValueError("Final training requires labels")
    arrays = bundle.modality_arrays()
    indices = np.arange(len(bundle.labels))
    reducers = _fit_reducers(arrays, indices, random_state)
    reduced = _transform(reducers, arrays, indices)
    scaler = StandardScaler().fit(reduced)
    resolved_parameters = algorithm.resolved_parameters(parameters)
    classifier = algorithm.build_estimator(resolved_parameters, random_state)
    classifier.fit(scaler.transform(reduced), bundle.labels)
    return FittedMultimodalModel(
        reducers=reducers,
        combined_scaler=scaler,
        classifier=classifier,
        feature_config=feature_config_dict(feature_config),
        pca_components=dict(PCA_COMPONENTS),
        algorithm_name=algorithm.name,
        algorithm_parameters=resolved_parameters,
        library_versions=library_versions(),
    )


def save_model(model: FittedMultimodalModel, path: str | Path) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    return output


def load_model(path: str | Path) -> FittedMultimodalModel:
    model = joblib.load(Path(path).expanduser())
    if not isinstance(model, FittedMultimodalModel):
        raise TypeError("Model artifact does not contain a FittedMultimodalModel")
    if not hasattr(model, "algorithm_name"):
        model.algorithm_name = "logistic_regression"
    if not hasattr(model, "algorithm_parameters"):
        model.algorithm_parameters = {}
    if not hasattr(model, "library_versions"):
        model.library_versions = {}
    _warn_on_version_mismatch(model, path)
    return model


def _warn_on_version_mismatch(model: FittedMultimodalModel, path: str | Path) -> None:
    """Warn when the artifact was written by different library versions.

    Estimator pickles are not guaranteed portable across scikit-learn versions,
    so a mismatch means predictions could silently differ from the run that
    produced the artifact. Retrain to remove the warning.
    """

    if not model.library_versions:
        warnings.warn(
            f"{path} records no library versions, so it cannot be checked against "
            f"the running scikit-learn {sklearn.__version__}. Retrain to record them.",
            RuntimeWarning,
            stacklevel=3,
        )
        return
    current = library_versions()
    mismatched = {
        name: (saved, current[name])
        for name, saved in model.library_versions.items()
        if name in current and saved != current[name]
    }
    if mismatched:
        details = "; ".join(
            f"{name} {saved} -> {running}" for name, (saved, running) in sorted(mismatched.items())
        )
        warnings.warn(
            f"{path} was written with different library versions ({details}). "
            "Predictions may differ from the original run; retrain to be certain.",
            RuntimeWarning,
            stacklevel=3,
        )
