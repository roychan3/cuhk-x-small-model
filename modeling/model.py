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
    precision_recall_fscore_support,
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


@dataclass(frozen=True)
class ValidationOutputs:
    """Machine-readable out-of-fold results for analysis and ensembling."""

    clip_ids: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray | None
    classes: np.ndarray
    fold_assignments: np.ndarray
    repeat_predictions: np.ndarray


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


def _transform_blocks(
    reducers: Mapping[str, Pipeline],
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        name: reducer.transform(arrays[name][indices])
        for name, reducer in reducers.items()
    }


def _concatenate_blocks(blocks: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(list(blocks.values()), axis=1)


def _feature_groups(blocks: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return concatenated-column indices grouped by source modality."""

    grouped: dict[str, list[np.ndarray]] = {}
    offset = 0
    for name, values in blocks.items():
        indices = np.arange(offset, offset + values.shape[1], dtype=np.int64)
        modality = name.split("_", 1)[0]
        grouped.setdefault(modality, []).append(indices)
        offset += values.shape[1]
    return {
        name: np.concatenate(parts)
        for name, parts in grouped.items()
    }


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


def _per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
) -> list[dict[str, float | int]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=classes,
        zero_division=0,
    )
    return [
        {
            "class_id": int(label),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(classes)
    ]


def _per_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for group in sorted({str(value) for value in groups}):
        indices = np.asarray([str(value) == group for value in groups])
        result.append(
            {
                "group": group,
                "clips": int(indices.sum()),
                "metrics": _metrics(y_true[indices], y_pred[indices]),
            }
        )
    return result


def _aligned_probabilities(
    classifier: Any,
    features: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray | None:
    if not hasattr(classifier, "predict_proba"):
        return None
    probabilities = np.asarray(classifier.predict_proba(features), dtype=np.float32)
    classifier_classes = np.asarray(classifier.classes_)
    aligned = np.zeros((len(features), len(classes)), dtype=np.float32)
    positions = {int(label): index for index, label in enumerate(classes)}
    for source_index, label in enumerate(classifier_classes):
        target_index = positions.get(int(label))
        if target_index is not None:
            aligned[:, target_index] = probabilities[:, source_index]
    return aligned


def _majority_vote(
    predictions: np.ndarray,
    classes: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """Vote across repeats, breaking ties with a seeded draw.

    ``np.argmax`` resolves every tie to the lowest class id, which biases the
    consensus towards low-numbered actions whenever ``n_repeats`` is even and
    the repeats disagree.
    """

    generator = np.random.default_rng(random_state)
    result = np.empty(predictions.shape[1], dtype=classes.dtype)
    for index in range(predictions.shape[1]):
        counts = np.asarray([(predictions[:, index] == label).sum() for label in classes])
        tied = np.flatnonzero(counts == counts.max())
        position = int(tied[0]) if tied.size == 1 else int(generator.choice(tied))
        result[index] = classes[position]
    return result


def cross_validate_detailed(
    bundle: RawFeatureBundle,
    algorithm: TrainingAlgorithm,
    parameter_candidates: Sequence[Mapping[str, Any]],
    *,
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
    selection_metric: str = "macro_f1",
) -> tuple[dict[str, Any], dict[str, object], ValidationOutputs]:
    """Evaluate parameters and return reproducible out-of-fold outputs."""

    if bundle.labels is None or bundle.groups is None:
        raise ValueError("Cross-validation requires labels and participant groups")
    if n_splits < 2:
        raise ValueError("Cross-validation requires at least two folds")
    if n_repeats < 1:
        raise ValueError("Cross-validation requires at least one repeat")
    if selection_metric not in {"accuracy", "macro_f1", "balanced_accuracy"}:
        raise ValueError(f"Unsupported selection metric: {selection_metric}")

    arrays = bundle.modality_arrays()
    labels = bundle.labels
    groups = bundle.groups
    candidates = [algorithm.resolved_parameters(parameters) for parameters in parameter_candidates]
    if not candidates:
        raise ValueError("Cross-validation requires at least one parameter candidate")
    classes = np.unique(labels)
    predictions = [
        np.full((n_repeats, len(labels)), -1, dtype=np.int64)
        for _ in candidates
    ]
    probabilities = [
        np.full((n_repeats, len(labels), len(classes)), np.nan, dtype=np.float32)
        for _ in candidates
    ]
    fold_assignments = np.full((n_repeats, len(labels)), -1, dtype=np.int16)

    fold_summaries: list[dict[str, object]] = []
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state + repeat,
        )
        for fold_in_repeat, (train_indices, validation_indices) in enumerate(
            splitter.split(np.zeros(len(labels)), labels, groups), start=1
        ):
            fold = repeat * n_splits + fold_in_repeat
            fold_seed = random_state + repeat * 1_000 + fold_in_repeat
            fold_assignments[repeat, validation_indices] = fold_in_repeat
            reducers = _fit_reducers(arrays, train_indices, fold_seed)
            train_blocks = _transform_blocks(reducers, arrays, train_indices)
            validation_blocks = _transform_blocks(reducers, arrays, validation_indices)
            feature_groups = _feature_groups(train_blocks)
            train_reduced = _concatenate_blocks(train_blocks)
            validation_reduced = _concatenate_blocks(validation_blocks)
            scaler = StandardScaler().fit(train_reduced)
            train_scaled = scaler.transform(train_reduced)
            validation_scaled = scaler.transform(validation_reduced)

            fold_metrics: dict[str, object] = {
                "fold": fold,
                "repeat": repeat + 1,
                "fold_in_repeat": fold_in_repeat,
                "training_clips": int(len(train_indices)),
                "validation_clips": int(len(validation_indices)),
                "validation_users": sorted(
                    {str(groups[index]) for index in validation_indices}
                ),
                "candidates": [],
            }
            for candidate_index, parameters in enumerate(candidates):
                classifier = algorithm.fit_estimator(
                    train_scaled,
                    labels[train_indices],
                    parameters,
                    fold_seed,
                    feature_groups,
                )
                predicted = classifier.predict(validation_scaled)
                predictions[candidate_index][repeat, validation_indices] = predicted
                predicted_probabilities = _aligned_probabilities(
                    classifier,
                    validation_scaled,
                    classes,
                )
                if predicted_probabilities is not None:
                    probabilities[candidate_index][
                        repeat, validation_indices
                    ] = predicted_probabilities
                fold_metrics["candidates"].append(
                    {
                        "parameters": parameters,
                        "metrics": _metrics(labels[validation_indices], predicted),
                    }
                )
            fold_summaries.append(fold_metrics)
            repeat_label = f" repeat {repeat + 1}/{n_repeats}," if n_repeats > 1 else ""
            print(
                f"Completed grouped{repeat_label} fold {fold_in_repeat}/{n_splits} "
                f"({len(validation_indices):,} validation clips)",
                flush=True,
            )

    consensus_predictions: list[np.ndarray] = []
    consensus_probabilities: list[np.ndarray | None] = []
    aggregate: list[dict[str, object]] = []
    for parameters, candidate_predictions, candidate_probabilities in zip(
        candidates,
        predictions,
        probabilities,
        strict=True,
    ):
        if (candidate_predictions < 0).any():
            raise RuntimeError("Cross-validation did not predict every clip")
        if n_repeats == 1:
            averaged_probabilities = (
                candidate_probabilities[0]
                if np.isfinite(candidate_probabilities).all()
                else None
            )
            consensus = candidate_predictions[0]
        elif np.isfinite(candidate_probabilities).all():
            averaged_probabilities = np.mean(candidate_probabilities, axis=0)
            consensus = classes[np.argmax(averaged_probabilities, axis=1)]
        else:
            averaged_probabilities = None
            consensus = _majority_vote(candidate_predictions, classes, random_state)
        repeat_metrics = [
            _metrics(labels, candidate_predictions[repeat])
            for repeat in range(n_repeats)
        ]
        metrics_std = {
            name: float(np.std([metrics[name] for metrics in repeat_metrics]))
            for name in repeat_metrics[0]
        }
        mean_metrics = {
            name: float(np.mean([metrics[name] for metrics in repeat_metrics]))
            for name in repeat_metrics[0]
        }
        consensus_predictions.append(consensus)
        consensus_probabilities.append(averaged_probabilities)
        aggregate.append(
            {
                "parameters": parameters,
                # ``metrics`` describes one fitted model, matching the single
                # estimator fit_final_model() ships. Averaging probabilities
                # across repeats instead measures an n_repeats-model ensemble,
                # which nothing in this repository trains or saves, so that
                # number is reported separately as ``consensus_metrics``.
                "metrics": mean_metrics,
                "metrics_std": metrics_std,
                "repeat_metrics": repeat_metrics,
                "consensus_metrics": _metrics(labels, consensus),
            }
        )
    best_index = max(
        range(len(candidates)),
        key=lambda index: aggregate[index]["metrics"][selection_metric],
    )
    selected_parameters = candidates[best_index]
    selected_predictions = consensus_predictions[best_index]
    report: dict[str, object] = {
        "algorithm": algorithm.name,
        "selection_metric": selection_metric,
        "selected_parameters": selected_parameters,
        "cv_repeats": n_repeats,
        "cv_random_state": random_state,
        # ``classes`` records the class ids behind every positional structure
        # below. Readers must not assume row i of the confusion matrix is
        # action i: ids are only contiguous while every action survives the
        # strict modality filter.
        "classes": [int(label) for label in classes],
        "metrics_basis": "mean_of_repeats",
        "detail_metrics_basis": "repeat_consensus",
        "aggregate_metrics": aggregate,
        "folds": fold_summaries,
        "per_class_metrics": _per_class_metrics(labels, selected_predictions, classes),
        "per_group_metrics": _per_group_metrics(labels, selected_predictions, groups),
        "confusion_matrix": confusion_matrix(
            labels,
            selected_predictions,
            labels=classes,
        ).tolist(),
    }
    outputs = ValidationOutputs(
        clip_ids=bundle.clip_ids.copy(),
        labels=labels.copy(),
        predictions=selected_predictions,
        probabilities=consensus_probabilities[best_index],
        classes=classes,
        fold_assignments=fold_assignments,
        repeat_predictions=predictions[best_index],
    )
    return selected_parameters, report, outputs


def cross_validate(
    bundle: RawFeatureBundle,
    algorithm: TrainingAlgorithm,
    parameter_candidates: Sequence[Mapping[str, Any]],
    *,
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
    selection_metric: str = "macro_f1",
) -> tuple[dict[str, Any], dict[str, object]]:
    """Compatibility wrapper returning the original two-value result."""

    selected, report, _ = cross_validate_detailed(
        bundle,
        algorithm,
        parameter_candidates,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
        selection_metric=selection_metric,
    )
    return selected, report


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
    blocks = _transform_blocks(reducers, arrays, indices)
    feature_groups = _feature_groups(blocks)
    reduced = _concatenate_blocks(blocks)
    scaler = StandardScaler().fit(reduced)
    resolved_parameters = algorithm.resolved_parameters(parameters)
    classifier = algorithm.fit_estimator(
        scaler.transform(reduced),
        bundle.labels,
        resolved_parameters,
        random_state,
        feature_groups,
    )
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


def save_validation_outputs(outputs: ValidationOutputs, path: str | Path) -> Path:
    """Save compact out-of-fold predictions and fold assignments."""

    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "clip_ids": outputs.clip_ids,
        "labels": outputs.labels,
        "predictions": outputs.predictions,
        "classes": outputs.classes,
        "fold_assignments": outputs.fold_assignments,
        "repeat_predictions": outputs.repeat_predictions,
    }
    if outputs.probabilities is not None:
        payload["probabilities"] = outputs.probabilities
    np.savez_compressed(output, **payload)
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
