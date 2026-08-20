# Extensible multimodal training

This module compares classification algorithms on a shared 384-dimensional
representation of Depth Color, IR, IMU, and Skeleton data. Training uses a
strict complete-case filter:

- all four modality directories must contain data;
- both IMU files must contain rows;
- all five devices (`WTLA`, `WTRA`, `WTC`, `WTLL`, `WTRL`) must occur.

The current local dataset retains 2,785 training clips and all 40 classes.
Missing or corrupt test feature blocks are mean-imputed by preprocessing
learned exclusively from the training split. In the inspected test copy, four
IR clips contain zero-filled placeholder files and one clip has no IMU rows;
some additional clips have only partial IMU device coverage.

## Features

- Depth Color and IR: grayscale 64x48 frames, five temporal summaries, an
  in-repository NumPy Histogram of Oriented Gradients implementation, coarse
  intensity grids, and percentiles.
- IMU: time-sorted accelerometer and gyroscope axes plus magnitudes, global
  statistics, and four-bin temporal statistics for each of five devices.
- Skeleton: primary-person tracking, hip centering, torso normalization,
  interpolation to 32 frames, positions, velocities, and confidence scores.

Each modality is independently mean-imputed, standardized, and reduced with
PCA. The four blocks produce the same 384 inputs for every registered
algorithm, keeping comparisons consistent.

## Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r modeling/requirements.txt
```

## Train and create a submission

```bash
python -m modeling.train \
  --algorithm logistic_regression \
  --dataset-root ~/AAI/opt-scratch/small-model \
  --n-jobs 8
```

The command performs participant-grouped cross-validation, fits the final
model, and writes:

- `artifacts/features/four_sensor_v1.npz`: algorithm-independent feature cache;
- `artifacts/logreg/model.joblib`: fitted preprocessing and classifier;
- `artifacts/logreg/validation.json`: metrics and confusion matrix;
- `outputs/logreg_submission.csv`: predictions in test CSV order.

Override the comparison grid with JSON:

```bash
python -m modeling.train \
  --algorithm logistic_regression \
  --search-space '{"C": [0.01, 0.1, 1.0]}'
```

Feature extraction can be run by itself:

```bash
python -m modeling.train --extract-only --n-jobs 8
```

Generate predictions again from an existing model:

```bash
python -m modeling.predict --algorithm logistic_regression
```

`--algorithm` selects which saved artifacts to load, defaulting to
`artifacts/<artifact_name>/model.joblib`. The submission path is named after the
algorithm recorded inside the loaded model, so a new algorithm writes to its own
file without any flag changes. Pass `--model` or `--output` to override either
path explicitly.

The previous `modeling.train_logreg` and `modeling.predict_logreg` entry points
remain as compatibility wrappers.

## Library versions and artifact portability

Estimator pickles are not guaranteed portable across scikit-learn versions, so
every model records the versions that produced it, and `validation.json`
repeats them under `library_versions`. Loading a model raises a
`RuntimeWarning` when the running versions differ, or when the artifact predates
this metadata:

```
artifacts/logreg/model.joblib was written with different library versions
(scikit-learn 1.9.0 -> 1.6.1). Predictions may differ from the original run;
retrain to be certain.
```

The warning is advisory, not fatal — a mismatched pickle often still loads and
predicts identically. Treat it as a prompt to retrain before trusting a
submission produced under different versions. `requirements.txt` intentionally
keeps a permissive `scikit-learn>=1.5,<2` range rather than pinning an exact
version, since the warning now makes any drift visible.

## Add another algorithm

Create a module under `modeling/algorithms/` that implements
`TrainingAlgorithm`, then register one instance:

```python
from sklearn.svm import SVC

from modeling.algorithms.base import TrainingAlgorithm, register_algorithm


class RbfSvmAlgorithm(TrainingAlgorithm):
    name = "rbf_svm"
    display_name = "RBF support-vector machine"
    artifact_name = "rbf_svm"
    default_parameters = {"C": 1.0, "gamma": "scale"}
    default_search_space = {
        "C": (0.1, 1.0, 10.0),
        "gamma": ("scale", "auto"),
    }

    def build_estimator(self, parameters, random_state):
        resolved = self.resolved_parameters(parameters)
        return SVC(**resolved, class_weight="balanced", random_state=random_state)


ALGORITHM = register_algorithm(RbfSvmAlgorithm())
```

Import the new module from `modeling/algorithms/__init__.py`. It then becomes a
valid `--algorithm` choice and automatically receives the same filtering,
feature cache, grouped folds, metrics, artifact format, and submission logic.

Compare completed runs with:

```bash
python -m modeling.compare                        # tab-separated to stdout
python -m modeling.compare --output comparison.csv
```

Columns: `label`, `algorithm`, `artifact_name`, `parameters`, `accuracy`,
`macro_f1`, `balanced_accuracy`, `training_clips`, `report`, sorted by
`macro_f1` descending. `algorithm` alone is not a key — two artifact
directories can hold runs of the same algorithm — so `label` disambiguates
them as `logistic_regression (logreg_tuned)` while leaving unique names bare.

The format is defined once in `visualization/comparison_format.py` (standard
library only, so this CLI gains no visualization dependencies) and shared with
the dashboard's `Algorithm comparison` page, which writes a byte-identical CSV
from its download button.

All scaling, imputation, and PCA fitting happens inside each validation fold.
The validation groups are participants, preventing clips from one person from
appearing in both training and validation data. Every algorithm report uses the
same schema, including selected parameters, accuracy, macro-F1, balanced
accuracy, per-fold metrics, and a confusion matrix.
