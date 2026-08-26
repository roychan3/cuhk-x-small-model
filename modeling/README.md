# Extensible multimodal training

This module compares classification algorithms on a shared representation of
Depth Color, IR, IMU, and Skeleton data. Training uses a strict complete-case
filter:

- all four modality directories must contain data;
- both IMU files must contain rows;
- all five devices (`WTLA`, `WTRA`, `WTC`, `WTLL`, `WTRL`) must occur.

The current local dataset retains 2,785 training clips and all 40 classes.
Missing or corrupt test feature blocks are mean-imputed by preprocessing
learned exclusively from the training split. In the inspected test copy, four
IR clips contain zero-filled placeholder files and one clip has no IMU rows;
some additional clips have only partial IMU device coverage.

## Features

- Depth Color base features: grayscale 64x48 frames, five temporal summaries,
  an in-repository NumPy Histogram of Oriented Gradients implementation,
  coarse intensity grids, and percentiles. The equivalent legacy IR block is
  disabled after grouped validation showed no benefit.
- Image engineered features: four-bin temporal pyramids containing appearance,
  absolute motion, signed motion, and motion-centroid grids. Depth Color also
  retains HSV spatial summaries and histograms instead of discarding the
  colorized depth encoding.
- IMU base features: time-sorted accelerometer and gyroscope axes plus
  magnitudes, global statistics, and four-bin temporal statistics for each of
  five devices.
- IMU engineered features: jerk and spectral summaries, circular Euler-angle
  statistics, relative quaternion motion, per-device covariance, sampling
  metadata, and left/right or limb/torso agreement features.
- Skeleton base features: primary-person tracking, pelvis centering, torso
  normalization, interpolation to 32 frames, positions, velocities, and
  confidence scores.
- Skeleton engineered features: joint angles and distances, torso geometry,
  root displacement, per-joint speed and acceleration, confidence summaries,
  and periodic-motion statistics.

Skeleton predictions use the **Human3.6M 17-joint order, not COCO**: joint 0 is
the root pelvis (at exactly `x == y == 0` in every frame), 10 is the head, and
3 and 6 are the ankles. Index them through the named constants in
`modeling/features.py` (`PELVIS`, `L_WRIST`, `HEAD`, …) rather than by number.
Reading these poses as COCO is silent — every index is in range and the
features still extract — but it turns "wrist to head" into a near-rigid
trunk length and inverts the torso vector. Artifacts written before this was
corrected record no `skeleton_layout` in their feature config, and
`feature_config_from_artifact` refuses them rather than predicting from
features they were never fitted on.

The three active high-dimensional base blocks are independently mean-imputed,
standardized, and reduced with PCA to 320 total components. The four compact
engineered blocks are independently imputed and standardized, then passed
through without sharing a PCA projection with the base features. With the
default configuration, every registered algorithm receives 2,833 inputs.
Models created before this change remain loadable: prediction automatically
re-enables the legacy IR extractor when an older model requires that block.

Registered algorithms include the original multinomial logistic regression and
modality-level late fusion. Both use these same cached feature blocks.
Late fusion trains one classifier per Depth, IR, IMU, and Skeleton group and
blends their probabilities with the original combined-feature classifier; it
does not add another feature extractor. On the reference data, modality
`C=0.03`, combined `C=0.1`, and the `stacked` weighting profile reach `accuracy
0.5587, macro_f1 0.4949, balanced_accuracy 0.4897` with three repeated grouped
splits, compared with `0.5370 / 0.4751 / 0.4681` for the original logistic
regression under the same repeated evaluation. Both figures are means over the
three repeats, so both describe a single fitted model.

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

- `artifacts/features/four_sensor_v3.npz`: algorithm-independent feature cache;
- `artifacts/logreg/model.joblib`: fitted preprocessing and classifier;
- `artifacts/logreg/validation.json`: metrics and confusion matrix;
- `artifacts/logreg/oof_predictions.npz`: clip-aligned out-of-fold predictions,
  probabilities, repeat predictions, and fold assignments;
- `outputs/logreg_submission.csv`: predictions in test CSV order.

Pass `--progress-file artifacts/training-progress.json` to publish atomic JSON
updates for dataset discovery, feature extraction, validation, final fitting,
and saving. The file carries `status` (`running`/`complete`/`error`), `stage`,
`current`/`total`, `message`, and `updated_at`, so a reader polling only the
file can tell a finished run from one that died mid-stage; a failure keeps the
stage it happened in. `visualization/build_manifest.py --progress-file` writes
the same schema through the shared `visualization/progress.py` writer. The
dashboard's **Training pipeline** page uses this interface and keeps one
progress file plus a complete log for every run.

Override the comparison grid with JSON:

```bash
python -m modeling.train \
  --algorithm logistic_regression \
  --search-space '{"C": [0.01, 0.1, 1.0]}'
```

Run a stronger same-feature comparison with repeated grouped validation:

```bash
python -m modeling.train \
  --algorithm late_fusion \
  --cv-repeats 3
```

Each algorithm writes to its own artifact/output directory and appears
automatically in `python -m modeling.compare` and the dashboard.

Feature extraction can be run by itself:

```bash
python -m modeling.train --extract-only --n-jobs 8
```

`modeling.features.extract_feature_bundle` now accepts an optional `progress_callback(done, total)` (called every `progress_every` clips, default 50) which the dashboard wires to a `st.progress` bar. The `visualization.predictions.generate_*` helpers expose a wider `progress_callback(done, total, message)` and adapt it to this two-argument form, so `Both (train + test)` maps setup to `0→5%`, train to `5→70%`, test to `70→95%` and finalization to `95→100%` — a single monotonic bar rather than a static spinner. Exceptions from the callback propagate.

```python
from modeling.features import FeatureConfig, extract_feature_bundle
from modeling.data import discover_training_clips

records, _ = discover_training_clips("/path/to/small-model")
bundle = extract_feature_bundle(records, FeatureConfig(), n_jobs=4, progress_callback=lambda done, total: print(f"{done}/{total}"))
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
accuracy, per-fold metrics, per-class metrics, per-participant metrics, and a
confusion matrix.

`--cv-repeats` repeats the grouped split with consecutive deterministic seeds.
Each repeat produces its own complete out-of-fold prediction, so a report at
`cv_repeats > 1` carries three views of the same candidate:

- `metrics` — the mean across repeats, and what candidate selection and the
  leaderboard use. This is the number to quote: it estimates the single model
  that `fit_final_model` actually trains and saves.
- `metrics_std` and `repeat_metrics` — the spread and the individual repeats,
  for judging whether a gap between two algorithms is larger than the noise.
- `consensus_metrics` — the score after averaging probabilities across repeats.
  That is an ensemble of `cv_repeats` models, which nothing here ships, so it
  runs slightly ahead of `metrics` and is not comparable to a single-split run.

`confusion_matrix`, `per_class_metrics`, and `per_group_metrics` are built from
the consensus predictions (`detail_metrics_basis`), which is why they can differ
from `metrics` in the last decimal. `classes` lists the class id behind each
confusion-matrix row: those ids are contiguous only while every action survives
the strict modality filter, so read positions through `classes` rather than
assuming row *i* is action *i*.
