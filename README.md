---
language:
- en
pretty_name: CUHK-X — Small Model Track
tags:
- multimodal
- human-activity
- action-recognition
- depth
- infrared
- thermal
- imu
- radar
- skeleton
task_categories:
- video-classification
---

# CUHK-X — Small Model Track

Multimodal **human action recognition (classification)**.
Given a multimodal clip, predict its action class (`action_id`, **0–39, 40 classes**).

## Repository layout

```
.
├── visualization/            # Streamlit dashboard and data indexing
│   ├── app.py
│   ├── dataset.py
│   ├── playback.py
│   ├── predictions.py
│   ├── training_pipeline.py
│   ├── build_manifest.py
│   └── requirements.txt
├── tests/                    # Repository tests
├── scripts/
│   └── prepare_sample_dataset.py # compact local UI/training fixture
├── Dockerfile                # Visualization container
├── Training/
│   └── class_mapping.csv     # action_id <-> action_name (40 classes)
└── Testing/
    ├── test.csv              # path + empty `prediction` (to fill)
    └── sample_submission.csv # submission example
```

## Dataset layout

The dataset is roughly 49 GB and is **not** stored in this repository. Keep a
local copy elsewhere and point `CUHKX_DATASET_ROOT` (or the dashboard sidebar)
at a directory laid out as:

```
<dataset-root>/
├── Training/
│   └── data/
│       └── HAR.z01 … HAR.z08 + HAR.zip   # multi-volume zip
│           →  HAR/data/<modality>/<action>/<user>/<trial>/<files>
└── Testing/
    └── data/
        └── small_model_track_test.zip
            →  small_model_track_test/<id>/<modality>/<files>
```

## Labels

- **Training labels live in the path**: in `HAR/data/<modality>/<action>/<user>/<trial>`,
  the `<action>` (e.g. `0_Wash_face`) is the class.
- Convert between `action_name` and `action_id` with `class_mapping.csv`.
- Test clips are anonymized (`SM_test_XXXX`); predict their `action_id`.

## Modalities (6; no RGB, no raw Depth)

| Modality      | Type                     | Example file                       |
|---------------|--------------------------|------------------------------------|
| `Depth_Color` | colorized depth (frames) | `Depth_<datetime>_<idx>_Color.png` |
| `IR`          | infrared (frames)        | `IR_<datetime>_<idx>.png`          |
| `Thermal`     | thermal (frames)         | `frame_000063.jpg`                 |
| `IMU`         | inertial sensor          | `*.csv`                            |
| `Radar`       | mmWave radar             | `radar_output_T<ts>.csv`           |
| `Skeleton`    | skeleton                 | pose data + `visualizations/`      |

Sampling rates differ across modalities; not every clip has every modality.

## Extracting the data

Training is a multi-volume zip (`HAR.z01`…`HAR.z08` + `HAR.zip`; keep all volumes in one folder):

All paths below are relative to `<dataset-root>`, not to this repository.

```bash
cd <dataset-root>/Training/data
zip -s 0 HAR.zip --out HAR_full.zip   # merge volumes (zip 3.0+)
unzip HAR_full.zip                    # -> HAR/data/<modality>/<action>/<user>/<trial>/...
```

7-Zip / WinRAR / double-click also handle split zips. Test set:

```bash
cd <dataset-root>/Testing/data && unzip small_model_track_test.zip   # -> small_model_track_test/<id>/<modality>/...
```

## Submission

In `Testing/test.csv`, fill each row's `prediction` with the predicted `action_id` (0–39).
See `sample_submission.csv` for the format.

## Statistics

- 40 action classes
- 405 test clips

## Quick start

```python
import csv
id2name = {r["action_id"]: r["action_name"]
           for r in csv.DictReader(open("Training/class_mapping.csv", encoding="utf-8-sig"))}
# Training: the <action> folder in the path is the label, e.g.
#   HAR/data/IR/0_Wash_face/user10/4-2-1/  ->  action_name="0_Wash_face", action_id="0"
```

## Visualization dashboard

This repository includes a Streamlit dashboard with:

- class balance, user/action coverage, modality availability, and clip-length plots;
- synchronized Depth Color, IR, and Thermal frames with play, pause, restart,
  speed, and manual scrubbing controls; playback stays in one browser component
  so advancing a frame does not remount the images or 3D viewers;
- a **Manual labeling** page that reviews test clips one at a time and records
  action IDs in `Testing/test_manual_label.csv`, one tracked table that both
  the sample and the full dataset write to;
- animated 3D skeletons, IMU magnitude traces, and radar point clouds;
- one **Workflow** page for choosing the sample or full dataset, extracting a
  named feature cache, training to explicit model/submission paths, and running
  prediction to a named CSV;
- automatic handoff of the selected dataset to every page, and of the
  prediction CSV to **Clip explorer**, where a **Prediction file** dropdown
  shows them with the synchronized sensor streams;
- missing-modality, empty-sensor, and timestamp-alignment diagnostics;
- **Algorithm comparison** page comparing every `artifacts/*/validation.json` (leaderboard, bar chart, confusion matrix + Δ recall, folds, metadata) — same source as `python -m modeling.compare`, and its CSV download is byte-identical to that command's output.

Set up the dashboard:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
CUHKX_DATASET_ROOT=/path/to/small-model streamlit run visualization/app.py
```

Open **Workflow** and choose **Sample dataset** or **Full dataset** once. The
same dataset selection is then used by feature extraction, training,
prediction, Overview, Data quality, Clip explorer, and Manual labeling. Each
workflow stage has its own run button and explicit output path. Background
feature/training runs keep live progress and timestamped logs under
`artifacts/training_runs/`.

For a quick end-to-end check without processing the full dataset, create the
ignored local fixture once:

```bash
python scripts/prepare_sample_dataset.py --source /path/to/small-model
```

This writes an approximately 30 MB, balanced fixture to
`artifacts/sample_dataset` (8 training clips, 2 test clips). Selecting **Sample
dataset** on Workflow also selects two folds, a one-candidate search, and
`artifacts/features/ui_sample.npz`.

After training, the generated submission is selected automatically. You can
also choose a model and an exact prediction output filename in Workflow, click
**Run prediction** for train, test, or both splits, then click **Visualize
predictions with sensor data**. Test-only output keeps the competition
`path,prediction` format; train and combined output use `clip_id,prediction` so
training clip paths remain intact. The saved CSV remains discoverable after
restarting the dashboard.

### Dataset root

Every entry point resolves the dataset root the same way, highest precedence
first:

1. an explicit value — the dashboard Workflow page, `--dataset-root`, or the script's
   first argument;
2. the `CUHKX_DATASET_ROOT` environment variable;
3. the built-in default, `~/AAI/opt-scratch/small-model`.

Change the default for everyone by editing `DEFAULT_DATASET_ROOT` in
`visualization/dataset.py`. The Docker image sets the environment variable to
`/data`.

### Build a reusable manifest

The manifest contains one row per logical clip and avoids rescanning the
archives on every launch (strongly recommended — `Overview`/`Clip explorer`/`Data quality` then load in ~0.2 s via `artifacts/cuhkx_manifest.parquet` auto-detected by the dashboard, instead of live-scanning 3k+ clips and running `deep_test` JSON checks):

```bash
python -m visualization.build_manifest \
  --dataset-root /path/to/small-model \
  --output artifacts/cuhkx_manifest.parquet
```

The dashboard pre-fills the sidebar with `artifacts/cuhkx_manifest.parquet` when it exists. If it is missing and the dataset is extracted, the dashboard first loads a representative 200 clips, builds the complete manifest in a separate process, shows exact file-count progress, and automatically switches to it when ready. A banner marks the view as partial, and quality checks that the shallow preview cannot run show `—` rather than `0`. Only extracted splits can be sampled, so a split still held in an archive is named in that banner and appears once the complete index is ready. Untick `Load 200 clips first` to use the original blocking live scan. Untick `Use saved manifest` to rebuild a stale manifest: the dashboard shows the preview again, runs a fresh build, and switches back when it finishes. If a build fails, `Clear cached index` retries it. The `Algorithm comparison` page does not need the dataset at all — it reads only `artifacts/*/validation.json`. `Clip explorer` indexes only the selected clip (~0.02 s) instead of all 3k+ clips (19.5 s train / 2.1 s test). CSV and JSON manifests are also supported (use `.csv` / `.json` suffix).

### Multipart training archive

The dashboard can index the original multipart training ZIP with `zipinfo`,
but Python cannot read frames directly from a split ZIP. Training playback is
enabled when either of these exists:

- `Training/data/HAR/data/...` (extracted data), or
- `Training/data/HAR_full.zip` (a merged ZIP).

Some distributed copies use different basenames for the `.z01`–`.z08`
volumes. Normalize those part names to `HAR.z01`–`HAR.z08` before running the
merge command shown above. The original files can be preserved by using
symlinks.

### Docker

The Docker image contains both visualization and modeling dependencies. Mount
the dataset read-only at `/data` so it is not copied into the image. Mount
`artifacts` and `outputs` writable so training caches, models, reports, logs,
and submissions persist after the container is removed. On first start the
container auto-prepares `artifacts/sample_dataset` from `/data` so **Workflow
→ Sample dataset** is immediately usable (no manual `prepare_sample_dataset.py`
needed) and `/data` is ready for **Full dataset**:

```bash
docker build -t cuhkx-visualization .
mkdir -p artifacts outputs
docker run --rm -p 127.0.0.1:8501:8501 \
  -v /path/to/small-model:/data:ro \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/outputs:/app/outputs" \
  cuhkx-visualization
# logs: [entrypoint] Preparing sample dataset from /data ... Sample dataset ready
```

On Linux add `--user "$(id -u):$(id -g)"`. The image runs as uid 10001, and a
bind mount keeps the host directory's ownership, so without it the background
builder cannot write into `artifacts`. Docker Desktop on macOS remaps ownership
and needs no extra flag.

Use `:ro` on the artifacts mount only when the dashboard will not build an
index or run training. A read-only artifacts mount cannot create the manifest,
feature cache, models, validation reports, or training logs.

Then open <http://localhost:8501>. See `visualization/README.md` for the
module-specific commands and layout. Real scores on 2,785 clips, both measured
the same way — three repeated grouped splits, averaged over repeats, so each
number describes one fitted model:

| Run | accuracy | macro-F1 | balanced accuracy |
| --- | --- | --- | --- |
| `logistic_regression` (`C=0.1`) | 0.5370 | 0.4751 | 0.4681 |
| `late_fusion` (`C=0.03`, combined `C=0.1`, `stacked`) | 0.5587 | 0.4949 | 0.4897 |

Quote scores only against runs with the same `cv_repeats`: a single split puts
`logistic_regression` at `accuracy 0.5382, macro_f1 0.4765`, which is within the
repeat-to-repeat spread rather than a real difference.

## Modeling baselines

The `modeling/` module provides an extensible multimodal training framework
using Depth Color, IR, IMU, and Skeleton features. It strictly filters
incomplete training clips, validates by held-out participant groups, shares a
feature cache across registered algorithms, and writes submissions in test CSV
order. The registered comparisons are multinomial logistic regression and
modality-level late fusion. Both consume the same cached feature representation
and preserve the submission/artifact formats.

Install and run it from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r modeling/requirements.txt
python -m modeling.train \
  --algorithm logistic_regression \
  --dataset-root /path/to/small-model \
  --n-jobs 8
```

For a repeated same-feature comparison:

```bash
python -m modeling.train \
  --algorithm late_fusion \
  --dataset-root /path/to/small-model \
  --cv-repeats 3 \
  --n-jobs 8
```

See `modeling/README.md` for feature definitions, filtering rules, artifacts,
and prediction-only usage.

## Tests

No test needs the dataset, and the whole suite runs in well under a second.
Run it from the repository root:

```bash
python3 -m unittest discover -s tests -t .
```

There are eight suites with different requirements:

| Suite | Tests | Requires | Behavior without the requirement |
|-------|-------|----------|----------------------------------|
| `tests/test_visualization_dataset.py` | 16 | standard library only | always runs |
| `tests/test_predictions.py` | 9 | standard library only | always runs |
| `tests/test_training_pipeline.py` | 29 | standard library only | always runs |
| `tests/test_comparison_format.py` | 13 | standard library only | always runs; its 2 CLI-parity tests skip without `modeling/requirements.txt` |
| `tests/test_manual_labels.py` | 22 | standard library only | always runs |
| `tests/test_model_predictions.py` | 39 | artifact discovery and CSV round-tripping need the standard library only; the rest need `numpy`, `visualization/requirements.txt` (for `visualization.app`), or `modeling/requirements.txt` | 5 tests always run; the other 34 skip. Dataset I/O is mocked throughout, so no suite needs the dataset |
| `tests/test_algorithm_comparison.py` | 19 | `visualization/requirements.txt` + `numpy` (for confusion-matrix math) | collapses to 1 skip |
| `tests/test_modeling.py` | 32 | `modeling/requirements.txt` | collapses to 1 skip |

So a bare interpreter reports `Ran 130 tests ... OK (skipped=37)`, while an
interpreter with `visualization` + `modeling` deps reports `Ran 179 tests ... OK`
(`skipped=1` when the optional real validation artifact is absent). A skip is
expected, not a failure. Install the extras to run
everything:

```bash
pip install -r visualization/requirements.txt   # for algorithm comparison tests
pip install -r modeling/requirements.txt        # for modeling tests
python3 -m unittest discover -s tests -t .
```

Run a single module verbosely:

```bash
python3 -m unittest tests.test_visualization_dataset -v
```

