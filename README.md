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
│   ├── build_manifest.py
│   └── requirements.txt
├── tests/                    # Repository tests
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
  speed, and manual scrubbing controls;
- animated 3D skeletons, IMU magnitude traces, and radar point clouds;
- predicted test actions loaded from any `path,prediction` submission CSV,
  including an all-clip distribution/table and labels while clips play;
- missing-modality, empty-sensor, and timestamp-alignment diagnostics;
- **Algorithm comparison** page comparing every `artifacts/*/validation.json` (leaderboard, bar chart, confusion matrix + Δ recall, folds, metadata) — same source as `python -m modeling.compare`, and its CSV download is byte-identical to that command's output.

Set up the dashboard:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
CUHKX_DATASET_ROOT=/path/to/small-model streamlit run visualization/app.py
```

After `modeling.train` or `modeling.predict` creates a `*_submission.csv` under
`outputs/`, the dashboard automatically selects the newest one. The sidebar also accepts
another CSV path or a direct upload. In `Clip explorer`, test clips show the
predicted action ID and name in both the selector and a banner above playback;
`Overview` shows prediction coverage, class distribution, and a per-clip table.

### Dataset root

Every entry point resolves the dataset root the same way, highest precedence
first:

1. an explicit value — the dashboard sidebar, `--dataset-root`, or the script's
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

The Docker image contains the visualization module only. Mount the dataset
read-only at `/data` so it is not copied into the image. Mount `artifacts`
writable so the progressive loader can save `cuhkx_manifest.parquet` and its
progress file, and so the cache persists after the container is removed. The
same mount also exposes algorithm reports such as
`artifacts/logreg/validation.json`:

```bash
docker build -t cuhkx-visualization .
mkdir -p artifacts
docker run --rm -p 127.0.0.1:8501:8501 \
  -v /path/to/small-model:/data:ro \
  -v "$(pwd)/artifacts:/app/artifacts" \
  cuhkx-visualization
```

On Linux add `--user "$(id -u):$(id -g)"`. The image runs as uid 10001, and a
bind mount keeps the host directory's ownership, so without it the background
builder cannot write into `artifacts`. Docker Desktop on macOS remaps ownership
and needs no extra flag.

Use `:ro` on the artifacts mount only when `cuhkx_manifest.parquet` has already
been built and the dashboard will only consume existing manifests and
validation reports. A read-only artifacts mount cannot run progressive cache
generation or rebuild the manifest; the dashboard reports the failed build
instead of stopping.

Then open <http://localhost:8501>. See `visualization/README.md` for the
module-specific commands and layout. Real scores from the reference run: `logistic_regression` `C=0.01` → `accuracy 0.4539, macro_f1 0.3896` on 2,785 clips.

## Logistic-regression baseline

The `modeling/` module provides an extensible four-modality training framework
using Depth Color, IR, IMU, and Skeleton features. It strictly filters
incomplete training clips, validates by held-out participant groups, shares a
feature cache across registered algorithms, and writes submissions in test CSV
order. Logistic regression is the initial registered algorithm.

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

See `modeling/README.md` for feature definitions, filtering rules, artifacts,
and prediction-only usage.

## Tests

No test needs the dataset, and the whole suite runs in well under a second.
Run it from the repository root:

```bash
python3 -m unittest discover -s tests -t .
```

There are five suites with different requirements:

| Suite | Tests | Requires | Behavior without the requirement |
|-------|-------|----------|----------------------------------|
| `tests/test_visualization_dataset.py` | 15 | standard library only | always runs |
| `tests/test_predictions.py` | 7 | standard library only | always runs |
| `tests/test_comparison_format.py` | 13 | standard library only | always runs; its 2 CLI-parity tests skip without `modeling/requirements.txt` |
| `tests/test_algorithm_comparison.py` | 16 | `visualization/requirements.txt` + `numpy` (for confusion-matrix math) | collapses to 1 skip |
| `tests/test_modeling.py` | 11 | `modeling/requirements.txt` | collapses to 1 skip |

So a bare interpreter reports `Ran 37 tests ... OK (skipped=4)`, while an
interpreter with `visualization` + `modeling` deps reports `Ran 62 tests ... OK`
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
