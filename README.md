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
- missing-modality, empty-sensor, and timestamp-alignment diagnostics.

Set up the dashboard:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
CUHKX_DATASET_ROOT=/path/to/small-model streamlit run visualization/app.py
```

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
archives on every launch:

```bash
python -m visualization.build_manifest \
  --dataset-root /path/to/small-model \
  --output artifacts/cuhkx_manifest.parquet
```

Enter the generated manifest path in the dashboard sidebar. CSV and JSON are
also supported; use an output filename ending in `.csv` or `.json`.

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
read-only at `/data` so it is not copied into the image:

```bash
docker build -t cuhkx-visualization .
docker run --rm -p 127.0.0.1:8501:8501 \
  -v /path/to/small-model:/data:ro \
  cuhkx-visualization
```

Then open <http://localhost:8501>. See `visualization/README.md` for the
module-specific commands and layout.

## Tests

The suite uses the standard library only, needs no dataset, and runs in
milliseconds. Run it from the repository root:

```bash
python3 -m unittest discover -s tests -t .
```

Run a single module verbosely:

```bash
python3 -m unittest tests.test_visualization_dataset -v
```
