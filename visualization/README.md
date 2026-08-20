# Visualization module

This module contains all code and dependencies for exploring the CUHK-X
dataset and comparing training algorithms.

## Contents

- `app.py`: Streamlit application — Overview, Clip explorer, Data quality, and Algorithm comparison pages.
- `algorithm_comparison.py`: Algorithm comparison page (reads `artifacts/*/validation.json` from `modeling.train`/`modeling.compare`; no `scikit-learn` needed).
- `comparison_format.py`: standard-library-only definition of the comparison table (columns, row builder, run labels), shared with `modeling/compare.py` so the page and the CLI cannot drift.
- `playback.py`: synchronized timeline and playback timing helpers.
- `dataset.py`: standard-library dataset discovery, indexing, quality checks, archive access, and manifest generation.
- `build_manifest.py`: command-line manifest generator.
- `requirements.txt`: visualization-only Python dependencies.

## Run locally

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r visualization/requirements.txt
CUHKX_DATASET_ROOT=/path/to/small-model streamlit run visualization/app.py
```

Build a reusable manifest (strongly recommended — makes Overview/Clip explorer/Data quality load in ~0.2 s instead of scanning the full dataset and running `deep_test` JSON checks):

```bash
python -m visualization.build_manifest \
  --dataset-root /path/to/small-model \
  --output artifacts/cuhkx_manifest.parquet
```
The sidebar `Saved manifest` field is pre-filled with `artifacts/cuhkx_manifest.parquet` when it exists, so the common case needs no input. When it does not exist, progressive loading indexes a representative 200 clips first (100 train and 100 test when both are available), starts the complete manifest build in a separate process, and shows exact file-count progress until it automatically switches to the finished cache. A banner marks the view as partial until that switch occurs, and `Data quality` shows `—` instead of `0` for the payload checks a shallow preview cannot run (`radar_empty`, `imu_empty_files`) and for `depth_skeleton_aligned` when no sampled clip carried both timestamps. Untick `Load 200 clips first` to force the original blocking live scan.

Untick `Use saved manifest` to rebuild a stale manifest. Each untick starts a fresh builder rather than reusing the previous one, and the dashboard switches back to the saved manifest once the rebuild lands. A failed build is kept as-is so that a rerun cannot respawn a doomed builder on every widget interaction; `Clear cached index` drops it and retries.

Progressive loading requires extracted directory sources. ZIP sources fall back to the complete scan because discovering all files for selected clips still requires reading the archive's full central directory. When only one split is extracted — a merged `HAR.zip` alongside an unpacked `small_model_track_test/`, say — the preview covers just that split and the banner names the one still waiting, so `Training clips 0` is not mistaken for a missing dataset.

### Algorithm comparison

`Algorithm comparison` is the 4th sidebar page and does **not** need the dataset. It discovers every `artifacts/*/validation.json` (one per `python -m modeling.train --algorithm <name>` run) and shows:

* Leaderboard sorted by `macro_f1` / `accuracy` / `balanced_accuracy`, with per-algorithm `selected_parameters`, `training_clips`, and `library_versions` warnings. The `Download comparison CSV` button emits `label, algorithm, artifact_name, parameters, accuracy, macro_f1, balanced_accuracy, training_clips, report` — byte-identical to `python -m modeling.compare --output comparison.csv`, because both read `COMPARISON_FIELDS` from `comparison_format.py`. The on-screen table adds `display_name` and leads with whichever metric you sorted by.
* Metrics bar chart (selected metric or all three)
* Confusion matrix (40×40, row-normalized toggle) + Δ recall `A−B` heatmap and per-class F1
* Cross-validation folds (line per fold + aggregate table) and run metadata (`test_feature_health`, `class_counts`, full JSON)

Add a new algorithm under `modeling/algorithms/` and rerun `python -m modeling.train --algorithm <name>` — it appears with no dashboard code change. Real scores from the reference run: `logistic_regression` (`C=0.01`) → `accuracy 0.4539, macro_f1 0.3896, balanced_accuracy 0.3845` on 2,785 training clips (see `artifacts/logreg/validation.json`). If the container is used only to read existing validation reports and an existing dataset manifest, the artifacts mount can be read-only:

```bash
docker run -d --rm --name cuhkx-dev -p 127.0.0.1:8501:8501 \
  -v /path/to/small-model:/data:ro \
  -v $(pwd)/artifacts:/app/artifacts:ro \
  cuhkx-visualization
```

Remove `:ro` from the artifacts mount when the dashboard should progressively
build or rebuild `cuhkx_manifest.parquet`, and on Linux add
`--user "$(id -u):$(id -g)"` so the container's uid 10001 can write to the
bind-mounted host directory.

### Clip explorer performance

The explorer now indexes only the selected clip (`build_clip_member_index` → `cached_clip_index`) instead of scanning all 3k+ clips. On the reference dataset this drops the first-clip latency from **19.5 s (train) / 2.1 s (test)** to **~0.02 s**. The full-scan helper `build_member_index` / `cached_member_index` is retained for batch operations.

### Fixes

* `display_columns()` deduplicates the leaderboard column list when `Sort by` is already one of `accuracy/macro_f1/balanced_accuracy` (previously `ValueError: Duplicate column names found`).
* `assign_labels()` gives every run a unique identity. `algorithm` alone is not unique — two artifact directories can hold runs of the same algorithm (`artifacts/logreg` and `artifacts/logreg_tuned` both report `logistic_regression`), which made the selectboxes show identical options that both resolved to the first report, and produced CSV rows separable only by their `report` path. Ambiguous names get an `(artifact-dir)` suffix; unique ones stay bare. `modeling/compare.py` shares the helper, so its output gained the same `label` and `artifact_name` columns.
* Figures are keyed off the same label as the leaderboard rows, so a report missing the `algorithm` key (it falls back to the artifact directory name) can no longer appear in the table but vanish from the charts.
* Isolates `Algorithm comparison` from the dataset spinner — switching to it no longer waits for `Loading dataset manifest…`.
* The saved manifest can be switched off, instead of being silently forced whenever `artifacts/cuhkx_manifest.parquet` exists.

## Tests

From the repository root (standard library only, no dataset required):

```bash
python3 -m unittest discover -s tests -t .
```

## Run with Docker

```bash
docker build -t cuhkx-visualization .
mkdir -p artifacts
docker run --rm -p 127.0.0.1:8501:8501 \
  -v /path/to/small-model:/data:ro \
  -v "$(pwd)/artifacts:/app/artifacts" \
  cuhkx-visualization
```

On Linux append `--user "$(id -u):$(id -g)"`; the image runs as uid 10001 and a
bind mount keeps host ownership, so the background builder cannot otherwise
write into `artifacts`. Docker Desktop on macOS remaps ownership already.

Open <http://localhost:8501>. The container treats `/data` as the dataset
root and never copies the dataset into the image. The writable `/app/artifacts`
mount persists the generated manifest and progress state on the host.
