# Visualization module

This module contains all code and dependencies for exploring the CUHK-X
dataset. It is independent from future training and evaluation modules.

## Contents

- `app.py`: Streamlit application, interactive plots, and automatic clip playback controls.
- `playback.py`: synchronized timeline and playback timing helpers.
- `dataset.py`: standard-library dataset discovery, indexing, quality checks,
  archive access, and manifest generation.
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

Build a reusable manifest:

```bash
python -m visualization.build_manifest \
  --dataset-root /path/to/small-model \
  --output artifacts/cuhkx_manifest.parquet
```

## Tests

From the repository root (standard library only, no dataset required):

```bash
python3 -m unittest discover -s tests -t .
```

## Run with Docker

```bash
docker build -t cuhkx-visualization .
docker run --rm -p 127.0.0.1:8501:8501 \
  -v /path/to/small-model:/data:ro \
  cuhkx-visualization
```

Open <http://localhost:8501>. The container treats `/data` as the dataset
root and never copies the dataset into the image.
