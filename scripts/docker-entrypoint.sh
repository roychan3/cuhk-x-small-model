#!/bin/bash
set -euo pipefail

SAMPLE_ROOT=/app/artifacts/sample_dataset
STAGING=/app/artifacts/.sample_dataset.building

# Bind mounts may replace these with host directories that do not exist yet.
mkdir -p /app/artifacts /app/outputs

# Prepare the small fixture so "Workflow > Sample dataset" works without a
# manual step. The full dataset is expected read-only at /data.
if [ -d "$SAMPLE_ROOT" ]; then
  echo "[entrypoint] Sample dataset already present at $SAMPLE_ROOT"
elif [ -d /data/Training ] && [ -d /data/Testing ]; then
  # prepare_sample_dataset.py builds into a staging directory and refuses to
  # start if one is left over, which a killed container can do. Clearing it
  # here keeps a hard stop from disabling the sample for every later start.
  rm -rf "$STAGING"
  echo "[entrypoint] Preparing sample dataset from /data ..."
  if python scripts/prepare_sample_dataset.py --source /data; then
    echo "[entrypoint] Sample dataset ready at $SAMPLE_ROOT"
  else
    echo "[entrypoint] WARNING: sample preparation failed; continuing without it." >&2
    echo "[entrypoint] Mount artifacts writable, and on Linux pass --user \"\$(id -u):\$(id -g)\"." >&2
  fi
else
  echo "[entrypoint] No /data mount with Training/ and Testing/ — skipping the sample dataset."
  echo "[entrypoint] Mount the full dataset with -v /path/to/small-model:/data:ro to enable both."
fi

exec "$@"
