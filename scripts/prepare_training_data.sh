#!/usr/bin/env bash
#
# Merge and extract the multi-volume CUHK-X training archive.
#
# Distributed copies name the volumes HAR-00N.zNN, but `zip` requires them to
# share the basename of the final volume (HAR.zNN + HAR.zip). This script
# symlinks them into place, merges the volumes, and extracts the result.
#
# Safe to re-run: each stage is skipped when its output already exists, and the
# original volume files are never renamed or removed.
#
# Usage:
#   bash scripts/prepare_training_data.sh [dataset-root]
#
# The dataset root defaults to $CUHKX_DATASET_ROOT, then to
# ~/AAI/opt-scratch/small-model.

set -euo pipefail

DATASET_ROOT="${1:-${CUHKX_DATASET_ROOT:-$HOME/AAI/opt-scratch/small-model}}"
DATA_DIR="$DATASET_ROOT/Training/data"
MERGED="$DATA_DIR/HAR_full.zip"
EXTRACTED="$DATA_DIR/HAR"

log() { printf '\n==> %s\n' "$1"; }

[ -d "$DATA_DIR" ] || { echo "No such directory: $DATA_DIR" >&2; exit 1; }
cd "$DATA_DIR"

if [ -d "$EXTRACTED/data" ]; then
    log "Already extracted: $EXTRACTED/data"
    du -sh "$EXTRACTED"
    exit 0
fi

log "Linking volume names"
shopt -s nullglob
volumes=(HAR-*.z[0-9][0-9])
shopt -u nullglob
if [ ${#volumes[@]} -eq 0 ] && [ ! -f "$MERGED" ]; then
    echo "Found no HAR-*.zNN volumes in $DATA_DIR" >&2
    exit 1
fi
for volume in "${volumes[@]}"; do
    suffix="${volume##*.}"
    ln -sfn "$volume" "HAR.$suffix"
    printf '    HAR.%s -> %s\n' "$suffix" "$volume"
done

[ -f HAR.zip ] || { echo "Missing final volume HAR.zip" >&2; exit 1; }

# Space check: the merge needs roughly the combined size of every volume, and
# extraction needs roughly twice that again.
needed_kb=0
for volume in "${volumes[@]}" HAR.zip; do
    needed_kb=$((needed_kb + $(du -k "$volume" | cut -f1)))
done
available_kb=$(df -k . | awk 'NR==2 {print $4}')
required_kb=$((needed_kb * 3))
printf '    volumes: %s GiB, need ~%s GiB, available %s GiB\n' \
    "$((needed_kb / 1048576))" "$((required_kb / 1048576))" "$((available_kb / 1048576))"
if [ "$available_kb" -lt "$required_kb" ]; then
    echo "Not enough free space for the merge plus extraction." >&2
    exit 1
fi

if [ -f "$MERGED" ]; then
    log "Merged archive already present: $MERGED"
else
    log "Merging volumes into HAR_full.zip (slow)"
    zip -s 0 HAR.zip --out "$MERGED"
fi

log "Extracting HAR_full.zip (slow)"
unzip -q "$MERGED"

[ -d "$EXTRACTED/data" ] || { echo "Extraction finished but $EXTRACTED/data is missing" >&2; exit 1; }

log "Done"
du -sh "$EXTRACTED"
printf '\nReclaim space once verified:\n    rm %s\n' "$MERGED"
