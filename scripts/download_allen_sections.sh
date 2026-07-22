#!/usr/bin/env bash
set -Eeuo pipefail

# Robust, resumable acquisition of Allen specimen 708424 histology JPEGs and
# Modified Brodmann SVG annotations.
#
# Behavior:
#   1. Reuses files already verified in metadata/image_files.tsv.
#   2. Byte-verifies unregistered existing JPEGs against Allen.
#   3. Quarantines files that do not match the requested Allen-direct response.
#   4. Redownloads only quarantined or missing JPEGs.
#   5. Validates counts, provenance, checksums, JPEG readability, dimensions,
#      and SVG annotation structure before reporting success.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Documents/git/ad-resilience}"
CONDA_ENV="${CONDA_ENV:-ad-resilience-xiv}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/raw/allen/specimen_708424}"
DOWNLOADER="${DOWNLOADER:-$PROJECT_ROOT/src/download_data/download_allen.py}"
MIN_FREE_GB="${MIN_FREE_GB:-10}"

LOG_DIR="$PROJECT_ROOT/results/logs"
STATUS_DIR="$PROJECT_ROOT/results/status"
LOCK_FILE="$STATUS_DIR/download_allen_sections.lock"

mkdir -p "$DATA_DIR" "$LOG_DIR" "$STATUS_DIR"

run_id="$(date +%Y%m%d_%H%M%S)_$$"
log_file="$LOG_DIR/download_allen_sections_${run_id}.log"
status_file="$STATUS_DIR/download_allen_sections_${run_id}.status"
quarantine_dir="$DATA_DIR/quarantine/$run_id"

error_code=""
error_line=""
error_command=""

on_error() {
    error_code="$1"
    error_line="$2"
    error_command="$3"

    printf '\nERROR\n'
    printf 'Exit code: %s\n' "$error_code"
    printf 'Line: %s\n' "$error_line"
    printf 'Command: %s\n' "$error_command"
}

on_exit() {
    local code=$?
    local finished host git_commit manifest manifest_rows
    local expected_nissl expected_ihc expected_svg
    local actual_nissl actual_ihc actual_svg bad_provenance
    local tmp_status

    trap - ERR EXIT
    set +e

    finished="$(date --iso-8601=seconds)"
    host="$(hostname)"
    git_commit="$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || printf 'unknown')"
    manifest="$DATA_DIR/metadata/image_files.tsv"

    expected_nissl="unknown"
    expected_ihc="unknown"
    expected_svg="unknown"
    actual_nissl="$(find "$DATA_DIR/nissl/images_orig" -maxdepth 1 -type f -name 'image_*.jpg' 2>/dev/null | wc -l)"
    actual_ihc="$(find "$DATA_DIR/ihc/images_orig" -maxdepth 1 -type f -name 'image_*.jpg' 2>/dev/null | wc -l)"
    actual_svg="$(find "$DATA_DIR/nissl/labels_orig" -maxdepth 1 -type f -name 'seg_*.svg' 2>/dev/null | wc -l)"
    manifest_rows="0"
    bad_provenance="unknown"

    if [[ -f "$DATA_DIR/secInfo.json" ]] && command -v jq >/dev/null 2>&1; then
        expected_nissl="$(jq -r '.nissl | length' "$DATA_DIR/secInfo.json" 2>/dev/null || printf 'unknown')"
        expected_ihc="$(jq -r '.ihc | length' "$DATA_DIR/secInfo.json" 2>/dev/null || printf 'unknown')"
        expected_svg="$(jq -r '.atlas_annotations | length' "$DATA_DIR/secInfo.json" 2>/dev/null || printf 'unknown')"
    fi

    if [[ -f "$manifest" ]]; then
        manifest_rows="$(awk 'END { print NR > 0 ? NR - 1 : 0 }' "$manifest")"
        bad_provenance="$(awk -F '\t' '
            NR > 1 && (
                $6 == "existing-unverified" ||
                $6 == "existing-mismatch" ||
                $7 != "allen-direct"
            ) { count++ }
            END { print count + 0 }
        ' "$manifest")"
    fi

    tmp_status="${status_file}.tmp.$$"
    {
        printf 'exit_code=%s\n' "$code"
        printf 'run_id=%s\n' "$run_id"
        printf 'finished=%s\n' "$finished"
        printf 'host=%s\n' "$host"
        printf 'git_commit=%s\n' "$git_commit"
        printf 'conda_environment=%s\n' "${CONDA_DEFAULT_ENV:-unknown}"
        printf 'data_dir=%s\n' "$DATA_DIR"
        printf 'log=%s\n' "$log_file"
        printf 'manifest=%s\n' "$manifest"
        printf 'quarantine_dir=%s\n' "$quarantine_dir"
        printf 'manifest_rows=%s\n' "$manifest_rows"
        printf 'bad_provenance_rows=%s\n' "$bad_provenance"
        printf 'expected_nissl=%s\n' "$expected_nissl"
        printf 'actual_nissl=%s\n' "$actual_nissl"
        printf 'expected_ihc=%s\n' "$expected_ihc"
        printf 'actual_ihc=%s\n' "$actual_ihc"
        printf 'expected_svg=%s\n' "$expected_svg"
        printf 'actual_svg=%s\n' "$actual_svg"
        if [[ -n "$error_code" ]]; then
            printf 'error_code=%s\n' "$error_code"
            printf 'error_line=%s\n' "$error_line"
            printf 'error_command=%q\n' "$error_command"
        fi
    } > "$tmp_status"
    mv -f "$tmp_status" "$status_file"

    printf '\nFinal exit code: %s\n' "$code"
    printf 'Status file: %s\n' "$status_file"
}

trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
trap on_exit EXIT

# Prevent concurrent writers from using the same output tree or provenance file.
command -v flock >/dev/null 2>&1 || {
    echo "Required command not found: flock" >&2
    exit 127
}
exec 9> "$LOCK_FILE"
if ! flock -n 9; then
    echo "Another Allen section download appears to be running." >&2
    echo "Lock file: $LOCK_FILE" >&2
    exit 75
fi

exec > >(tee -a "$log_file") 2>&1

echo "Started: $(date --iso-8601=seconds)"
echo "Run ID: $run_id"
echo "Host: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Output: $DATA_DIR"
echo "Log: $log_file"
echo

[[ -d "$PROJECT_ROOT" ]] || {
    echo "Project root does not exist: $PROJECT_ROOT" >&2
    exit 2
}
[[ -r "$CONDA_SH" ]] || {
    echo "Conda initialization script is unavailable: $CONDA_SH" >&2
    exit 2
}
[[ -f "$DOWNLOADER" ]] || {
    echo "Downloader script is unavailable: $DOWNLOADER" >&2
    exit 2
}
[[ "$MIN_FREE_GB" =~ ^[0-9]+$ ]] || {
    echo "MIN_FREE_GB must be a non-negative integer; got: $MIN_FREE_GB" >&2
    exit 2
}

# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    echo "Failed to activate conda environment: $CONDA_ENV" >&2
    exit 2
fi

for command_name in python jq grep sha256sum; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "Required command not found: $command_name" >&2
        exit 127
    }
done

python - <<'PY'
import numpy
import requests
import scipy
from PIL import Image

print("Python dependency check: OK")
PY

echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(python --version 2>&1)"
echo "Downloader SHA-256: $(sha256sum "$DOWNLOADER" | awk '{print $1}')"
echo "Git commit: $(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || printf 'unknown')"
echo

available_kb="$(df -Pk "$DATA_DIR" | awk 'NR == 2 { print $4 }')"
required_kb=$((MIN_FREE_GB * 1024 * 1024))
echo "Available disk space: $((available_kb / 1024 / 1024)) GiB"
echo "Required minimum: ${MIN_FREE_GB} GiB"
if (( available_kb < required_kb )); then
    echo "Insufficient free disk space under $DATA_DIR" >&2
    exit 28
fi

run_downloader() {
    python -u "$DOWNLOADER" \
        --data-dir "$DATA_DIR" \
        --specimen-id 708424 \
        --atlas-id 265297126 \
        --stains nissl ihc \
        --downsample 5 \
        --image-download-mode allen-direct \
        --skip-masks \
        --retries 8 \
        --backoff 1.5 \
        --log-level INFO \
        "$@"
}

echo
echo "Pass 1: resume downloads and verify unregistered existing JPEGs."
run_downloader --verify-existing

manifest="$DATA_DIR/metadata/image_files.tsv"
[[ -s "$manifest" ]] || {
    echo "Image provenance manifest is missing after pass 1: $manifest" >&2
    exit 3
}

bad_paths_file="$(mktemp "$STATUS_DIR/download_allen_bad_paths.XXXXXX")"
trap 'rm -f "$bad_paths_file"' RETURN
awk -F '\t' '
    NR > 1 && (
        $6 == "existing-unverified" ||
        $6 == "existing-mismatch" ||
        $7 != "allen-direct"
    ) { print $4 }
' "$manifest" > "$bad_paths_file"

if [[ -s "$bad_paths_file" ]]; then
    echo
    echo "Quarantining JPEGs that lack verified allen-direct provenance:"
    mkdir -p "$quarantine_dir"

    while IFS= read -r relative_path; do
        [[ -n "$relative_path" ]] || continue
        source_path="$DATA_DIR/$relative_path"
        destination_path="$quarantine_dir/$relative_path"

        if [[ -e "$source_path" ]]; then
            mkdir -p "$(dirname "$destination_path")"
            echo "  $relative_path"
            mv -- "$source_path" "$destination_path"
        else
            echo "  already absent: $relative_path"
        fi
    done < "$bad_paths_file"

    echo
    echo "Pass 2: redownload quarantined JPEGs directly from Allen."
    run_downloader
else
    echo "No JPEGs required quarantine."
fi
rm -f "$bad_paths_file"
trap - RETURN

echo
echo "Validating completed acquisition..."

secinfo="$DATA_DIR/secInfo.json"
sections_tsv="$DATA_DIR/metadata/sections.tsv"
atlas_tsv="$DATA_DIR/metadata/atlas_annotations.tsv"

for required_file in "$secinfo" "$manifest" "$sections_tsv" "$atlas_tsv"; do
    [[ -s "$required_file" ]] || {
        echo "Required metadata file is missing or empty: $required_file" >&2
        exit 3
    }
done

expected_nissl="$(jq -r '.nissl | length' "$secinfo")"
expected_ihc="$(jq -r '.ihc | length' "$secinfo")"
expected_svg="$(jq -r '.atlas_annotations | length' "$secinfo")"
expected_total=$((expected_nissl + expected_ihc))

actual_nissl="$(find "$DATA_DIR/nissl/images_orig" -maxdepth 1 -type f -name 'image_*.jpg' | wc -l)"
actual_ihc="$(find "$DATA_DIR/ihc/images_orig" -maxdepth 1 -type f -name 'image_*.jpg' | wc -l)"
actual_svg="$(find "$DATA_DIR/nissl/labels_orig" -maxdepth 1 -type f -name 'seg_*.svg' | wc -l)"
manifest_rows="$(awk 'END { print NR - 1 }' "$manifest")"

printf 'Nissl JPEGs: %s/%s\n' "$actual_nissl" "$expected_nissl"
printf 'IHC/PV JPEGs: %s/%s\n' "$actual_ihc" "$expected_ihc"
printf 'Atlas SVGs: %s/%s\n' "$actual_svg" "$expected_svg"
printf 'Image provenance rows: %s/%s\n' "$manifest_rows" "$expected_total"

[[ "$actual_nissl" -eq "$expected_nissl" ]] || {
    echo "Nissl image count is incomplete" >&2
    exit 4
}
[[ "$actual_ihc" -eq "$expected_ihc" ]] || {
    echo "IHC/PV image count is incomplete" >&2
    exit 4
}
[[ "$actual_svg" -eq "$expected_svg" ]] || {
    echo "Atlas SVG count is incomplete" >&2
    exit 4
}
[[ "$manifest_rows" -eq "$expected_total" ]] || {
    echo "Image provenance manifest count is incomplete" >&2
    exit 4
}

python - "$DATA_DIR" <<'PY'
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

root = Path(sys.argv[1]).resolve()
manifest_path = root / "metadata" / "image_files.tsv"
allowed_statuses = {
    "downloaded",
    "verified-existing",
    "manifest-verified-existing",
}
errors: list[str] = []

with manifest_path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

for row in rows:
    relative_path = row["relative_output_path"]
    path = root / relative_path

    if row["status"] not in allowed_statuses:
        errors.append(f"{relative_path}: unsupported status {row['status']!r}")
    if row["verified_mode"] != "allen-direct":
        errors.append(
            f"{relative_path}: verified_mode is {row['verified_mode']!r}, "
            "expected 'allen-direct'"
        )
    if not path.is_file():
        errors.append(f"{relative_path}: file is missing")
        continue

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != row["sha256"]:
        errors.append(f"{relative_path}: SHA-256 does not match manifest")

    try:
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        errors.append(f"{relative_path}: unreadable JPEG ({exc})")
        continue

    if image_format != "JPEG":
        errors.append(f"{relative_path}: format is {image_format!r}, expected JPEG")
    if width != int(row["width_px"]) or height != int(row["height_px"]):
        errors.append(
            f"{relative_path}: dimensions {width}x{height} do not match "
            f"manifest {row['width_px']}x{row['height_px']}"
        )

if errors:
    print("Image provenance or integrity validation failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)

print(f"Validated {len(rows)} JPEG files against provenance checksums and headers.")
PY

invalid_svgs="$(
    find "$DATA_DIR/nissl/labels_orig" -maxdepth 1 -type f -name 'seg_*.svg' -print0 \
        | while IFS= read -r -d '' svg; do
            grep -q '<svg' "$svg" && grep -q 'structure_id="[0-9]' "$svg" \
                || printf '%s\n' "$svg"
          done
)"
if [[ -n "$invalid_svgs" ]]; then
    echo "SVG files missing an SVG root or structure IDs:" >&2
    printf '%s\n' "$invalid_svgs" >&2
    exit 6
fi

echo "Validated $actual_svg SVG annotation files."
echo
echo "Allen section and annotation download completed and validated."
du -sh "$DATA_DIR"
