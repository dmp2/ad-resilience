#!/usr/bin/env bash
set -Eeuo pipefail

# Resumable raw acquisition for Allen specimen 708424. The Python downloader
# owns verification/quarantine decisions; this wrapper owns process locking,
# logs, status publication, preflight checks, and final read-only validation.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd -P)}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/raw/allen/specimen_708424}"
DOWNLOADER="${DOWNLOADER:-$PROJECT_ROOT/src/download_data/download_allen.py}"
MIN_FREE_GB="${MIN_FREE_GB:-10}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/results/logs}"
STATUS_DIR="${STATUS_DIR:-$PROJECT_ROOT/results/status}"
LOCK_FILE="${LOCK_FILE:-$STATUS_DIR/download_allen_sections.lock}"
PREFLIGHT_ONLY=0

if [[ "${1:-}" == "--preflight-only" ]]; then
    PREFLIGHT_ONLY=1
    shift
fi
if (($#)); then
    printf 'Unknown wrapper argument: %s\n' "$1" >&2
    exit 2
fi

for value_name in MIN_FREE_GB; do
    [[ "${!value_name}" =~ ^[0-9]+$ ]] || {
        printf '%s must be a non-negative integer; got %s\n' "$value_name" "${!value_name}" >&2
        exit 2
    }
done
[[ -d "$PROJECT_ROOT" ]] || { printf 'Missing project root: %s\n' "$PROJECT_ROOT" >&2; exit 2; }
[[ -f "$DOWNLOADER" ]] || { printf 'Missing downloader: %s\n' "$DOWNLOADER" >&2; exit 2; }
command -v flock >/dev/null 2>&1 || { echo 'Required command not found: flock' >&2; exit 127; }
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_COMMAND=("$PYTHON_BIN")
elif python -c 'import numpy,requests,scipy,PIL' >/dev/null 2>&1; then
    PYTHON_COMMAND=(python)
elif command -v conda >/dev/null 2>&1; then
    PYTHON_COMMAND=(conda run -n "${CONDA_ENV:-ad-resilience-xiv}" python)
else
    echo 'No Python with the acquisition dependencies was found; set PYTHON_BIN.' >&2
    exit 127
fi
command -v "${PYTHON_COMMAND[0]}" >/dev/null 2>&1 || { printf 'Python launcher is unavailable: %s\n' "${PYTHON_COMMAND[0]}" >&2; exit 127; }

mkdir -p "$LOG_DIR" "$STATUS_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    printf 'Another Allen acquisition holds lock: %s\n' "$LOCK_FILE" >&2
    exit 75
fi

run_id="$(date -u +%Y%m%dT%H%M%SZ)_$$"
log_file="$LOG_DIR/download_allen_sections_${run_id}.log"
status_file="$STATUS_DIR/download_allen_sections_${run_id}.status"
inventory_file="$STATUS_DIR/allen_inventory_observed_${run_id}.json"
validation_file="$STATUS_DIR/allen_validation_${run_id}.json"
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

write_status() {
    local code="$1" temporary dataset_usage="unknown" validation_status="not-run"
    temporary="${status_file}.tmp.$$"
    if [[ -d "$DATA_DIR" ]]; then
        dataset_usage="$(du -sh "$DATA_DIR" 2>/dev/null | awk '{print $1}' || printf unknown)"
    fi
    if [[ -s "$validation_file" ]]; then
        validation_status="$("${PYTHON_COMMAND[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$validation_file" 2>/dev/null || printf unreadable)"
    fi
    {
        printf 'run_id=%s\n' "$run_id"
        printf 'started_utc=%s\n' "$started"
        printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'exit_code=%s\n' "$code"
        printf 'validation_status=%s\n' "$validation_status"
        printf 'project_root=%s\n' "$PROJECT_ROOT"
        printf 'data_dir=%s\n' "$DATA_DIR"
        printf 'dataset_disk_usage=%s\n' "$dataset_usage"
        printf 'log=%s\n' "$log_file"
        printf 'inventory_report=%s\n' "$inventory_file"
        printf 'validation_report=%s\n' "$validation_file"
    } >"$temporary"
    mv -f "$temporary" "$status_file"
}

on_exit() {
    local code=$?
    trap - EXIT
    set +e
    write_status "$code"
    printf '\nExit code: %s\nStatus: %s\n' "$code" "$status_file"
    exit "$code"
}
trap on_exit EXIT

exec > >(tee -a "$log_file") 2>&1
printf 'Started: %s\nProject root: %s\nData: %s\nLog: %s\n' "$started" "$PROJECT_ROOT" "$DATA_DIR" "$log_file"

"${PYTHON_COMMAND[@]}" - <<'PY'
import numpy
import requests
import scipy
from PIL import Image
print("Python dependency check: OK")
PY
"${PYTHON_COMMAND[@]}" -m py_compile "$DOWNLOADER"

available_kb="$(df -Pk "$PROJECT_ROOT" | awk 'NR == 2 {print $4}')"
required_kb=$((MIN_FREE_GB * 1024 * 1024))
printf 'Available disk: %s GiB; required: %s GiB\n' "$((available_kb / 1024 / 1024))" "$MIN_FREE_GB"
if ((available_kb < required_kb)); then
    echo 'Insufficient free disk space.' >&2
    exit 28
fi

if ((PREFLIGHT_ONLY)); then
    echo 'Preflight: PASS'
    exit 0
fi

echo 'Acquiring accepted Allen API series, 106 multi-group SVGs, and graph-16 ontology.'
set +e
"${PYTHON_COMMAND[@]}" -u "$DOWNLOADER" \
    --data-dir "$DATA_DIR" \
    --image-download-mode allen-direct \
    --downsample 5 \
    --skip-masks \
    --inventory-json "$inventory_file" \
    --retries 8 \
    --retry-backoff 1.5 \
    --log-level INFO
acquisition_code=$?
set -e
if ((acquisition_code == 3)); then
    echo 'API_INVENTORY_CHANGED: no inventory was accepted and no acquisition mutation was performed.' >&2
    printf 'Review %s, then rerun the downloader manually with the exact --accept-api-inventory-sha256 digest.\n' "$inventory_file" >&2
    exit 3
fi
if ((acquisition_code != 0)); then
    exit "$acquisition_code"
fi

echo 'Running offline read-only validation.'
"${PYTHON_COMMAND[@]}" -u "$DOWNLOADER" \
    --data-dir "$DATA_DIR" \
    --validate-only \
    --validation-json "$validation_file"

echo 'Disk usage:'
du -sh "$DATA_DIR"
echo 'Allen raw acquisition and validation: PASS'
