#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$HOME/Documents/git/ad-resilience"
cd "$PROJECT_ROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ad-resilience-xiv

mkdir -p \
    data/raw/openneuro \
    results/logs \
    results/status

run_id="$(date +%Y%m%d_%H%M%S)"
log_file="results/logs/download_openneuro_${run_id}.log"
status_file="results/status/download_openneuro_${run_id}.status"

on_error() {
    code=$?
    echo
    echo "ERROR"
    echo "Exit code: $code"
    echo "Line: $LINENO"
    echo "Command: $BASH_COMMAND"
}

on_exit() {
    code=$?
    {
        echo "exit_code=$code"
        echo "finished=$(date --iso-8601=seconds)"
        echo "host=$(hostname)"
        echo "log=$log_file"
    } > "$status_file"

    echo
    echo "Final exit code: $code"
    echo "Status file: $status_file"
}

trap on_error ERR
trap on_exit EXIT

exec > >(tee -a "$log_file") 2>&1

echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Dataset: OpenNeuro ds003590 version 1.0.2"
echo

command -v datalad
command -v git-annex
datalad --version
git-annex version | head

python -u src/download_data/download_openneuro_ds003590.py \
    --data-dir data/raw/openneuro/ds003590 \
    --version 1.0.2 \
    --content revised

echo
echo "OpenNeuro download completed."
du -sh data/raw/openneuro/ds003590
