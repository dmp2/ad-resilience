#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$HOME/Documents/git/ad-resilience"
DOWNLOAD_DIR="$PROJECT_ROOT/data/raw/allen/brainspan_34yr"
LOG_DIR="$PROJECT_ROOT/results/logs"
MANIFEST_DIR="$PROJECT_ROOT/results/manifests/brainspan_34yr"

mkdir -p "$DOWNLOAD_DIR" "$LOG_DIR" "$MANIFEST_DIR"
cd "$DOWNLOAD_DIR"

run_date="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/download_brainspan_ding_${run_date}.log"

exec > >(tee -a "$log_file") 2>&1

echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Download directory: $DOWNLOAD_DIR"
echo

download_archive() {
    local filename="$1"
    local url="$2"

    echo "============================================================"
    echo "File: $filename"
    echo "URL:  $url"
    echo "Time: $(date --iso-8601=seconds)"

    # A completed, readable tar archive is not downloaded again.
    if [[ -f "$filename" ]] && tar -tzf "$filename" >/dev/null 2>&1; then
        echo "Archive already exists and passes tar integrity check."
    else
        wget \
            --continue \
            --tries=0 \
            --timeout=60 \
            --read-timeout=60 \
            --retry-connrefused \
            --waitretry=5 \
            --progress=dot:giga \
            -O "$filename" \
            "$url"
    fi

    echo "Checking archive integrity..."
    gzip -t "$filename"
    tar -tzf "$filename" >/dev/null

    echo "Saving archive manifest..."
    tar -tzf "$filename" \
        > "$MANIFEST_DIR/${filename%.tgz}_contents.txt"

    sha256sum "$filename" \
        > "$MANIFEST_DIR/${filename}.sha256"

    ls -lh "$filename"
    echo
}

# Highest-priority anatomical reference.
download_archive \
    "allen_34yr_7T_structural_mri.tgz" \
    "https://www.brainspan.org/api/v2/well_known_file_download/157926961"

# Highest-resolution diffusion archive.
download_archive \
    "allen_34yr_3T_dwi_900um.tgz" \
    "https://www.brainspan.org/api/v2/well_known_file_download/157926979"

# Lower-resolution companion diffusion archive.
download_archive \
    "allen_34yr_3T_dwi_1200um.tgz" \
    "https://www.brainspan.org/api/v2/well_known_file_download/157926982"

# Small optional structural comparison.
download_archive \
    "allen_34yr_3T_structural_mri.tgz" \
    "https://www.brainspan.org/api/v2/well_known_file_download/157926656"

echo "============================================================"
echo "All BrainSpan imaging downloads completed."
echo "Completed: $(date --iso-8601=seconds)"
echo
du -sh "$DOWNLOAD_DIR"
df -h "$DOWNLOAD_DIR"
