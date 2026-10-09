#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$HOME/Documents/git/ad-resilience"
DOWNLOAD_DIR="$PROJECT_ROOT/data/raw/allen/ahrav2/2025"
LOG_DIR="$PROJECT_ROOT/results/logs"
MANIFEST_DIR="$PROJECT_ROOT/results/manifests/ahrav2_2025"

mkdir -p "$DOWNLOAD_DIR" "$LOG_DIR" "$MANIFEST_DIR"
cd "$DOWNLOAD_DIR"

run_date="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/download_allen_ahrav2_${run_date}.log"

exec > >(tee -a "$log_file") 2>&1

echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Download directory: $DOWNLOAD_DIR"
echo

download_nii_gz() {
    local filename="$1"
    local url="$2"

    echo "============================================================"
    echo "File: $filename"
    echo "URL:  $url"
    echo "Time: $(date --iso-8601=seconds)"

    # A completed, readable gzip-compressed NIfTI is not downloaded again.
    if [[ -f "$filename" ]] && gzip -t "$filename" >/dev/null 2>&1; then
        echo "File already exists and passes gzip integrity check."
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

    echo "Checking gzip integrity..."
    gzip -t "$filename"

    echo "Saving checksum..."
    sha256sum "$filename" \
        > "$MANIFEST_DIR/${filename}.sha256"

    ls -lh "$filename"
    echo
}

# ---------------------------------------------------------------------
# Allen Human Reference Atlas 3D, version 2 (AHRAv2)
#
# Whole-brain atlas in symmetric MNI152 ICBM2009b space.
# Resolution: 500 um isotropic.
# Release: 2025
# ---------------------------------------------------------------------

# Anatomical MRI template.
# This is the image we will register to our single-subject 7T MRI.
download_nii_gz \
    "ahrav2_icbm2009b_symmetric_template_500um.nii.gz" \
    "https://allen-atlas-assets.s3.us-west-2.amazonaws.com/templates/hmba-adult-human-icbm2009b-template/2025/template_500.nii.gz"

# Whole-brain gyral parcellation.
download_nii_gz \
    "ahrav2_gyral_annotations_500um.nii.gz" \
    "https://allen-atlas-assets.s3.us-west-2.amazonaws.com/annotation-sets/hmba-adult-human-ahrav2gyral-annotation/2025/annotations_compressed_500.nii.gz"

# Whole-brain modified Brodmann parcellation.
download_nii_gz \
    "ahrav2_brodmann_annotations_500um.nii.gz" \
    "https://allen-atlas-assets.s3.us-west-2.amazonaws.com/annotation-sets/hmba-adult-human-ahrav2brodmann-annotation/2025/annotations_compressed_500.nii.gz"


# Save a small provenance record alongside the checksums.
cat > "$MANIFEST_DIR/sources.tsv" <<'EOF'
local_filename	source_url
ahrav2_icbm2009b_symmetric_template_500um.nii.gz	https://allen-atlas-assets.s3.us-west-2.amazonaws.com/templates/hmba-adult-human-icbm2009b-template/2025/template_500.nii.gz
ahrav2_gyral_annotations_500um.nii.gz	https://allen-atlas-assets.s3.us-west-2.amazonaws.com/annotation-sets/hmba-adult-human-ahrav2gyral-annotation/2025/annotations_compressed_500.nii.gz
ahrav2_brodmann_annotations_500um.nii.gz	https://allen-atlas-assets.s3.us-west-2.amazonaws.com/annotation-sets/hmba-adult-human-ahrav2brodmann-annotation/2025/annotations_compressed_500.nii.gz
EOF


echo "============================================================"
echo "Allen AHRAv2 downloads completed."
echo "Completed: $(date --iso-8601=seconds)"
echo
echo "Files:"
ls -lh "$DOWNLOAD_DIR"
echo
echo "Checksums / provenance:"
ls -lh "$MANIFEST_DIR"
echo
du -sh "$DOWNLOAD_DIR"
df -h "$DOWNLOAD_DIR"