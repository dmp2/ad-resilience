#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT="/cis/home/dpadova/Documents/git/ad-resilience"

cd "$PROJECT"

mkdir -p results/logs

LOG="$PROJECT/results/logs/allen_annotation_driven_full_groups_31_265297118.log"

# Capture launcher failures as well as Python output.
exec > >(tee -a "$LOG") 2>&1

trap 'status=$?; echo "EXIT STATUS: $status at $(date -Is)"' EXIT

# pyjnius_activate.sh expects CLASSPATH to exist.
export CLASSPATH="${CLASSPATH:-}"

source /cis/home/dpadova/miniconda3/etc/profile.d/conda.sh

conda activate wsi-pipeline

DATASET="data/derivatives/allen/specimen_708424/histology_symmetric_nissl_native_200um_section_aligned"

ANNOTATIONS="data/derivatives/allen/specimen_708424/annotations_symmetric_nissl_native_200um_section_aligned"

REGISTRATION="results/allen/specimen_708424/emlddmm/native-200um-clean/HIST_NISSL_SYMMETRIC_SECTION_ALIGNED_to_MRI_7T_WHOLE"

OUTPUT="data/derivatives/allen/specimen_708424/annotations_dense_annotation_driven_200um_groups_31_265297118"

PAIR_CONFIG="configs/allen_dense_pairwise.json"

WSI_REPOSITORY="/cis/home/dpadova/Documents/git/wsi-tissue-pipeline"

echo "================================================"
echo "Allen annotation-driven full-stack densification"
echo "Started: $(date -Is)"
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Groups: 31, 265297118"
echo "Driver: annotation"
echo "Output: $OUTPUT"
echo "================================================"

test -f "$PAIR_CONFIG"
test -d "$DATASET"
test -d "$ANNOTATIONS"
test -d "$REGISTRATION"
test -d "$WSI_REPOSITORY"

# Prevent accidental execution with an older production CLI.
PYTHONPATH="$PWD/src" \
python -m preprocess.densify_allen_annotations --help \
    | grep -q -- '--driver'

PYTHONPATH="$PWD/src" \
python -m preprocess.densify_allen_annotations --help \
    | grep -q -- '--graphic-groups'

/usr/bin/time -v \
env \
    PYTHONPATH="$PWD/src" \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
python -m preprocess.densify_allen_annotations \
    --dataset "$DATASET" \
    --annotations "$ANNOTATIONS" \
    --registration-run "$REGISTRATION" \
    --output "$OUTPUT" \
    --driver annotation \
    --graphic-groups 31 265297118 \
    --output-format tiff \
    --tiff-compression deflate \
    --pair-config "$PAIR_CONFIG" \
    --wsi-repository "$WSI_REPOSITORY" \
    --device auto

echo "Completed successfully: $(date -Is)"