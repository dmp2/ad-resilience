#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
left_root="data/derivatives/allen/specimen_708424/emlddmm_7t"
symmetric_source="data/derivatives/allen/specimen_708424/histology_symmetric"
symmetric_prepared="data/derivatives/allen/specimen_708424/emlddmm_7t_symmetric"
mri_root="data/derivatives/allen/specimen_708424/mri_7t_whole"
contract="configs/emlddmm/allen_708424_hist_symmetric_to_mri7t_t1.json"

case "${mode}" in
  prepare-left)
    PYTHONPATH=src python -m preprocess.prepare_allen_emlddmm_inputs \
      --data-dir data/raw/allen/specimen_708424 \
      --output-dir "${left_root}" \
      --series all
    ;;
  prepare-mri)
    PYTHONPATH=src python -m preprocess.prepare_allen_7t_mri \
      --output-dir "${mri_root}"
    ;;
  verify-mri)
    PYTHONPATH=src python -m preprocess.prepare_allen_7t_mri \
      --output-dir "${mri_root}" \
      --verify-existing
    ;;
  build-symmetric)
    PYTHONPATH=src python -m preprocess.build_allen_symmetric_histology \
      --left-dataset "${left_root}" \
      --output-dir "${symmetric_source}"
    ;;
  prepare-symmetric)
    PYTHONPATH=src python -m preprocess.prepare_allen_emlddmm_inputs \
      --source-dataset "${symmetric_source}" \
      --output-dir "${symmetric_prepared}" \
      --preserve-source-grid
    ;;
  verify-symmetric)
    PYTHONPATH=src python -m preprocess.prepare_allen_emlddmm_inputs \
      --source-dataset "${symmetric_source}" \
      --output-dir "${symmetric_prepared}" \
      --preserve-source-grid \
      --verify-existing
    ;;
  validate-contract)
    PYTHONPATH=src python -c \
      "from pathlib import Path; from preprocess.run_allen_emlddmm import validate_registration_contract; validate_registration_contract(Path('${contract}'), require_prepared_histology=True)"
    ;;
  register|pilot-all|pilot-nissl|pilot-pv|full-all)
    echo "Registration execution is intentionally disabled until a reviewed cross-space initialization is recorded in ${contract}." >&2
    exit 2
    ;;
  *)
    echo "Usage: $0 {prepare-left|prepare-mri|verify-mri|build-symmetric|prepare-symmetric|verify-symmetric|validate-contract}" >&2
    exit 2
    ;;
esac
