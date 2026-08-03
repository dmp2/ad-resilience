#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

stage="${1:-}"
case "${stage}" in
  atlas-free|registration|postprocess) ;;
  *) echo "Usage: $0 {atlas-free|registration|postprocess}" >&2; exit 2 ;;
esac
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
python_bin="/cis/home/dpadova/miniconda3/envs/pylddmm_env3.10/bin/python"
tmp_root="/cis/home/dpadova/.cache/ad-resilience/tmp"
output_root="${project_root}/results/allen/specimen_708424/emlddmm/full-coarse/HIST_NISSL_to_MRI_7T_WHOLE_eA1e6"
checkpoint_dir="${output_root}/checkpoints"; log_dir="${output_root}/logs"
lock_path="${output_root%/*}/.HIST_NISSL_to_MRI_7T_WHOLE_eA1e6.lock"
rss_limit_kib=$((64 * 1024 * 1024)); run_tmp=""; stage_pid=""; stage_pgid=""; watchdog_reason=""
cleanup_tmp() {
  [[ -n "${run_tmp}" && -e "${run_tmp}" ]] || return 0
  local resolved root; resolved="$(realpath -e "${run_tmp}")"; root="$(realpath -e "${tmp_root}")"
  case "${resolved}" in "${root}"/*) rm -rf -- "${resolved}";; *) echo "Refusing unsafe cleanup: ${resolved}" >&2; return 1;; esac
}
terminate_stage() {
  [[ -n "${stage_pgid}" ]] || return 0
  kill -TERM -- "-${stage_pgid}" 2>/dev/null || true
  for _ in {1..10}; do kill -0 "${stage_pid}" 2>/dev/null || return 0; sleep 1; done
  kill -KILL -- "-${stage_pgid}" 2>/dev/null || true
}
on_signal() { local code="$1" name="$2"; watchdog_reason="signal ${name}"; terminate_stage; exit "${code}"; }
on_exit() { local code="$?"; trap - EXIT; cleanup_tmp || true; exit "${code}"; }
trap 'on_signal 130 INT' INT; trap 'on_signal 143 TERM' TERM; trap on_exit EXIT
[[ -x "${python_bin}" ]] || { echo "Missing Python: ${python_bin}" >&2; exit 1; }
mkdir -p "${tmp_root}"; chmod 700 "${tmp_root}"
home_resolved="$(realpath -e "${HOME}")"; tmp_resolved="$(realpath -e "${tmp_root}")"
case "${tmp_resolved}" in "${home_resolved}"/*) ;; *) echo "TMP root is outside HOME" >&2; exit 1;; esac
[[ "$(stat -c '%u' "${tmp_resolved}")" == "$(id -u)" && "$(stat -c '%a' "${tmp_resolved}")" == 700 ]] || { echo "TMP root ownership/mode invalid" >&2; exit 1; }
mkdir -p "${output_root%/*}"; exec 9>"${lock_path}"; flock -n 9 || { echo "Another full-coarse stage holds the lock" >&2; exit 1; }
mkdir -p "${checkpoint_dir}" "${log_dir}"
case "${stage}" in
 registration) jq -e '.status=="complete" and .stage=="atlas-free"' "${checkpoint_dir}/atlas-free.json" >/dev/null || { echo "Registration requires completed atlas-free" >&2; exit 1; };;
 postprocess) jq -e '.status=="complete"' "${checkpoint_dir}/atlas-free.json" >/dev/null && jq -e '.status=="complete"' "${checkpoint_dir}/registration.json" >/dev/null || { echo "Postprocess prerequisites incomplete" >&2; exit 1; };;
esac
if [[ -f "${checkpoint_dir}/${stage}.json" ]] && jq -e '.status=="complete" or .status=="review_required"' "${checkpoint_dir}/${stage}.json" >/dev/null; then echo "Stage ${stage} already complete" >&2; exit 1; fi
run_tmp="$(mktemp -d --tmpdir="${tmp_resolved}" "allen-nissl-${stage}.XXXXXXXX")"; run_tmp="$(realpath -e "${run_tmp}")"
case "${run_tmp}" in "${tmp_resolved}"/*) ;; *) echo "Run TMP escaped approved root" >&2; exit 1;; esac
rss_peak_file="${run_tmp}/process_group_peak_rss_kib"; printf '0\n' > "${rss_peak_file}"
export TMPDIR="${tmp_resolved}" TMP="${tmp_resolved}" TEMP="${tmp_resolved}" PYTHONPATH="${project_root}/src" MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 BLIS_NUM_THREADS=8 TORCH_NUM_THREADS=8
export EMLDDMM_OUTPUT_ROOT="${output_root}" EMLDDMM_RUN_TMP="${run_tmp}" EMLDDMM_RSS_PEAK_FILE="${rss_peak_file}" EMLDDMM_RSS_LIMIT_KIB="${rss_limit_kib}"
log_path="${log_dir}/${stage}.log"; echo "Starting ${stage}; log=${log_path}; tmp=${run_tmp}"
setsid "${python_bin}" -m preprocess.run_allen_emlddmm_full_coarse_nissl "${stage}" > >(tee -a "${log_path}") 2>&1 &
stage_pid=$!; stage_pgid="${stage_pid}"; peak_kib=0
while kill -0 "${stage_pid}" 2>/dev/null; do
 current_kib="$(ps -eo pgid=,rss= | awk -v pg="${stage_pgid}" '$1==pg {sum+=$2} END {print sum+0}')"
 if (( current_kib > peak_kib )); then peak_kib="${current_kib}"; printf '%s\n' "${peak_kib}" > "${rss_peak_file}"; fi
 if (( current_kib >= rss_limit_kib )); then watchdog_reason="stage process-group RSS reached ${current_kib} KiB (limit ${rss_limit_kib} KiB)"; echo "RSS watchdog: ${watchdog_reason}" >&2; terminate_stage; break; fi
 sleep 2
done
set +e; wait "${stage_pid}"; stage_status=$?; set -e; stage_pid=""; stage_pgid=""
if [[ -n "${watchdog_reason}" ]]; then
 jq -n --arg stage "${stage}" --arg reason "${watchdog_reason}" --arg time "$(date --iso-8601=seconds)" --argjson peak "${peak_kib}" --argjson limit "${rss_limit_kib}" '{stage:$stage,status:"failed",failure_reason:$reason,failure_time:$time,peak_process_group_rss_kib:$peak,rss_limit_kib:$limit,production_refinement_launched:false}' > "${checkpoint_dir}/${stage}.json.tmp"
 mv "${checkpoint_dir}/${stage}.json.tmp" "${checkpoint_dir}/${stage}.json"; exit 137
fi
if (( stage_status != 0 )); then
  now="$(date --iso-8601=seconds)"
  existing="${checkpoint_dir}/${stage}.json"
  if [[ -f "${existing}" ]]; then
    jq --arg stage "${stage}" --arg time "${now}" --argjson code "${stage_status}" --argjson peak "${peak_kib}" '. + {stage:$stage,status:"failed",exit_code:$code,failure_time:$time,peak_process_group_rss_kib:$peak,production_refinement_launched:false}' "${existing}" > "${existing}.tmp"
  else
    jq -n --arg stage "${stage}" --arg time "${now}" --argjson code "${stage_status}" --argjson peak "${peak_kib}" '{stage:$stage,status:"failed",exit_code:$code,failure_time:$time,peak_process_group_rss_kib:$peak,production_refinement_launched:false}' > "${existing}.tmp"
  fi
  mv "${existing}.tmp" "${existing}"
  exit "${stage_status}"
fi
jq -e --arg stage "${stage}" '.stage==$stage and (.status=="complete" or .status=="review_required")' "${checkpoint_dir}/${stage}.json" >/dev/null
echo "Stage ${stage} finished; peak process-group RSS ${peak_kib} KiB"
