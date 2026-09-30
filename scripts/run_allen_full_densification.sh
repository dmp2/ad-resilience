#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT="/cis/home/dpadova/Documents/git/ad-resilience"
SCRIPT="$PROJECT/scripts/run_allen_full_densification.sh"
SESSION="allen-annotation-densify-corrected"

cd "$PROJECT"

DATASET="data/derivatives/allen/specimen_708424/histology_symmetric_nissl_native_200um_section_aligned"
ANNOTATIONS="data/derivatives/allen/specimen_708424/annotations_symmetric_nissl_native_200um_section_aligned"
REGISTRATION="results/allen/specimen_708424/emlddmm/native-200um-clean/HIST_NISSL_SYMMETRIC_SECTION_ALIGNED_to_MRI_7T_WHOLE"
HISTORICAL_OUTPUT="data/derivatives/allen/specimen_708424/annotations_dense_annotation_driven_200um_groups_31_265297118"
OUTPUT="data/derivatives/allen/specimen_708424/annotations_dense_annotation_driven_corrected_200um_groups_31_265297118"
PAIR_CONFIG="configs/allen_dense_pairwise.json"
PAIR_PROFILE="section-to-section-diffeo"
EXPECTED_CONFIG_SHA256="c77abf66122961056da1d42ec8e544731582a0fb7be84b828b0f69806ae49457"
WSI_REPOSITORY="/cis/home/dpadova/Documents/git/wsi-tissue-pipeline"
SCHEDULE_ROOT="results/diagnostics/allen_cpu_gpu_scheduling"
GPU_WORKLIST="$SCHEDULE_ROOT/corrected_production_gpu_worklist.tsv"
CPU_WORKLIST="$SCHEDULE_ROOT/corrected_production_cpu_worklist.tsv"
INVENTORY="$SCHEDULE_ROOT/final_cpu_gpu_eligibility_105_pairs.tsv"
LOG_ROOT="$PROJECT/results/logs"

COMMON_ARGS=(
    --dataset "$DATASET"
    --annotations "$ANNOTATIONS"
    --registration-run "$REGISTRATION"
    --output "$OUTPUT"
    --driver annotation
    --graphic-groups 31 265297118
    --output-format tiff
    --tiff-compression deflate
    --pair-config "$PAIR_CONFIG"
    --pair-profile "$PAIR_PROFILE"
    --expected-configuration-sha256 "$EXPECTED_CONFIG_SHA256"
    --wsi-repository "$WSI_REPOSITORY"
)
SCHEDULE_ARGS=(
    --gpu-worklist "$GPU_WORKLIST"
    --cpu-worklist "$CPU_WORKLIST"
    --schedule-inventory "$INVENTORY"
)

activate_environment() {
    # A worker invokes this launcher again for consolidation. Avoid asking Conda
    # to deactivate/reactivate the environment already inherited by that child.
    if [[ "${CONDA_DEFAULT_ENV:-}" == "wsi-pipeline" ]]; then
        return
    fi
    # Conda activation hooks are not required to be nounset-safe. Restore the
    # launcher's strict mode immediately after activation.
    set +u
    export CLASSPATH="${CLASSPATH:-}"
    source /cis/home/dpadova/miniconda3/etc/profile.d/conda.sh
    conda activate wsi-pipeline
    set -u
}

check_inputs() {
    test "$OUTPUT" != "$HISTORICAL_OUTPUT"
    test -f "$PAIR_CONFIG"
    test -f "$GPU_WORKLIST"
    test -f "$CPU_WORKLIST"
    test -f "$INVENTORY"
    test -d "$DATASET"
    test -d "$ANNOTATIONS"
    test -d "$REGISTRATION"
    test -d "$WSI_REPOSITORY"
    command -v tmux >/dev/null
    command -v flock >/dev/null

    PYTHONPATH="$PWD/src" python -m preprocess.densify_allen_annotations --help \
        | grep -q -- '--pair-worklist'
    PYTHONPATH="$PWD/src" python -m preprocess.densify_allen_annotations --help \
        | grep -q -- '--initialize-only'
    PYTHONPATH="$PWD/src" python -m preprocess.densify_allen_annotations --help \
        | grep -q -- '--consolidate-only'
}

initialize_corrected_output() {
    mkdir -p "$LOG_ROOT"
    local log="$LOG_ROOT/allen_annotation_densify_corrected_initialize.log"
    echo "Initializing corrected output without registration: $OUTPUT" | tee -a "$log"
    /usr/bin/time -v \
    env \
        PYTHONPATH="$PWD/src" \
        MPLBACKEND=Agg \
        PYTHONUNBUFFERED=1 \
    python -m preprocess.densify_allen_annotations \
        "${COMMON_ARGS[@]}" \
        "${SCHEDULE_ARGS[@]}" \
        --initialize-only \
        --device cpu \
        2>&1 | tee -a "$log"
}

record_exit_status() {
    local state_dir="$1"
    local worker="$2"
    local status="$3"
    local temporary="$state_dir/.${worker}.exit.$$"
    printf '%s\n' "$status" > "$temporary"
    mv "$temporary" "$state_dir/${worker}.exit"
}

consolidate_run() {
    local state_dir="${1:-}"
    mkdir -p "$LOG_ROOT"
    local log="$LOG_ROOT/allen_annotation_densify_corrected_consolidate.log"
    if [[ -n "$state_dir" && -f "$state_dir/consolidated.exit" ]]; then
        return 0
    fi
    set +e
    /usr/bin/time -v \
    env \
        PYTHONPATH="$PWD/src" \
        MPLBACKEND=Agg \
        PYTHONUNBUFFERED=1 \
    python -m preprocess.densify_allen_annotations \
        "${COMMON_ARGS[@]}" \
        --consolidate-only \
        --device cpu \
        2>&1 | tee -a "$log"
    local status=${PIPESTATUS[0]}
    set -e
    if [[ -n "$state_dir" ]]; then
        record_exit_status "$state_dir" consolidated "$status"
    fi
    return "$status"
}

run_worker() {
    local worker="$1"
    local device worklist
    case "$worker" in
        gpu)
            device="cuda:0"
            worklist="$GPU_WORKLIST"
            ;;
        cpu)
            device="cpu"
            worklist="$CPU_WORKLIST"
            ;;
        *)
            echo "Unknown worker: $worker" >&2
            return 2
            ;;
    esac
    : "${ALLEN_LAUNCH_STATE:?ALLEN_LAUNCH_STATE is required for worker mode}"
    local state_dir="$ALLEN_LAUNCH_STATE"
    mkdir -p "$state_dir" "$LOG_ROOT"
    local log="$LOG_ROOT/allen_annotation_densify_corrected_${worker}.log"

    exec > >(tee -a "$log") 2>&1
    echo "================================================"
    echo "Allen corrected annotation densification: $worker"
    echo "Started: $(date -Is)"
    echo "Environment: $CONDA_DEFAULT_ENV"
    echo "Device: $device"
    echo "Worklist: $worklist"
    echo "Output: $OUTPUT"
    echo "================================================"

    set +e
    /usr/bin/time -v \
    env \
        PYTHONPATH="$PWD/src" \
        MPLBACKEND=Agg \
        PYTHONUNBUFFERED=1 \
    python -m preprocess.densify_allen_annotations \
        "${COMMON_ARGS[@]}" \
        --pair-worklist "$worklist" \
        --worker-id "$worker" \
        --assigned-device "$device" \
        --device "$device"
    local worker_status=$?
    set -e
    record_exit_status "$state_dir" "$worker" "$worker_status"
    echo "WORKER EXIT STATUS: $worker_status at $(date -Is)"

    if [[ -f "$state_dir/gpu.exit" && -f "$state_dir/cpu.exit" ]]; then
        local coordinator_status=0
        flock -x "$state_dir/consolidate.lock" \
            "$SCRIPT" consolidate "$state_dir" || coordinator_status=$?
        echo "CONSOLIDATION EXIT STATUS: $coordinator_status at $(date -Is)"
    fi
    return "$worker_status"
}

run_serial_pair() {
    local pair="${1:?serial mode requires PAIR}"
    local device="${2:-cpu}"
    local overwrite="${3:-}"
    if [[ "$device" != "cpu" && "$device" != "cuda:0" ]]; then
        echo "Serial DEVICE must be cpu or cuda:0" >&2
        return 2
    fi
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Refusing serial work while tmux session $SESSION exists." >&2
        return 1
    fi
    initialize_corrected_output
    mkdir -p "$LOG_ROOT"
    local log="$LOG_ROOT/allen_annotation_densify_corrected_serial_${pair}.log"
    local extra=()
    if [[ "$overwrite" == "--overwrite" ]]; then
        extra+=(--overwrite)
    elif [[ -n "$overwrite" ]]; then
        echo "The only supported fourth argument is --overwrite" >&2
        return 2
    fi
    set +e
    /usr/bin/time -v \
    env \
        PYTHONPATH="$PWD/src" \
        MPLBACKEND=Agg \
        PYTHONUNBUFFERED=1 \
    python -m preprocess.densify_allen_annotations \
        "${COMMON_ARGS[@]}" \
        --pair "$pair" \
        --worker-id "serial-$pair" \
        --assigned-device "$device" \
        --device "$device" \
        "${extra[@]}" \
        2>&1 | tee -a "$log"
    local status=${PIPESTATUS[0]}
    set -e
    flock -x "$OUTPUT/metadata/consolidate.lock" "$SCRIPT" consolidate || true
    return "$status"
}

show_status() {
    echo "Session: $SESSION"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux list-panes -t "$SESSION" -a \
            -F '#{session_name}:#{window_name} dead=#{pane_dead} exit=#{pane_dead_status} pid=#{pane_pid}'
    else
        echo "tmux session is not present"
    fi
    echo "Corrected output: $OUTPUT"
    echo "Run status: $OUTPUT/metadata/corrected_run_status.json"
    echo "Failed pairs: $OUTPUT/metadata/failures/failed_pairs.tsv"
    echo "Retry worklist: $OUTPUT/metadata/failures/retry_worklist.tsv"
    echo "Known unresolved: $OUTPUT/metadata/failures/known_unresolved.tsv"
}

launch_tmux() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Refusing to create duplicate tmux session: $SESSION" >&2
        return 1
    fi
    initialize_corrected_output

    local launch_id
    launch_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
    local state_dir="$OUTPUT/metadata/launches/$launch_id"
    mkdir -p "$state_dir"

    tmux new-session -d -s "$SESSION" -n gpu -c "$PROJECT"
    tmux new-window -d -t "$SESSION" -n cpu -c "$PROJECT"
    tmux set-window-option -t "$SESSION" remain-on-exit on

    local gpu_command cpu_command
    printf -v gpu_command '%q ' env ALLEN_LAUNCH_STATE="$state_dir" bash "$SCRIPT" worker gpu
    printf -v cpu_command '%q ' env ALLEN_LAUNCH_STATE="$state_dir" bash "$SCRIPT" worker cpu
    tmux send-keys -t "$SESSION:gpu" "$gpu_command" C-m
    tmux send-keys -t "$SESSION:cpu" "$cpu_command" C-m

    echo "Created tmux session: $SESSION"
    echo "GPU worker: cuda:0, 80 pairs"
    echo "CPU worker: cpu, 24 pairs"
    echo "Known unresolved pair excluded: 1716-1787"
    echo "Attach with: tmux attach -t $SESSION"
}

usage() {
    cat <<EOF
Usage:
  bash scripts/run_allen_full_densification.sh
  bash scripts/run_allen_full_densification.sh resume
  bash scripts/run_allen_full_densification.sh status
  bash scripts/run_allen_full_densification.sh serial PAIR [cpu|cuda:0] [--overwrite]
  bash scripts/run_allen_full_densification.sh consolidate
EOF
}

mode="${1:-launch}"
case "$mode" in
    launch|resume)
        activate_environment
        check_inputs
        launch_tmux
        ;;
    worker)
        activate_environment
        check_inputs
        run_worker "${2:-}"
        ;;
    serial)
        activate_environment
        check_inputs
        run_serial_pair "${2:-}" "${3:-cpu}" "${4:-}"
        ;;
    consolidate)
        activate_environment
        check_inputs
        consolidate_run "${2:-}"
        ;;
    status)
        show_status
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
