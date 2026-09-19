#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_TARGET="all"
BENCHMARKS="beaf,pope,chair,causalhal"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_TARGET="$2"
            shift 2
            ;;
        --benchmarks)
            BENCHMARKS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

cd "$REPOSITORY_ROOT"

log_info() {
    printf '[%s] INFO: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

run_step() {
    local name="$1"
    shift
    log_info "=== STARTING TASK: $name ==="
    log_info "Command: $*"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log_info "[DRY-RUN] Skipping actual execution."
    else
        "$@"
        log_info "=== COMPLETED TASK: $name ==="
    fi
    printf '\n'
}

run_llava15_pipeline() {
    log_info "=========================================="
    log_info "     STARTING LLaVA-1.5 EVALUATIONS       "
    log_info "=========================================="

    if [[ "$BENCHMARKS" == *"beaf"* ]]; then
        run_step "LLaVA-1.5 BEAF" ./experiments/scripts/run_llava15_beaf.sh
    fi

    if [[ "$BENCHMARKS" == *"pope"* ]]; then
        for split in random popular adversarial; do
            run_step "LLaVA-1.5 POPE-COCO ($split)" ./experiments/scripts/run_llava15_pope_coco.sh --split "$split"
        done
    fi

    if [[ "$BENCHMARKS" == *"chair"* ]]; then
        run_step "LLaVA-1.5 CHAIR" ./experiments/scripts/run_llava15_chair.sh
    fi

    if [[ "$BENCHMARKS" == *"causalhal"* ]]; then
        run_step "LLaVA-1.5 CausalHal" ./experiments/scripts/run_llava15_causalhal.sh
    fi
}

run_qwen2vl_pipeline() {
    log_info "=========================================="
    log_info "     STARTING Qwen2-VL EVALUATIONS        "
    log_info "=========================================="

    if [[ "$BENCHMARKS" == *"beaf"* ]]; then
        run_step "Qwen2-VL BEAF" ./experiments/scripts/run_qwen2vl_beaf.sh
    fi

    if [[ "$BENCHMARKS" == *"pope"* ]]; then
        for split in random popular adversarial; do
            run_step "Qwen2-VL POPE-COCO ($split)" ./experiments/scripts/run_qwen2vl_pope_coco.sh --split "$split"
        done
    fi

    if [[ "$BENCHMARKS" == *"chair"* ]]; then
        run_step "Qwen2-VL CHAIR" ./experiments/scripts/run_qwen2vl_chair.sh
    fi

    if [[ "$BENCHMARKS" == *"causalhal"* ]]; then
        run_step "Qwen2-VL CausalHal" ./experiments/scripts/run_qwen2vl_causalhal.sh
    fi
}

case "$MODEL_TARGET" in
    llava15)
        run_llava15_pipeline
        ;;
    qwen2vl)
        run_qwen2vl_pipeline
        ;;
    all)
        run_llava15_pipeline
        run_qwen2vl_pipeline
        ;;
    *)
        printf 'Invalid model target: %s (choose llava15, qwen2vl, or all)\n' "$MODEL_TARGET" >&2
        exit 1
        ;;
esac

log_info "All selected pipeline evaluations finished successfully!"
