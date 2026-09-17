#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="$REPOSITORY_ROOT/experiments/configs/llava15_beaf.env"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
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

if [[ ! -f "$CONFIG_FILE" ]]; then
    printf 'Configuration file not found: %s\n' "$CONFIG_FILE" >&2
    exit 1
fi

source "$CONFIG_FILE"

required_variables=(
    MODEL_PATH BEAF_IMAGE_DIR BEAF_QNA_FILE BEAF_CAPTION_FILE OUTPUT_DIR
    CUDA_VISIBLE_DEVICES SEED CD_ALPHA CD_BETA NOISE_STEP THE GAMMA_GAIN
    GAMMA_REDUCE GAIN_PER REDUCE_PER BIAS_WEIGHT BIAS_SAMPLE_NUM CW_EPSILON
    CW_NUM_STEPS CW_C CW_LR MAX_NEW_TOKENS
)

for variable in "${required_variables[@]}"; do
    if [[ -z "${!variable:-}" ]]; then
        printf 'Missing required configuration variable: %s\n' "$variable" >&2
        exit 1
    fi
done

answers_file="$OUTPUT_DIR/llava15_beaf_answers_seed${SEED}.json"

inference_command=(
    python experiments/eval/beaf_llava.py
    --model-path "$MODEL_PATH"
    --question-file "$BEAF_QNA_FILE"
    --image-folder "$BEAF_IMAGE_DIR"
    --caption-file "$BEAF_CAPTION_FILE"
    --answers-file "$answers_file"
    --use_cd
    --cd_alpha "$CD_ALPHA"
    --cd_beta "$CD_BETA"
    --noise_step "$NOISE_STEP"
    --the "$THE"
    --gamma_gain "$GAMMA_GAIN"
    --gamma_reduce "$GAMMA_REDUCE"
    --gain_per "$GAIN_PER"
    --reduce_per "$REDUCE_PER"
    --bias_weight "$BIAS_WEIGHT"
    --bias_sample_num "$BIAS_SAMPLE_NUM"
    --cw_epsilon "$CW_EPSILON"
    --cw_num_steps "$CW_NUM_STEPS"
    --cw_c "$CW_C"
    --cw_lr "$CW_LR"
    --seed "$SEED"
    --max-new-tokens "$MAX_NEW_TOKENS"
)

evaluation_command=(
    python experiments/eval/beaf_metric.py
    --qna-path "$BEAF_QNA_FILE"
    --model-answers "$answers_file"
)

printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
if [[ -n "${HF_HOME:-}" ]]; then
    printf 'HF_HOME=%s\n' "$HF_HOME"
fi
if [[ -n "${HF_HUB_DISABLE_XET:-}" ]]; then
    printf 'HF_HUB_DISABLE_XET=%s\n' "$HF_HUB_DISABLE_XET"
fi
printf '%q ' "${inference_command[@]}"
printf '\n'
printf '%q ' "${evaluation_command[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

if [[ -n "${HF_HOME:-}" ]]; then
    export HF_HOME
fi
if [[ -n "${HF_HUB_DISABLE_XET:-}" ]]; then
    export HF_HUB_DISABLE_XET
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${inference_command[@]}"
"${evaluation_command[@]}"
