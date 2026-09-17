#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="$REPOSITORY_ROOT/experiments/configs/llava15_pope_coco.env"
SPLIT="random"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
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

case "$SPLIT" in
    random|popular|adversarial)
        ;;
    *)
        printf 'Invalid POPE split: %s\n' "$SPLIT" >&2
        exit 2
        ;;
esac

required_variables=(
    MODEL_PATH COCO_IMAGE_DIR POPE_DATA_DIR POPE_CAPTION_FILE OUTPUT_DIR
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

question_file="$POPE_DATA_DIR/coco_pope_${SPLIT}.json"
answers_file="$OUTPUT_DIR/llava15_coco_pope_${SPLIT}_answers_bias_weight${BIAS_WEIGHT}_bias_sample_num${BIAS_SAMPLE_NUM}_alpha${CD_ALPHA}_beta${CD_BETA}_the${THE}_gamma${GAMMA_GAIN}_per${GAIN_PER}_cw_epsilon${CW_EPSILON}_cw_c${CW_C}_cw_lr${CW_LR}_seed${SEED}.jsonl"

command=(
    python experiments/eval/object_hallucination_vqa_llava.py
    --model-path "$MODEL_PATH"
    --question-file "$question_file"
    --image-folder "$COCO_IMAGE_DIR"
    --caption-file "$POPE_CAPTION_FILE"
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

printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
if [[ -n "${HF_HOME:-}" ]]; then
    printf 'HF_HOME=%s\n' "$HF_HOME"
fi
printf '%q ' "${command[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

if [[ -n "${HF_HOME:-}" ]]; then
    export HF_HOME
fi
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${command[@]}"
