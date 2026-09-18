# SHIELD Configuration Used

## Scope

This document records the prepared server-side LLaVA-1.5-7B SHIELD launch configuration. No local model inference, dataset evaluation, or benchmark result was run.

## Server Environment

- Server paths confirmed by the user:
  - BEAF dataset (original JPGs + manipulated PNGs + `beaf_qna.json`): `/home/nvidia-lab/ai4life/phuongnh/vlm-truth/data/beaf/`
  - COCO val2014 images (actual JPGs): `/home/nvidia-lab/ai4life/phuongnh/vlm-truth/data/coco2014/val2014/val2014/`
  - `beaf_qna.json` (ver1, 26064 entries) already exists at `/home/nvidia-lab/ai4life/phuongnh/vlm-truth/data/beaf/beaf_qna.json`
  - Official `beaf_metric.py` already exists on the server at the same `beaf/` folder
- Server has GPU(s); run everything on the server.

## Editable Configuration Files

Change only the path and runtime values in these files when the server paths are available:

- POPE-COCO: `experiments/configs/llava15_pope_coco.env` (already set to server paths)
- CHAIR: `experiments/configs/llava15_chair.env` (already set to server paths)
- BEAF: `experiments/configs/llava15_beaf.env` (already set to server paths)

Both files are shell environment files. Quote any value containing spaces, for example `PROMPT="Describe this image."`.

Optional runtime variables:

- `CUDA_VISIBLE_DEVICES`: GPU index to use (default `0`). On a single-GPU server leave it as `0`; on a multi-GPU server set it to a free GPU index, for example `CUDA_VISIBLE_DEVICES=3`.
- `HF_HOME`: Hugging Face cache directory. Set to `/home/nvidia-lab/data_mount/hf_cache` in all three configs (the previously used server cache location; verify with `df -h /home/nvidia-lab/data_mount` that it has at least ~20 GB free for the LLaVA model + CLIP download). The launchers export it automatically when non-empty.
- `HF_HUB_DISABLE_XET`: set to `1` in all three configs to use the standard HTTP download path, matching the previously used server workflow.

## Base Model

- Model: `liuhaotian/llava-v1.5-7b`
- SHIELD uses the repository's vendored LLaVA implementation in `experiments/llava/`.
- The launchers do not modify the original author scripts in `experiments/scripts/llava1.5_*.bash`.

## Author SHIELD Defaults

| Benchmark | cd_alpha | cd_beta | noise_step |   the | gamma_gain | gamma_reduce | gain_per | reduce_per | bias_weight | bias_sample_num | cw_epsilon | cw_num_steps | cw_c | cw_lr | seed |
| --------- | -------: | ------: | ---------: | ----: | ---------: | -----------: | -------: | ---------: | ----------: | --------------: | ---------: | -----------: | ---: | ----: | ---: |
| POPE-COCO |      2.0 |    0.35 |        999 | 0.011 |        3.0 |          3.0 |      0.5 |        0.0 |         0.1 |              32 |       0.14 |           30 |   12 |  0.14 |   42 |
| CHAIR     |      2.0 |    0.35 |        500 | 0.002 |        3.0 |          3.0 |     0.55 |        0.0 |        0.01 |              32 |       0.14 |           30 |   12 |  0.02 |   42 |
| BEAF      |      2.0 |    0.35 |        999 | 0.011 |        3.0 |          3.0 |      0.5 |        0.0 |         0.1 |              32 |       0.14 |           30 |   12 |  0.14 |   42 |

These values match the default expansions in `experiments/scripts/llava1.5_pope_coco.bash` and `experiments/scripts/llava1.5_chair.bash`.

## POPE-COCO

Configured source paths:

- `MODEL_PATH`: LLaVA checkpoint directory or Hugging Face model ID
- `COCO_IMAGE_DIR`: COCO val2014 image directory
- `POPE_DATA_DIR`: directory that contains `coco_pope_{random,popular,adversarial}.json`
- `POPE_CAPTION_FILE`: SHIELD first-round caption JSONL
- `OUTPUT_DIR`: generated-answer destination

The launcher generates answers with `model.generate`; it does not compute yes/no logits.

Preview the resolved command before a server run:

```bash
bash experiments/scripts/run_llava15_pope_coco.sh --config experiments/configs/llava15_pope_coco.env --split random --dry-run
```

Run each POPE split:

```bash
bash experiments/scripts/run_llava15_pope_coco.sh --config experiments/configs/llava15_pope_coco.env --split random
bash experiments/scripts/run_llava15_pope_coco.sh --config experiments/configs/llava15_pope_coco.env --split popular
bash experiments/scripts/run_llava15_pope_coco.sh --config experiments/configs/llava15_pope_coco.env --split adversarial
```

Evaluate each generated JSONL with the matching ground truth using `experiments/eval/eval_pope.py`. It reports Precision, Recall, F1, Accuracy, and yes-answer proportion.

```bash
python experiments/eval/eval_pope.py \
  --gt_files /path/to/coco_pope_random.json \
  --gen_files /path/to/llava15_coco_pope_random_answers_*.jsonl
```

## CHAIR

Configured source paths:

- `MODEL_PATH`: LLaVA checkpoint directory or Hugging Face model ID
- `COCO_IMAGE_DIR`: COCO val2014 image directory
- `CHAIR_QUESTION_FILE`: selected CHAIR question JSONL
- `CHAIR_CAPTION_FILE`: SHIELD first-round caption JSONL
- `CHAIR_CACHE_FILE`: `chair.pkl` cache or equivalent server path
- `OUTPUT_DIR`: generated-caption destination

The prepared CHAIR configuration uses the required protocol:

- Prompt: `Describe this image.`
- `MAX_NEW_TOKENS=128`
- The included CHAIR question file (`experiments/data/CHAIR/questions.jsonl`) contains exactly 500 records over 500 unique images, identical to upstream.

Preview or run:

```bash
bash experiments/scripts/run_llava15_chair.sh --config experiments/configs/llava15_chair.env --dry-run
bash experiments/scripts/run_llava15_chair.sh --config experiments/configs/llava15_chair.env
```

The launcher runs `chair_eval.py` after inference and reports CHAIRs, CHAIRi, Recall, and Caption Length.

## BEAF

**Priority: run BEAF first** (BEAF > POPE-COCO > CHAIR). This pipeline follows the official BEAF run procedure from `kaist-ami/BEAF`: iterate `beaf_qna.json` in id order, ask each POPE-style yes/no question, collect model answers into `[{"id": int, "answer": str}]`, then run `beaf_metric.py`.

Configured source paths:

- `MODEL_PATH`: LLaVA checkpoint directory or Hugging Face model ID
- `BEAF_IMAGE_DIR`: directory containing both original COCO JPEGs and manipulated images (`_NN.png` and `_NN.jpg`)
- `BEAF_QNA_FILE`: `beaf_qna.json` (ver1, 26064 entries)
- `BEAF_CAPTION_FILE`: SHIELD first-round caption JSONL — generated for ALL 2,223 unique BEAF images (originals + manipulated) so every evaluated image has its own caption (strict 1:1 SHIELD pairing, no cross-image caption reuse)
- `OUTPUT_DIR`: generated-answer destination

### Data Download

The BEAF dataset already exists on the server. No download needed.

### Run

Preview or run:

```bash
bash experiments/scripts/run_llava15_beaf.sh --config experiments/configs/llava15_beaf.env --dry-run
bash experiments/scripts/run_llava15_beaf.sh --config experiments/configs/llava15_beaf.env
```

The launcher first generates missing first-round captions for the unique BEAF images not yet captioned (resumable; roughly 1-2 hours for all 2,223 images), then runs inference, then `beaf_metric.py` and reports: Accuracy, Precision, Recall, F1, TU, IG, SB+, SB-, ID, F1(TU,ID). Expected inference duration is roughly 30-40 hours on one GPU (26,064 questions; the SHIELD pipeline recomputes the CLIP attack per question).

Caption generation follows the SHIELD authors' measured settings: the shipped `first_cap` files (500 POPE captions) were produced by `llava-v1.5-7b` with the recorded detailed-description prompt and `max_new_tokens=70` (verified: 452/500 shipped captions measure exactly 70 tokens with the Vicuna tokenizer), so the BEAF captions are generated with the same prompt, model, and 70-token cap.

### BEAF Eval Notes

- Before inference, `beaf_llava.py` pre-flights the dataset: it fails fast (before any GPU work) if any image file or any first-round caption is missing - no silently skipped questions.
- The included `experiments/eval/beaf_metric.py` is the official metric from `kaist-ami/BEAF` with documented deviations: the stale hardcoded asserts (`26118`/`1727`, ver0 constants) are replaced by an explicit length check and denominators computed from the data (ver1: `26064` questions, `1778` removed-question tuples - matching the paper's formula), and answers containing neither `yes` nor `no` raise a clear error listing the offending ids instead of the official script's silent crash - review and normalize such answers in the answers file before scoring.
- Model answers must be a JSON array of `{"id": int, "answer": str}` in strict id order (0..26063); every question must be answered.
- Every evaluated image (original and manipulated) gets its own first-round caption; the caption generator captions each unique image file exactly as referenced by the qna, so manipulated `_NN.png`/`_NN.jpg` names are captioned 1:1.

## Runtime Prerequisites

Use a dedicated server environment for SHIELD. The repository specifies Python 3.10, PyTorch 2.0.1, TorchVision 0.15.2, and Transformers 4.31.0 in `README.md` and `requirements.txt`.

The included author installation command targets CUDA 11.8:

```bash
conda create -n shield-env python=3.10
conda activate shield-env
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

`bitsandbytes` has been removed from `requirements.txt` because SHIELD runs fp16 only and its 0.41.0 CUDA setup breaks the transformers import chain on servers where CUDA runtime libraries are bundled inside the pip torch wheel. If an existing environment still has it installed, remove it with `pip uninstall -y bitsandbytes`.

Install the additional CHAIR dependency before evaluation:

```bash
pip install git+https://github.com/clips/pattern.git
```

## Validation Already Performed

- `python3 -m unittest tests/test_llava15_pope_launcher.py -v`
- `bash experiments/scripts/run_llava15_pope_coco.sh --dry-run`
- `bash experiments/scripts/run_llava15_chair.sh --dry-run`
- `bash experiments/scripts/run_llava15_beaf.sh --dry-run`
- `python3 -m py_compile experiments/eval/object_hallucination_vqa_llava.py experiments/eval/chair-llava.py experiments/eval/beaf_llava.py experiments/eval/beaf_metric.py`
- `bash -n experiments/scripts/run_llava15_pope_coco.sh experiments/scripts/run_llava15_chair.sh experiments/scripts/run_llava15_beaf.sh`

The checks validate configuration parsing, resolved paths, author-default arguments, evaluator syntax, and shell syntax only. They do not load a checkpoint or access dataset images.

## How to Run on the Server (Step by Step)

### Step 0: Transfer the repository to the server

```bash
# On local machine, create a bundle of the repository (or push/pull via git)
git archive HEAD -o shield_repo.tar.gz
scp shield_repo.tar.gz nvidia-lab@SERVER_IP:~/ai4life/phuongnh/

# On the server
cd ~/ai4life/phuongnh
tar xzf shield_repo.tar.gz -C vlm-truth/shield_repo
cd vlm-truth/shield_repo
```

### Step 1: Create the conda environment on the server (one time only)

```bash
conda create -y -n shield-env python=3.10
conda activate shield-env
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install git+https://github.com/clips/pattern.git
```

Verify the environment:

```bash
python -c "import torch, torchvision, transformers; print(torch.__version__, torch.cuda.is_available())"
```

Expected: `2.0.1 True` (CUDA available on server).

### Step 2: Run BEAF (priority 1)

```bash
# Preview the command (no model loaded):
bash experiments/scripts/run_llava15_beaf.sh --config experiments/configs/llava15_beaf.env --dry-run

# Run inference + metric:
nohup bash experiments/scripts/run_llava15_beaf.sh --config experiments/configs/llava15_beaf.env > beaf_run.log 2>&1 &
tail -f beaf_run.log
```

Results are written to `output/llava15_beaf_answers_seed42.json` and the metric table is printed at the end of `beaf_run.log` (Accuracy, Precision, Recall, F1, TU, IG, SB+, SB-, ID, F1(TU,ID)).

### Step 3: Run POPE-COCO (priority 2)

```bash
# Preview:
bash experiments/scripts/run_llava15_pope_coco.sh --config experiments/configs/llava15_pope_coco.env --split random --dry-run

# Run all three splits:
for split in random popular adversarial; do
  nohup bash experiments/scripts/run_llava15_pope_coco.sh --config experiments/configs/llava15_pope_coco.env --split $split > pope_${split}.log 2>&1 &
done
tail -f pope_random.log
```

Then evaluate each generated answer file:

```bash
python experiments/eval/eval_pope.py \
  --gt_files experiments/data/POPE/coco/coco_pope_random.json \
  --gen_files output/llava15_coco_pope_random_answers_*.jsonl
```

Repeat with `popular` and `adversarial` for all three split results.

### Step 4: Run CHAIR (priority 3)

```bash
bash experiments/scripts/run_llava15_chair.sh --config experiments/configs/llava15_chair.env --dry-run
nohup bash experiments/scripts/run_llava15_chair.sh --config experiments/configs/llava15_chair.env > chair_run.log 2>&1 &
tail -f chair_run.log
```

The launcher runs `chair_eval.py` after inference and prints CHAIRs, CHAIRi, Recall, and Caption Length.

### Step 5: If something fails

Paste the error log into this chat so the agent can fix it. Common issues:

- CUDA OOM: reduce `MAX_NEW_TOKENS` or split the question file into chunks
- Missing image file: check `BEAF_IMAGE_DIR` or `COCO_IMAGE_DIR` config values
- Metric crash: check that `beaf_qna.json` entry count matches the answer file length
- `bitsandbytes` CUDA Setup failure at import (`RuntimeError: CUDA Setup failed despite GPU being available`): SHIELD does not use quantization. Uninstall it and rerun:

```bash
pip uninstall -y bitsandbytes
python -c "from transformers import AutoModelForCausalLM; print('transformers OK')"
```
- Hugging Face `Not enough free disk space to download the file`: set `HF_HOME` in the `.env` config to a directory with at least 20 GB free (the model needs ~14 GB plus ~2 GB for CLIP), then rerun. When `HF_HOME` is empty the launchers use the system default (`~/.cache/huggingface`).
