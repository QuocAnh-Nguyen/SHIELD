import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from setproctitle import setproctitle
    setproctitle("phuongnh_vlm_truth")
except ImportError:
    print("Warning: setproctitle not installed - process stays named 'python' (pip install setproctitle)")
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

from transformers import set_seed

import shield
from shield.caption import find_text_by_image, load_captions


def answer_one_question(args, model, tokenizer, line, image, image_tensor, shield_kw):
    idx = line["id"]
    qs_text = line["question"]

    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_text
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs_text

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            **shield_kw,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    shield.wrap(model, tokenizer,
        caption_file=args.caption_file,
        cd_alpha=args.cd_alpha,
        cd_beta=args.cd_beta,
        the=args.the,
        gamma_gain=args.gamma_gain,
        gamma_reduce=args.gamma_reduce,
        gain_per=args.gain_per,
        reduce_per=args.reduce_per,
        bias_weight=args.bias_weight,
        bias_sample_num=args.bias_sample_num,
        epsilon=args.cw_epsilon,
        num_steps=args.cw_num_steps,
        c=args.cw_c,
        lr=args.cw_lr,
    )

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)

    missing_images = [
        q["image"] for q in questions
        if not os.path.exists(os.path.join(args.image_folder, q["image"]))
    ]
    if missing_images:
        print(f"Missing {len(missing_images)} image files, e.g.: {missing_images[:10]}")
        sys.exit(1)

    captions = load_captions(args.caption_file)
    missing_captions = [
        q["image"] for q in questions
        if find_text_by_image(q["image"], captions) is None
    ]
    if missing_captions:
        print(f"Missing {len(missing_captions)} first-round captions, e.g.: {missing_captions[:10]}")
        sys.exit(1)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    results = []
    answer_map = {}
    if os.path.exists(answers_file):
        try:
            with open(answers_file, "r") as f:
                for entry in json.load(f):
                    if entry.get("answer", "") != "":
                        answer_map[entry["id"]] = entry["answer"]
        except json.JSONDecodeError:
            print("Existing answers file is corrupt - starting fresh")
            answer_map = {}

    pending = [q for q in questions if q["id"] not in answer_map]
    print(f"Resuming: {len(answer_map)} answers saved, {len(pending)} pending (failed/empty answers are retried)")

    failed_questions = []
    cached_image = None
    image, image_tensor, shield_kw = None, None, None
    for line in tqdm(pending, initial=len(answer_map), total=len(questions)):
        idx = line["id"]
        image_file = line["image"]

        answer_text = None
        for attempt in range(3):
            try:
                if image_file != cached_image:
                    image_path = os.path.join(args.image_folder, image_file)
                    if not os.path.exists(image_path):
                        raise FileNotFoundError(f"Missing image file for question {idx}: {image_path}")
                    image = Image.open(image_path)
                    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    shield_kw = model.shield_prepare(image, image_tensor, image_file, use_cd=args.use_cd)
                    cached_image = image_file
                answer_text = answer_one_question(args, model, tokenizer, line, image, image_tensor, shield_kw)
                break
            except Exception as e:
                torch.cuda.empty_cache()
                print(f"[ERROR] question {idx} ({image_file}) attempt {attempt + 1}/3: {type(e).__name__}: {e}")

        if answer_text is None:
            failed_questions.append(idx)
            print(f"[ERROR] question {idx} failed after 3 attempts - recorded as empty answer, process keeps running")
            answer_map[idx] = ""
        else:
            answer_map[idx] = answer_text

        tmp_file = answers_file + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump([{"id": i, "answer": a} for i, a in sorted(answer_map.items())], f, indent=2)
        os.replace(tmp_file, answers_file)

    print(f"Saved {len(results)} answers to {answers_file}")
    if failed_questions:
        print(f"WARNING: {len(failed_questions)} questions failed and were recorded as empty answers: {failed_questions[:20]}")
        print("Retry them by removing their entries from the answers file and re-running (resume re-attempts them).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=2.0)
    parser.add_argument("--cd_beta", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--the", type=float, default=0.011)
    parser.add_argument("--gamma_gain", type=float, default=3.0)
    parser.add_argument("--gamma_reduce", type=float, default=3.0)
    parser.add_argument("--gain_per", type=float, default=0.5)
    parser.add_argument("--reduce_per", type=float, default=0.0)
    parser.add_argument("--bias_weight", type=float, default=0.1)
    parser.add_argument("--bias_sample_num", type=float, default=32)
    parser.add_argument("--cw_epsilon", type=float, default=0.14)
    parser.add_argument("--cw_num_steps", type=float, default=30)
    parser.add_argument("--cw_c", type=float, default=12)
    parser.add_argument("--cw_lr", type=float, default=0.14)
    parser.add_argument("--caption-file", type=str, required=True)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)

    shield.clear_bias_cache()
    shield.clear_clip_cache()
