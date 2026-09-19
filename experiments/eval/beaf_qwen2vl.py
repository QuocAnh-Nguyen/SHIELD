import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from setproctitle import setproctitle
    setproctitle("phuongnh_vlm_truth")
except ImportError:
    print("Warning: setproctitle not installed - process stays named 'python' (pip install setproctitle)")

from transformers import set_seed

import shield
from shield.qwen2vl import wrap_qwen2vl
from shield.caption import find_text_by_image, load_captions


def eval_model(args):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    model_path = os.path.expanduser(args.model_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    tokenizer = processor.tokenizer

    wrap_qwen2vl(model, tokenizer,
        caption_file=args.caption_file,
        qwen_processor=processor,
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
    image_path_cached, image, pixel_values, image_grid_thw, shield_kw = None, None, None, None, None
    for line in tqdm(pending, initial=len(answer_map), total=len(questions)):
        idx = line["id"]
        image_file = line["image"]
        qs_text = line["question"]

        image_path = os.path.join(args.image_folder, image_file)

        answer_text = None
        for attempt in range(3):
            try:
                if image_file != cached_image:
                    if not os.path.exists(image_path):
                        raise FileNotFoundError(f"Missing image file for question {idx}: {image_path}")
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": qs_text + " Please answer with yes or no."},
                            ],
                        }
                    ]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to("cuda")
                    image = image_inputs[0]
                    image_grid_thw = inputs.image_grid_thw
                    pixel_values = inputs.pixel_values[0]
                    shield_kw = model.shield_prepare(image, pixel_values, image_file, image_grid_thw, use_cd=args.use_cd)
                    cached_image = image_file

                messages_q = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": qs_text + " Please answer with yes or no."},
                        ],
                    }
                ]
                text_q = processor.apply_chat_template(messages_q, tokenize=False, add_generation_prompt=True)
                image_inputs_q, video_inputs_q = process_vision_info(messages_q)
                inputs_q = processor(
                    text=[text_q],
                    images=image_inputs_q,
                    videos=video_inputs_q,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")

                with torch.inference_mode():
                    output_ids = model.generate(
                        inputs_q.input_ids,
                        **shield_kw,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_q.input_ids, output_ids)
                ]
                answer_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()
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

    print(f"Saved {len(answer_map)} answers to {answers_file}")
    if failed_questions:
        print(f"WARNING: {len(failed_questions)} questions failed and were recorded as empty answers: {failed_questions[:20]}")
        print("Retry them by re-running (resume re-attempts failed/empty answers automatically).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
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
    shield.clear_qwen2vl_bias_cache()
    shield.clear_clip_cache()
