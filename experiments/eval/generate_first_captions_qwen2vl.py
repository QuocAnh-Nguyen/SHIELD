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

CAPTION_PROMPT = (
    "Describe the image in as much detail as possible. First, focus on the main objects, "
    "such as people or items in the foreground. After describing the main objects, provide "
    "details about the background, including natural environments and other surroundings."
)


def unique_images_in_order(questions):
    seen, images = set(), []
    for q in questions:
        image_file = q.get("image_name") or q["image"]
        if image_file not in seen:
            seen.add(image_file)
            images.append(image_file)
    return images


def load_existing(output_file):
    if not os.path.exists(output_file):
        return []
    entries = []
    with open(output_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def eval_model(args, image_files, existing_entries):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    model_path = os.path.expanduser(args.model_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    model_name = model.config._name_or_path

    done = {entry["image"] for entry in existing_entries}
    pending = [image_file for image_file in image_files if image_file not in done]

    os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(args.output_file))), exist_ok=True)
    out_file = open(os.path.expanduser(args.output_file), "a")

    for image_file in tqdm(pending):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": os.path.join(args.image_folder, image_file)},
                    {"type": "text", "text": args.prompt},
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
        )
        inputs = inputs.to("cuda")

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        outputs = outputs.strip()

        out_file.write(json.dumps({"prompt": args.prompt,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "image": image_file}) + "\n")
        out_file.flush()
    out_file.close()


def load_questions(question_file):
    with open(os.path.expanduser(question_file), "r") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=CAPTION_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=70)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    questions = load_questions(args.question_file)
    image_files = unique_images_in_order(questions)
    existing_entries = load_existing(args.output_file)
    done = {entry["image"] for entry in existing_entries}
    pending_count = sum(1 for image_file in image_files if image_file not in done)

    print(f"unique images: {len(image_files)}, already captioned: {len(done)}, pending: {pending_count}")
    if pending_count == 0:
        print("first-round captions complete")
        sys.exit(0)

    eval_model(args, image_files, existing_entries)
