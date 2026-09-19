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
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

from transformers import set_seed

DEFAULT_CAPTION_PROMPT = (
    "Describe the image in as much detail as possible. First, focus on the main objects, "
    "such as people or items in the foreground. After describing the main objects, provide "
    "details about the background, including natural environments and other surroundings."
)


def unique_images_in_order(questions, orig_only):
    seen, images = set(), []
    for q in questions:
        image_file = q["image"]
        if orig_only and not image_file.endswith(".jpg"):
            continue
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
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    done = {entry["image"] for entry in existing_entries}
    pending = [image_file for image_file in image_files if image_file not in done]

    os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(args.output_file))), exist_ok=True)
    out_file = open(os.path.expanduser(args.output_file), "a")

    for image_file in tqdm(pending):
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + args.prompt
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + args.prompt

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        out_file.write(json.dumps({"prompt": args.prompt,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "image": image_file}) + "\n")
        out_file.flush()
    out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--prompt", type=str, default=DEFAULT_CAPTION_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=70)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--orig-only", action="store_true", default=False)
    args = parser.parse_args()
    set_seed(args.seed)

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
    image_files = unique_images_in_order(questions, args.orig_only)
    existing_entries = load_existing(args.output_file)
    done = {entry["image"] for entry in existing_entries}
    pending_count = sum(1 for image_file in image_files if image_file not in done)

    print(f"unique images: {len(image_files)}, already captioned: {len(done)}, pending: {pending_count}")
    if pending_count == 0:
        print("first-round captions complete")
        sys.exit(0)

    eval_model(args, image_files, existing_entries)
