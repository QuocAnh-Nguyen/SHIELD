"""SHIELD wrapper for Qwen2-VL (Qwen2VLForConditionalGeneration).

Adapts the three SHIELD mechanisms (token re-weighting, noise-derived token
subtraction, contrastive decoding) to the Qwen2-VL architecture. The LLaVA
wrapper in ``shield/wrapper.py`` is untouched; this module reuses the
model-agnostic pieces (feature weighting, captions, CLIP attack, sampling).

Requires ``transformers`` with Qwen2-VL support (>= 4.37; developed against
4.47.1). All Qwen2-VL imports are lazy so the LLaVA environment (4.31) keeps
working.
"""

import types

import torch
import torch.nn.functional as F

from . import sampling
from .attack import cw_attack, pgd_attack
from .caption import find_text_by_image, load_captions, process_caption
from .clip_utils import CLIP_MEAN, CLIP_STD, get_clip_text_features, load_clip_model
from .feature import compute_shield_image_weights, clear_bias_cache, map_text_segments


_qwen2vl_bias_cache = {}


def get_qwen2vl_bias(sample_num, pixel_values, image_grid_thw, vision_tower, cache_key=None):
    """Noise-derived bias features (paper Eq. 9) with the SAME size as the image.

    Qwen2-VL uses dynamic resolution, so the noise inputs are generated at the
    evaluated image's own grid; the result is cached per grid key.
    """
    global _qwen2vl_bias_cache

    if cache_key is None:
        cache_key = tuple(int(v) for v in image_grid_thw.flatten().tolist())

    if cache_key not in _qwen2vl_bias_cache:
        t, h, w = [int(v) for v in image_grid_thw[0].tolist()]
        merged_tokens = (h // 2) * (w // 2)
        randn = torch.rand(
            (sample_num * t * h * w, pixel_values.size(-1)),
            device=pixel_values.device,
        ).to(vision_tower.get_dtype())
        grid = image_grid_thw.repeat(sample_num, 1)
        feats = vision_tower(randn, grid_thw=grid)
        feats = feats.view(sample_num, merged_tokens, -1)
        _qwen2vl_bias_cache[cache_key] = feats.mean(dim=0).unsqueeze(0).half()

    return _qwen2vl_bias_cache[cache_key]


def clear_qwen2vl_bias_cache():
    """Clear the cached Qwen2-VL bias features."""
    global _qwen2vl_bias_cache
    _qwen2vl_bias_cache = {}


def qwen2vl_clip_attack(image, qwen_processor, text, epsilon, num_steps, c, lr,
                        attack_type="cw", clip_model=None, clip_processor=None):
    """CLIP attack bridged into the Qwen2-VL pixel space (paper Eq. 11-13).

    The perturbation is optimised in CLIP space on the CLIP-preprocessed
    image, converted back to [0, 1] units, resized to the image size, applied
    to the image, and re-processed by the Qwen2-VL processor. Returns
    (pixel_values, image_grid_thw) for the attacked image.
    """
    if clip_model is None or clip_processor is None:
        clip_model, clip_processor = load_clip_model()

    dtype = next(clip_model.parameters()).dtype
    inputs = clip_processor(
        text=text, images=image, return_tensors="pt", padding=True, truncation=True
    ).to("cuda")
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    if attack_type == "cw":
        perturbation = cw_attack(clip_model, inputs, epsilon=epsilon, num_steps=num_steps, c=c, lr=lr)
    elif attack_type == "pgd":
        perturbation = pgd_attack(clip_model, inputs, epsilon=epsilon, num_steps=num_steps,
                                  alpha=epsilon / 2.5)
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")

    perturbation = perturbation.to(image_tensor_device := "cpu").float()
    perturbation = perturbation * torch.tensor(CLIP_STD).view(3, 1, 1)

    import numpy as np
    from PIL import Image

    width, height = image.size
    perturbation = F.interpolate(
        perturbation.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
    ).squeeze(0)

    img_arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    perturbed_arr = np.nan_to_num(img_arr + perturbation.permute(1, 2, 0).numpy(), nan=0.0, posinf=1.0, neginf=0.0)
    perturbed_arr = np.clip(perturbed_arr, 0.0, 1.0)
    perturbed_image = Image.fromarray((perturbed_arr * 255.0).astype(np.uint8))

    proc_out = qwen_processor.image_processor(images=[perturbed_image], return_tensors="pt")
    return proc_out["pixel_values"][0], proc_out["image_grid_thw"]


def _insert_caption_tokens(input_ids, cap_ids, image_token_id, vision_end_token_id):
    """Insert caption token ids right after each ``<|vision_end|>``.

    Returns (modified_input_ids, insertion_points) where insertion_points are
    the positions (in the modified sequence) right after each image block.
    """
    ids = input_ids[0].tolist()
    cap_list = cap_ids[0].tolist()
    modified, points = [], []
    for pos, token in enumerate(ids):
        modified.append(token)
        if token == vision_end_token_id:
            modified.extend(cap_list)
            points.append(len(modified))
    return (
        torch.tensor([modified], dtype=input_ids.dtype, device=input_ids.device),
        points,
    )


def _qwen2vl_patched_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    rope_deltas=None,
    cache_position=None,
    cap_tensor=None,
    the=None,
    gamma_gain=None,
    gamma_reduce=None,
    input_cap_ids=None,
    gain_per=None,
    reduce_per=None,
    use_cd_branch=None,
    bias_weight=None,
    bias_sample_num=None,
    **kwargs,
):
    from transformers.modeling_outputs import BaseModelOutputWithPast

    if pixel_values is None or inputs_embeds is not None:
        return self._shield_original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
        )

    vision_tower = self.visual
    pixel_values = pixel_values.type(vision_tower.get_dtype())
    image_embeds = vision_tower(pixel_values, grid_thw=image_grid_thw)

    if use_cd_branch:
        top_k_indices = None
        enhanced = image_embeds
        modified_input_ids = input_ids
    else:
        bias_tensor = get_qwen2vl_bias(
            int(bias_sample_num), pixel_values, image_grid_thw, vision_tower
        )
        enhanced, top_k_indices = compute_shield_image_weights(
            image_embeds.unsqueeze(0),
            cap_tensor,
            the,
            gamma_gain,
            gain_per,
            bias_weight,
            bias_tensor,
        )
        enhanced = enhanced.squeeze(0)

        if top_k_indices is not None and top_k_indices.numel() > 0:
            modified_input_ids, _ = _insert_caption_tokens(
                input_ids, input_cap_ids,
                self.config.image_token_id, self.config.vision_end_token_id,
            )
        else:
            modified_input_ids = input_ids

    inputs_embeds = self.model.embed_tokens(modified_input_ids)
    image_mask = (
        (modified_input_ids == self.config.image_token_id)
        .unsqueeze(-1)
        .expand_as(inputs_embeds)
        .to(inputs_embeds.device)
    )
    enhanced = enhanced.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, enhanced)

    if attention_mask is not None:
        pad_left = torch.ones(
            (attention_mask.shape[0], inputs_embeds.shape[1] - attention_mask.shape[1]),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat((pad_left, attention_mask), dim=1)

    if position_ids is None:
        position_ids, rope_deltas = self.get_rope_index(
            modified_input_ids, image_grid_thw, video_grid_thw, attention_mask
        )
        self.rope_deltas = rope_deltas

    outputs = self.model(
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=None,
    )

    logits = self.lm_head(outputs[0])

    if not return_dict:
        return (logits,) + outputs[1:]
    return BaseModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def _qwen2vl_patched_prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    **kwargs,
):
    if past_key_values is not None:
        input_ids = input_ids[:, -1:]
        pixel_values = None
        image_grid_thw = None
    else:
        pixel_values = kwargs.get("images", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        attention_mask = kwargs.get("attention_mask", attention_mask)

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": None,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "cap_tensor": kwargs.get("cap_tensor", None),
        "the": kwargs.get("the", None),
        "gamma_gain": kwargs.get("gamma_gain", None),
        "gamma_reduce": kwargs.get("gamma_reduce", None),
        "input_cap_ids": kwargs.get("input_cap_ids", None),
        "gain_per": kwargs.get("gain_per", None),
        "reduce_per": kwargs.get("reduce_per", None),
        "use_cd_branch": False,
        "bias_weight": kwargs.get("bias_weight", None),
        "bias_sample_num": kwargs.get("bias_sample_num", None),
    }
    return model_inputs


def _qwen2vl_patched_prepare_inputs_for_generation_cd(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    **kwargs,
):
    if past_key_values is not None:
        input_ids = input_ids[:, -1:]
        pixel_values = None
        image_grid_thw = None
    else:
        pixel_values = kwargs.get("images_cd", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        attention_mask = kwargs.get("attention_mask", attention_mask)

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": None,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "cap_tensor": kwargs.get("cap_tensor", None),
        "the": kwargs.get("the", None),
        "gamma_gain": kwargs.get("gamma_gain", None),
        "gamma_reduce": kwargs.get("gamma_reduce", None),
        "input_cap_ids": kwargs.get("input_cap_ids", None),
        "gain_per": kwargs.get("gain_per", None),
        "reduce_per": kwargs.get("reduce_per", None),
        "use_cd_branch": True,
        "bias_weight": kwargs.get("bias_weight", None),
        "bias_sample_num": kwargs.get("bias_sample_num", None),
    }
    return model_inputs


def _qwen2vl_shield_prepare(self, image, pixel_values, image_file, image_grid_thw, use_cd=True, **overrides):
    """Prepare all SHIELD inputs for a single Qwen2-VL image.

    Returns a ``dict`` that can be unpacked directly into
    ``model.generate(input_ids, **shield_kw, ...)``.
    """
    cfg = self._shield_config
    params = {**cfg["defaults"], **overrides}

    captions = cfg["captions"]
    caption_raw = find_text_by_image(image_file, captions)
    text_in, text_ad = process_caption(caption_raw)

    input_cap_ids = cfg["tokenizer"](text_in, return_tensors="pt")["input_ids"].cuda()
    cap_tensor = get_clip_text_features(text_in, cfg["clip_model"], cfg["clip_processor"])

    if use_cd:
        pixel_values_cd, grid_cd = qwen2vl_clip_attack(
            image,
            cfg["qwen_processor"],
            text_ad,
            epsilon=params["epsilon"],
            num_steps=int(params["num_steps"]),
            c=params["c"],
            lr=params["lr"],
            attack_type=params.get("attack_type", "cw"),
            clip_model=cfg["clip_model"],
            clip_processor=cfg["clip_processor"],
        )
        images_cd = pixel_values_cd.unsqueeze(0).half().cuda()
        image_grid_thw = grid_cd.to(image_grid_thw.device) if image_grid_thw is not None else grid_cd.cuda()
    else:
        images_cd = None

    return {
        "images": pixel_values,
        "images_cd": images_cd,
        "cd_alpha": params["cd_alpha"],
        "cd_beta": params["cd_beta"],
        "cap_tensor": cap_tensor,
        "input_cap_ids": input_cap_ids,
        "the": params["the"],
        "gamma_gain": params["gamma_gain"],
        "gamma_reduce": params["gamma_reduce"],
        "gain_per": params["gain_per"],
        "reduce_per": params["reduce_per"],
        "bias_weight": params["bias_weight"],
        "bias_sample_num": params["bias_sample_num"],
        "image_grid_thw": image_grid_thw,
    }


def wrap_qwen2vl(model, tokenizer, caption_file=None, qwen_processor=None, **kwargs):
    """Wrap a Qwen2-VL model with SHIELD capabilities.

    After calling this function the model gains:

    * ``model.shield_prepare(image, image_file, image_grid_thw)``
      -- returns a dict of kwargs ready to be unpacked into ``model.generate()``.
    * Monkey-patching of ``forward`` and the generation input preparation.

    Parameters
    ----------
    model : nn.Module
        A ``Qwen2VLForConditionalGeneration`` instance.
    tokenizer : PreTrainedTokenizer
        The tokenizer paired with *model*.
    caption_file : str, optional
        Path to a JSONL caption file.
    qwen_processor : AutoProcessor
        The Qwen2-VL processor (needed to re-process attacked images).
    **kwargs
        Default SHIELD hyper-parameters.
    """
    from transformers import CLIPModel, CLIPProcessor

    clip_model_name = kwargs.pop("clip_model_name", "openai/clip-vit-large-patch14-336")
    clip_model = CLIPModel.from_pretrained(clip_model_name, torch_dtype=torch.float16).cuda().eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    defaults = {
        "cd_alpha": 1.0,
        "cd_beta": 0.1,
        "the": 0.0,
        "gamma_gain": 1.0,
        "gamma_reduce": 0.0,
        "gain_per": 0.0,
        "reduce_per": 0.0,
        "bias_weight": 0.0,
        "bias_sample_num": 32,
        "epsilon": 0.14,
        "num_steps": 30,
        "c": 12,
        "lr": 0.02,
        "attack_type": "cw",
    }
    defaults.update(kwargs)

    model._shield_config = {
        "captions": load_captions(caption_file) if caption_file else None,
        "tokenizer": tokenizer,
        "clip_model": clip_model,
        "clip_processor": clip_processor,
        "qwen_processor": qwen_processor,
        "defaults": defaults,
    }

    model._shield_original_forward = type(model).forward

    model.forward = types.MethodType(_qwen2vl_patched_forward, model)
    model.prepare_inputs_for_generation = types.MethodType(
        _qwen2vl_patched_prepare_inputs_for_generation, model
    )
    model.prepare_inputs_for_generation_cd = types.MethodType(
        _qwen2vl_patched_prepare_inputs_for_generation_cd, model
    )
    model.shield_prepare = types.MethodType(_qwen2vl_shield_prepare, model)
    model.sample = types.MethodType(sampling.sample, model)
    model.greedy_search = types.MethodType(sampling.greedy_search, model)

    sampling.enable_shield_sampling()

    return model
