"""Helpers for loading Qwen3-VL models with action special tokens."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def _detach_vision_feature_outputs(
    image_embeds: tuple[torch.Tensor, ...] | torch.Tensor,
    deepstack_image_embeds: list[torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...] | torch.Tensor, list[torch.Tensor]]:
    if isinstance(image_embeds, tuple):
        image_embeds = tuple(tensor.detach() for tensor in image_embeds)
    else:
        image_embeds = image_embeds.detach()
    deepstack_image_embeds = [tensor.detach() for tensor in deepstack_image_embeds]
    return image_embeds, deepstack_image_embeds


def _wrap_vision_feature_fn(
    feature_fn: Callable[..., tuple[Any, list[torch.Tensor]]],
) -> Callable[..., tuple[Any, list[torch.Tensor]]]:
    def wrapped(*args, **kwargs):
        with torch.no_grad():
            image_embeds, deepstack_image_embeds = feature_fn(*args, **kwargs)
        return _detach_vision_feature_outputs(image_embeds, deepstack_image_embeds)

    return wrapped


def enable_frozen_vision_no_grad(model) -> None:
    """Run vision feature extraction under no_grad when the vision tower is frozen."""
    backbone = getattr(model, "model", model)
    if not hasattr(backbone, "get_image_features"):
        raise AttributeError("Cannot find get_image_features on Qwen3-VL backbone")

    if getattr(backbone, "_frozen_vision_no_grad_enabled", False):
        return

    backbone.get_image_features = _wrap_vision_feature_fn(backbone.get_image_features)
    if hasattr(backbone, "get_video_features"):
        backbone.get_video_features = _wrap_vision_feature_fn(backbone.get_video_features)
    backbone._frozen_vision_no_grad_enabled = True


def prepare_qwen3vl_processor_and_model(
    model_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool = True,
    num_action_tokens: int = 256,
    attn_implementation: str | None = None,
):
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_files_only)
    special_tokens = [f"<action_{i}>" for i in range(num_action_tokens)]
    existing = set(processor.tokenizer.additional_special_tokens or [])
    tokens_to_add = [token for token in special_tokens if token not in existing]
    added = 0
    if tokens_to_add:
        added = processor.tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "local_files_only": local_files_only,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    if len(processor.tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(processor.tokenizer))

    vocab_size = model.get_input_embeddings().weight.shape[0]
    assert len(processor.tokenizer) == vocab_size, (
        f"Tokenizer length {len(processor.tokenizer)} != embedding rows {vocab_size}"
    )
    return processor, model, added
