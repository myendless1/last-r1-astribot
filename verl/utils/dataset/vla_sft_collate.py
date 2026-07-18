"""Collate function for VLA SFT batches."""

from __future__ import annotations

from typing import Any

import torch


def vla_sft_collate_fn(
    batch: list[dict[str, Any]],
    pad_token_id: int,
    max_prompt_length: int,
) -> dict[str, torch.Tensor]:
    max_prompt_len = max_prompt_length
    action_chunks_len, action_token_len = batch[0]["action_token_ids"].shape
    action_len = action_chunks_len * action_token_len
    total_len = max_prompt_len + action_len

    batch_size = len(batch)
    input_ids = torch.full((batch_size, total_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, total_len, dtype=torch.long)

    pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
    image_grid_thw = torch.cat([item["image_grid_thw"] for item in batch], dim=0)
    labels = torch.full((batch_size, total_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        prompt_ids = item["input_ids_prompt"]
        if prompt_ids.shape[0] > max_prompt_len:
            raise ValueError(
                f"Prompt length {prompt_ids.shape[0]} exceeds max_prompt_length={max_prompt_len}"
            )
        prompt_attn = item["attention_mask_prompt"]
        prompt_len = prompt_ids.shape[0]
        # Right padding avoids fully-masked leading query rows in SDPA. Action
        # slots are created separately by the model after this fixed prompt area.
        input_ids[i, :prompt_len] = prompt_ids
        attention_mask[i, :prompt_len] = prompt_attn

        action_ids = item["action_token_ids"].reshape(-1)
        action_mask = item["action_loss_mask"].reshape(-1)
        action_start = max_prompt_len
        action_end = action_start + action_len

        # Action tokens are supervision only. Parallel LaST-R1 decoding creates
        # zero action-slot embeddings inside the model, so GT actions must never
        # be copied into input_ids or attention_mask.
        label_action = action_ids.clone()
        label_action[action_mask == 0] = -100
        labels[i, action_start:action_end] = label_action

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
