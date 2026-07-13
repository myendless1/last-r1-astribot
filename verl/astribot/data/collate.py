from __future__ import annotations

import torch


def astribot_sft_collate(batch: list[dict], *, pad_token_id: int, latent_pad_id: int, latent_end_id: int, max_prompt_length: int) -> dict[str, torch.Tensor]:
    latent_length = batch[0]["latent_gt_embeds"].shape[0]
    horizon, action_dim = batch[0]["action_token_ids"].shape
    action_length = horizon * action_dim
    total_length = max_prompt_length + latent_length + 1 + action_length
    size = len(batch)
    input_ids = torch.full((size, total_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((size, total_length), dtype=torch.long)
    action_labels = torch.empty((size, action_length), dtype=torch.long)
    action_mask = torch.empty((size, action_length), dtype=torch.float32)
    for index, item in enumerate(batch):
        prompt = item["input_ids_prompt"]
        if len(prompt) > max_prompt_length:
            raise ValueError(f"Prompt length {len(prompt)} exceeds {max_prompt_length}")
        start = max_prompt_length - len(prompt)
        input_ids[index, start:max_prompt_length] = prompt
        attention_mask[index, start:max_prompt_length] = 1
        input_ids[index, max_prompt_length : max_prompt_length + latent_length] = latent_pad_id
        input_ids[index, max_prompt_length + latent_length] = latent_end_id
        # Never expose ground-truth actions through input embeddings.
        input_ids[index, -action_length:] = latent_pad_id
        attention_mask[index, max_prompt_length:] = 1
        action_labels[index] = item["action_token_ids"].reshape(-1)
        action_mask[index] = item["action_loss_mask"].reshape(-1)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": torch.cat([item["pixel_values"] for item in batch]),
        "image_grid_thw": torch.cat([item["image_grid_thw"] for item in batch]),
        "latent_gt_embeds": torch.stack([item["latent_gt_embeds"] for item in batch]),
        "latent_mask": torch.stack([item["latent_mask"] for item in batch]),
        "action_labels": action_labels,
        "action_mask": action_mask,
    }
