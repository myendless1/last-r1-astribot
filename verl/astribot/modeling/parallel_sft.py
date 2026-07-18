from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ParallelSFTOutput:
    action_logits: torch.Tensor
    predicted_latents: torch.Tensor
    latent_end_logits: torch.Tensor


@dataclass
class ParallelActionOutput:
    """Logits produced for every action slot in one non-autoregressive forward."""

    action_logits: torch.Tensor


def parallel_action_forward(
    model,
    batch: dict[str, torch.Tensor],
    *,
    prompt_length: int,
    action_length: int,
    action_slot_id: int,
) -> ParallelActionOutput:
    """Decode a complete action block in parallel, matching LaST-R1 rollout.

    Action tokens from the dataset are labels only: they are never fed back into
    the model.  Every action position receives a zero embedding and the custom
    ``attn_mode="causal"`` mask makes the action block internally bidirectional
    while keeping the prompt causal.
    """
    if prompt_length < 1:
        raise ValueError(f"prompt_length must be positive, got {prompt_length}")
    if action_length < 1:
        raise ValueError(f"action_length must be positive, got {action_length}")

    backbone = model.model
    prompt_ids = batch["input_ids"][:, :prompt_length]
    prompt_mask = batch["attention_mask"][:, :prompt_length]
    if prompt_ids.shape != prompt_mask.shape:
        raise ValueError(
            f"Prompt ids/mask shape mismatch: {prompt_ids.shape} != {prompt_mask.shape}"
        )

    image_features = backbone.get_image_features(batch["pixel_values"], batch["image_grid_thw"])
    prompt_embeddings = backbone.get_input_embeddings()(prompt_ids)
    action_slots = torch.zeros(
        prompt_embeddings.shape[0],
        action_length,
        prompt_embeddings.shape[-1],
        dtype=prompt_embeddings.dtype,
        device=prompt_embeddings.device,
    )
    inputs_embeds = torch.cat([prompt_embeddings, action_slots], dim=1)
    attention_mask = torch.cat(
        [
            prompt_mask,
            torch.ones(
                prompt_mask.shape[0],
                action_length,
                dtype=prompt_mask.dtype,
                device=prompt_mask.device,
            ),
        ],
        dim=1,
    )
    rope_input_ids = torch.cat(
        [
            prompt_ids,
            torch.full(
                (prompt_ids.shape[0], action_length),
                int(action_slot_id),
                dtype=prompt_ids.dtype,
                device=prompt_ids.device,
            ),
        ],
        dim=1,
    )

    outputs = backbone(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        pixel_values=None,
        image_grid_thw=batch["image_grid_thw"],
        astribot_image_features=image_features,
        rope_input_ids=rope_input_ids,
        return_dict=True,
        action_length=action_length,
        latent_length=0,
        attn_mode="causal",
        use_cache=False,
    )
    action_logits = model.lm_head(outputs.last_hidden_state[:, -action_length:])
    return ParallelActionOutput(action_logits=action_logits)


def parallel_sft_forward(
    model,
    batch: dict[str, torch.Tensor],
    *,
    prompt_length: int,
    latent_length: int,
    action_length: int,
    latent_end_id: int,
) -> ParallelSFTOutput:
    """Differentiable latent AR followed by one parallel action decode."""
    backbone = model.model
    image_features = backbone.get_image_features(batch["pixel_values"], batch["image_grid_thw"])
    prompt_ids = batch["input_ids"][:, :prompt_length]
    prompt_mask = batch["attention_mask"][:, :prompt_length]
    embeddings = backbone.get_input_embeddings()(prompt_ids)
    mask = prompt_mask
    rope_ids = prompt_ids
    predicted = []

    for _ in range(latent_length):
        outputs = backbone(
            input_ids=None,
            inputs_embeds=embeddings,
            attention_mask=mask,
            pixel_values=None,
            image_grid_thw=batch["image_grid_thw"],
            astribot_image_features=image_features,
            rope_input_ids=rope_ids,
            return_dict=True,
            action_length=0,
            use_cache=False,
        )
        next_latent = outputs.last_hidden_state[:, -1]
        predicted.append(next_latent)
        embeddings = torch.cat([embeddings, next_latent.unsqueeze(1)], dim=1)
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
        rope_ids = torch.cat([rope_ids, torch.full_like(rope_ids[:, :1], latent_end_id)], dim=1)

    end_ids = torch.full((embeddings.shape[0], 1), latent_end_id, dtype=torch.long, device=embeddings.device)
    end_embedding = backbone.get_input_embeddings()(end_ids)
    action_slots = torch.zeros(
        embeddings.shape[0], action_length, embeddings.shape[-1], dtype=embeddings.dtype, device=embeddings.device
    )
    embeddings = torch.cat([embeddings, end_embedding, action_slots], dim=1)
    mask = torch.cat(
        [mask, torch.ones(mask.shape[0], 1 + action_length, dtype=mask.dtype, device=mask.device)], dim=1
    )
    rope_ids = torch.cat(
        [
            rope_ids,
            torch.full(
                (rope_ids.shape[0], 1 + action_length),
                latent_end_id,
                dtype=rope_ids.dtype,
                device=rope_ids.device,
            ),
        ],
        dim=1,
    )
    outputs = backbone(
        input_ids=None,
        inputs_embeds=embeddings,
        attention_mask=mask,
        pixel_values=None,
        image_grid_thw=batch["image_grid_thw"],
        astribot_image_features=image_features,
        rope_input_ids=rope_ids,
        return_dict=True,
        action_length=action_length,
        latent_length=latent_length,
        attn_mode="causal",
        use_cache=False,
    )
    final_latent_index = prompt_length + latent_length - 1
    latent_end_logits = model.lm_head(outputs.last_hidden_state[:, final_latent_index])
    action_logits = model.lm_head(outputs.last_hidden_state[:, -action_length:])
    return ParallelSFTOutput(action_logits, torch.stack(predicted, dim=1), latent_end_logits)
