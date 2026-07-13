from __future__ import annotations

import numpy as np
import torch

from verl.workers.actor.action_tokenizer import ActionTokenizer

from ..data.embodiment import ACTION_KEY
from ..protocol import AstribotProtocol


@torch.no_grad()
def predict_action_chunk(model, batch: dict[str, torch.Tensor], *, protocol: AstribotProtocol,
                         tokenizer, normalizer, temperature: float = 0.0) -> tuple[np.ndarray, torch.Tensor]:
    action_tokenizer = ActionTokenizer(tokenizer)
    latent_end_id = tokenizer.convert_tokens_to_ids("<latent_end>")
    result = model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"], image_grid_thw=batch["image_grid_thw"],
        astribot_parallel_sft=True, prompt_length=batch["input_ids"].shape[1] - protocol.latent_length - 1 - protocol.action_length,
        action_length=protocol.action_length, latent_length=protocol.latent_length,
        latent_end_id=latent_end_id,
    )
    latents = result.predicted_latents
    logits = result.action_logits[..., action_tokenizer.action_0_id : action_tokenizer.action_0_id + 256].float()
    if temperature > 0:
        distribution = torch.distributions.Categorical(logits=logits / temperature)
        token_ids = distribution.sample() + action_tokenizer.action_0_id
    else:
        token_ids = logits.argmax(-1) + action_tokenizer.action_0_id
    normalized = action_tokenizer.decode_token_ids_to_actions(token_ids[0].cpu().numpy())
    normalized = normalized.reshape(protocol.action_horizon, protocol.action_dim)
    actions = normalizer.denormalize(normalized, ACTION_KEY)
    return np.asarray(actions, dtype=np.float32), latents
