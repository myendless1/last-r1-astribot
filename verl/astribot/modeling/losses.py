from __future__ import annotations

import torch
import torch.nn.functional as F


def latent_cosine_loss(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if predicted.shape != target.shape:
        raise ValueError(f"Latent shape mismatch: {predicted.shape} != {target.shape}")
    loss = 1 - F.cosine_similarity(predicted.float(), target.float(), dim=-1)
    weights = mask.to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1)


def parallel_action_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, action_0_id: int) -> torch.Tensor:
    if logits.shape[:2] != labels.shape or labels.shape != mask.shape:
        raise ValueError(f"Action shape mismatch: logits={logits.shape}, labels={labels.shape}, mask={mask.shape}")
    local_labels = labels - int(action_0_id)
    losses = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), local_labels.reshape(-1), reduction="none")
    weights = mask.reshape(-1).to(losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1)


def latent_end_loss(logits: torch.Tensor | None, latent_end_id: int) -> torch.Tensor:
    if logits is None:
        raise RuntimeError("Model did not return latent_end_logits; enable return_latent_end_logits for SFT")
    target = torch.full((logits.shape[0],), int(latent_end_id), device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits.float(), target)
