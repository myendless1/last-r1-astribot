import torch

from transformers.integrations.sdpa_attention import build_mask_causal, change_mask_causal


def test_causal_mask_has_full_action_block_only():
    mask = build_mask_causal(8, 8, action_length=3, latent_length=2)
    assert mask.dtype == torch.bool
    assert mask[0, 0, -3:, -3:].all()
    assert not mask[0, 0, 1, 2]


def test_additive_mask_uses_zero_for_allowed_action_entries():
    mask = torch.full((1, 1, 8, 8), torch.finfo(torch.float32).min)
    mask = change_mask_causal(mask, action_length=3, latent_length=2)
    assert torch.all(mask[0, 0, -3:, -3:] == 0)
