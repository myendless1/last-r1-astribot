from types import SimpleNamespace

import torch

from verl.astribot.modeling.parallel_sft import parallel_action_forward
from verl.utils.dataset.vla_sft_collate import vla_sft_collate_fn


class _FakeBackbone:
    def __init__(self):
        self.embedding = torch.nn.Embedding(512, 8)
        self.last_kwargs = None

    def get_image_features(self, pixel_values, image_grid_thw):
        return (pixel_values,), []

    def get_input_embeddings(self):
        return self.embedding

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"])


class _FakeModel:
    def __init__(self):
        self.model = _FakeBackbone()
        self.lm_head = torch.nn.Linear(8, 512, bias=False)


def test_parallel_action_uses_zero_slots_and_ignores_action_label_inputs():
    model = _FakeModel()
    prompt_length = 4
    action_length = 3
    common = {
        "attention_mask": torch.ones(1, prompt_length + action_length, dtype=torch.long),
        "pixel_values": torch.zeros(1, 8),
        "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
    }
    first = parallel_action_forward(
        model,
        {**common, "input_ids": torch.tensor([[1, 2, 3, 4, 10, 11, 12]])},
        prompt_length=prompt_length,
        action_length=action_length,
        action_slot_id=100,
    )
    second = parallel_action_forward(
        model,
        {**common, "input_ids": torch.tensor([[1, 2, 3, 4, 20, 21, 22]])},
        prompt_length=prompt_length,
        action_length=action_length,
        action_slot_id=100,
    )

    assert first.action_logits.shape == (1, action_length, 512)
    assert torch.equal(first.action_logits, second.action_logits)
    assert torch.count_nonzero(model.model.last_kwargs["inputs_embeds"][:, -action_length:]) == 0
    assert model.model.last_kwargs["action_length"] == action_length
    assert model.model.last_kwargs["attn_mode"] == "causal"


def test_parallel_collate_keeps_action_labels_out_of_model_inputs():
    item = {
        "input_ids_prompt": torch.tensor([7, 8]),
        "attention_mask_prompt": torch.ones(2, dtype=torch.long),
        "pixel_values": torch.zeros(1, 4),
        "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
        "action_token_ids": torch.tensor([[101, 102], [103, 104]]),
        "action_loss_mask": torch.ones(2, 2),
    }
    batch = vla_sft_collate_fn([item], pad_token_id=0, max_prompt_length=4)

    assert torch.equal(batch["input_ids"][0, :4], torch.tensor([7, 8, 0, 0]))
    assert torch.equal(batch["attention_mask"][0, :4], torch.tensor([1, 1, 0, 0]))
    assert torch.equal(batch["input_ids"][0, 4:], torch.zeros(4, dtype=torch.long))
    assert torch.equal(batch["attention_mask"][0, 4:], torch.zeros(4, dtype=torch.long))
    assert torch.equal(batch["labels"][0, 4:], torch.tensor([101, 102, 103, 104]))
