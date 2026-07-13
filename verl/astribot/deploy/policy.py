from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from ..data.collate import astribot_sft_collate
from ..data.embodiment import STATE_KEY, select_astribot_dims
from ..data.image_composition import preprocess_t_layout
from ..data.normalization import FeatureNormalizer
from ..modeling.inference import predict_action_chunk
from ..protocol import load_protocol
from verl.workers.actor.action_tokenizer import ActionTokenizer


class AstribotPolicy:
    def __init__(self, checkpoint: str | Path, norm_path: str | Path, device: str = "cuda"):
        checkpoint = Path(checkpoint); self.protocol = load_protocol(checkpoint / "astribot_protocol.json")
        self.processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa").to(device).eval()
        self.normalizer = FeatureNormalizer(norm_path, expected_scope=self.protocol.dimension_scope, expected_metadata={"action_horizon": self.protocol.action_horizon, "action_frame_stride": self.protocol.action_frame_stride})
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer); self.device = device

    def metadata(self): return self.protocol.to_dict()

    def infer(self, observation: dict) -> dict:
        layout = preprocess_t_layout(observation["images"], (self.protocol.image_height, self.protocol.image_width),
            color_order=observation.get("image_color_order", "bgr"))
        image = Image.fromarray((layout.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        state = select_astribot_dims(np.asarray(observation["state"], np.float32), self.protocol.dimension_scope)
        state = self.normalizer.normalize(state, STATE_KEY)
        state_tokens = self.action_tokenizer(np.clip(state, -1, 1))
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
            {"type": "text", "text": str(observation.get("prompt", "")) + "\nState: " + state_tokens}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        placeholder = np.full((self.protocol.action_horizon, self.protocol.action_dim), self.action_tokenizer.action_0_id, np.int64)
        sample = {"input_ids_prompt": inputs.input_ids[0], "pixel_values": inputs.pixel_values, "image_grid_thw": inputs.image_grid_thw,
            "latent_gt_embeds": torch.zeros(self.protocol.latent_length, self.model.config.text_config.hidden_size),
            "latent_mask": torch.zeros(self.protocol.latent_length), "action_token_ids": torch.from_numpy(placeholder),
            "action_loss_mask": torch.ones_like(torch.from_numpy(placeholder), dtype=torch.float32)}
        batch = astribot_sft_collate([sample], pad_token_id=self.processor.tokenizer.pad_token_id,
            latent_pad_id=self.processor.tokenizer.convert_tokens_to_ids("<latent_pad>"),
            latent_end_id=self.processor.tokenizer.convert_tokens_to_ids("<latent_end>"), max_prompt_length=self.protocol.max_prompt_length)
        batch = {key: value.to(self.device) for key, value in batch.items()}
        actions, _ = predict_action_chunk(self.model, batch, protocol=self.protocol, tokenizer=self.processor.tokenizer, normalizer=self.normalizer)
        return {"actions": actions, "metadata": self.metadata()}
