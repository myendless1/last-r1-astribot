#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from verl.astribot.data.collate import astribot_sft_collate
from verl.astribot.data.lerobot_dataset import AstribotLeRobotDataset
from verl.astribot.data.normalization import FeatureNormalizer
from verl.astribot.modeling.inference import predict_action_chunk
from verl.astribot.protocol import load_protocol
from verl.workers.actor.action_tokenizer import ActionTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm", type=Path, required=True)
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("eval_outputs/astribot_open_loop.json"))
    args = parser.parse_args()
    protocol = load_protocol(args.checkpoint / "astribot_protocol.json")
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.checkpoint, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa").to(args.device).eval()
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    normalizer = FeatureNormalizer(args.norm, expected_scope=protocol.dimension_scope)
    dataset = AstribotLeRobotDataset(root=args.dataset_root, processor=processor, action_tokenizer=action_tokenizer,
        dimension_scope=protocol.dimension_scope, norm_path=args.norm, latent_cache_path=args.latent_cache,
        hidden_size=model.config.text_config.hidden_size, action_horizon=protocol.action_horizon, action_frame_stride=protocol.action_frame_stride,
        future_frame_stride=protocol.future_frame_stride, latent_length=protocol.latent_length, image_size_hw=(protocol.image_height, protocol.image_width))
    count = min(args.samples, len(dataset)); records = []
    for index in np.linspace(0, len(dataset) - 1, count, dtype=int):
        sample = dataset[int(index)]
        batch = astribot_sft_collate([sample], pad_token_id=processor.tokenizer.pad_token_id,
            latent_pad_id=processor.tokenizer.convert_tokens_to_ids("<latent_pad>"),
            latent_end_id=processor.tokenizer.convert_tokens_to_ids("<latent_end>"), max_prompt_length=protocol.max_prompt_length)
        batch = {key: value.to(args.device) for key, value in batch.items()}
        pred, latents = predict_action_chunk(model, batch, protocol=protocol, tokenizer=processor.tokenizer, normalizer=normalizer)
        gt = sample["raw_action_chunk"].numpy(); valid = sample["action_loss_mask"].numpy() > 0
        records.append({"index": int(index), "mae": float(np.abs(pred[valid] - gt[valid]).mean()),
                        "latent_cosine": float(torch.nn.functional.cosine_similarity(latents.float(), batch["latent_gt_embeds"].float(), -1).mean().item())})
    summary = {"checkpoint": str(args.checkpoint), "protocol": protocol.to_dict(), "samples": records,
               "mean_mae": float(np.mean([item["mae"] for item in records]))}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
