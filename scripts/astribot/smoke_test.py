#!/usr/bin/env python3
from __future__ import annotations

import argparse

from omegaconf import OmegaConf
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from verl.astribot.data.lerobot_dataset import AstribotLeRobotDataset
from verl.astribot.protocol import AstribotProtocol
from verl.workers.actor.action_tokenizer import ActionTokenizer


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    cfg = OmegaConf.load(args.config); processor = AutoProcessor.from_pretrained(cfg.model.partial_pretrain, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(cfg.model.partial_pretrain, local_files_only=True)
    protocol = AstribotProtocol(dimension_scope=cfg.robot.dimension_scope, action_horizon=cfg.data.action_horizon, latent_length=cfg.model.latent_length, max_prompt_length=cfg.data.max_prompt_length, action_frame_stride=cfg.data.action_frame_stride, future_frame_stride=cfg.data.future_frame_stride)
    dataset = AstribotLeRobotDataset(root=cfg.data.dataset_root, processor=processor, action_tokenizer=ActionTokenizer(processor.tokenizer),
        dimension_scope=protocol.dimension_scope, norm_path=cfg.data.normalization_path, latent_cache_path=cfg.data.latent_cache_path,
        hidden_size=model.config.text_config.hidden_size, action_horizon=protocol.action_horizon,
        action_frame_stride=cfg.data.action_frame_stride, future_frame_stride=cfg.data.future_frame_stride, latent_length=protocol.latent_length, image_size_hw=cfg.data.image_size_hw)
    sample = dataset[0]
    assert sample["action_token_ids"].shape == (protocol.action_horizon, protocol.action_dim)
    assert sample["latent_gt_embeds"].shape == (protocol.latent_length, model.config.text_config.hidden_size)
    print("Astribot dataset smoke test passed", protocol.to_dict())


if __name__ == "__main__": main()
