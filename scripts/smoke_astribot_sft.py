#!/usr/bin/env python3
"""Smoke tests for astribot VLA SFT pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from verl.utils.dataset.astribot_lerobot_sft_dataset import (
    ASTRIBOT_IMAGE_SIZES,
    AstribotLeRobotSFTDataset,
    denormalize_vector,
    normalize_vector,
)
from verl.utils.dataset.vla_sft_collate import vla_sft_collate_fn
from verl.utils.vla_model_utils import prepare_qwen3vl_processor_and_model
from verl.workers.actor.action_tokenizer import ActionTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="/media/damoxing/datasets/vae4d/lerobot-vae4d-org/astribot/astribot_filter_300",
    )
    parser.add_argument("--model-path", default="/media/damoxing/ckp/qwen_ft/Qwen3-VL-2B-Instruct")
    parser.add_argument("--statistics-path", default="assets/astribot_dataset_statistics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    stats_path = repo_root / args.statistics_path

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)["astribot_centrifuge_multidrop"]

    processor, model, added = prepare_qwen3vl_processor_and_model(
        args.model_path, local_files_only=True, attn_implementation="sdpa"
    )
    print(f"tokenizer/model vocab aligned, added_tokens={added}")
    action_tokenizer = ActionTokenizer(processor.tokenizer, need_to_sub=3)

    dataset = AstribotLeRobotSFTDataset(
        dataset_root=args.dataset_root,
        processor=processor,
        statistics_path=stats_path,
        dataset_name="astribot_centrifuge_multidrop",
        image_keys=[
            "observation.images.cam_main",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ],
        action_tokenizer=action_tokenizer,
        action_token_len=16,
        action_chunks_len=8,
        action_frame_stride=4,
        image_sizes=ASTRIBOT_IMAGE_SIZES,
        episode_indices=[0],
    )
    sample = dataset[0]
    print("prompt_len", sample["input_ids_prompt"].shape[0])
    print("action_token_ids.shape", tuple(sample["action_token_ids"].shape))
    print("action_frame_stride", dataset.action_frame_stride)
    chunk_indices = [sample["frame_index"] + i * dataset.action_frame_stride for i in range(8)]
    print("action chunk frame indices", chunk_indices)

    batch = vla_sft_collate_fn(
        [sample],
        pad_token_id=processor.tokenizer.pad_token_id,
        max_prompt_length=576,
    )
    print("batch input_ids", tuple(batch["input_ids"].shape))
    assert int(batch["input_ids"].max()) < model.get_input_embeddings().weight.shape[0]

    mask = np.array(stats["action"]["mask"], dtype=bool)
    q01 = np.array(stats["action"]["q01"], dtype=np.float32)
    q99 = np.array(stats["action"]["q99"], dtype=np.float32)
    raw_action = np.stack(dataset._load_episode_df(0)["action"].to_numpy()).astype(np.float32)[0]
    norm = normalize_vector(raw_action, mask, q01, q99)
    decoded = action_tokenizer.decode_token_ids_to_actions(action_tokenizer.action_to_token_ids(norm))
    unnorm = denormalize_vector(decoded, mask, q01, q99)
    l1 = float(np.mean(np.abs(unnorm[mask] - raw_action[mask])))
    print("action roundtrip masked L1", l1)

    model = model.to(args.device)
    batch = {k: v.to(args.device) for k, v in batch.items()}
    with torch.no_grad():
        with torch.autocast(device_type=args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
                astribot_parallel_action=True,
                prompt_length=576,
                action_length=128,
                action_slot_id=action_tokenizer.action_0_id,
            )
    local_logits = out.action_logits[..., action_tokenizer.action_0_id : action_tokenizer.action_0_id + 256]
    print("parallel action logits", tuple(local_logits.shape), "finite", bool(torch.isfinite(local_logits).all()))
    assert local_logits.shape == (1, 128, 256)
    assert torch.isfinite(local_logits).all()
    print("smoke tests passed")


if __name__ == "__main__":
    main()
