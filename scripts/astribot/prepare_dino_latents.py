#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from verl.astribot.data.embodiment import IMAGE_KEYS
from verl.astribot.data.image_composition import compose_t_layout_rgb
from verl.astribot.data.lerobot_dataset import _decode_image, dataset_fingerprint


def _read_views(root: Path, path: Path, row: int, row_id: int | None) -> list[np.ndarray]:
    columns = list(IMAGE_KEYS)
    if row_id is None:
        table = pq.read_table(path, columns=columns, memory_map=True)
        image_row = row
    else:
        table = pq.read_table(path, columns=["index", *columns], filters=[("index", "=", row_id)], memory_map=True)
        if len(table) != 1:
            raise KeyError(f"Expected one image row for index={row_id}, got {len(table)}")
        image_row = 0
    return [_decode_image(table.column(key)[image_row].as_py(), root) for key in IMAGE_KEYS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dino-path", required=True)
    parser.add_argument("--latent-length", type=int, default=8)
    parser.add_argument("--future-frame-stride", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    processor = AutoImageProcessor.from_pretrained(args.dino_path, local_files_only=True)
    model = AutoModel.from_pretrained(args.dino_path, local_files_only=True).to(args.device).eval()
    episodes = defaultdict(list)
    for path in sorted((args.root / "data").glob("**/*.parquet")):
        schema = set(pq.ParquetFile(path).schema_arrow.names)
        columns = ["episode_index", "frame_index"] + (["index"] if "index" in schema else [])
        table = pq.read_table(path, columns=columns, memory_map=True)
        for row, episode in enumerate(table.column("episode_index").to_pylist()):
            frame = int(table.column("frame_index")[row].as_py())
            row_id = int(table.column("index")[row].as_py()) if "index" in table.column_names else None
            episodes[int(episode)].append((frame, path, row, row_id))

    args.output.mkdir(parents=True, exist_ok=True)
    for episode, rows in sorted(episodes.items()):
        rows.sort(key=lambda item: item[0])
        encoded = []
        for start in range(0, len(rows), args.batch_size):
            images = []
            for _, path, row, row_id in rows[start : start + args.batch_size]:
                views = _read_views(args.root, path, row, row_id)
                images.append(Image.fromarray(compose_t_layout_rgb(*views)))
            inputs = processor(images=images, return_tensors="pt").to(args.device)
            with torch.inference_mode():
                cls = model(**inputs).last_hidden_state[:, 0].float()
            if cls.shape[-1] < args.hidden_size:
                raise ValueError(f"DINO hidden size {cls.shape[-1]} < {args.hidden_size}")
            indices = cls.abs().topk(args.hidden_size, dim=-1).indices
            encoded.append(torch.gather(cls, -1, indices).cpu().numpy())

        frame_latents = np.concatenate(encoded, axis=0)
        output = np.zeros((len(rows), args.latent_length, args.hidden_size), np.float16)
        mask = np.zeros((len(rows), args.latent_length), np.uint8)
        for frame in range(len(rows)):
            for slot in range(args.latent_length):
                future = frame + (slot + 1) * args.future_frame_stride
                if future < len(rows):
                    output[frame, slot] = frame_latents[future]
                    mask[frame, slot] = 1
        np.savez_compressed(args.output / f"episode_{episode:06d}.npz", latents=output, mask=mask)

    metadata = {
        "dataset_fingerprint": dataset_fingerprint(args.root),
        "dino_path": args.dino_path,
        "latent_length": args.latent_length,
        "hidden_size": args.hidden_size,
        "future_frame_stride": args.future_frame_stride,
        "image_layout": "t_layout",
        "storage_dtype": "float16",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
