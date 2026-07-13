from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

from verl.workers.actor.action_tokenizer import ActionTokenizer

from .embodiment import ACTION_KEY, IMAGE_KEYS, RAW_DIM, STATE_KEY, select_astribot_dims
from .image_composition import preprocess_t_layout
from .latent_cache import LatentCache
from .normalization import FeatureNormalizer


def dataset_fingerprint(root: str | Path) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    for relative in ("meta/info.json", "meta/tasks.parquet"):
        path = root / relative
        if path.exists():
            digest.update(path.read_bytes())
    for path in sorted((root / "data").glob("**/*.parquet")):
        stat = path.stat()
        digest.update(f"{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def _sort_key(path: Path):
    return [int(value) if value.isdigit() else value for value in re.split(r"(\d+)", str(path))]


def _column_numpy(table, key: str, dtype):
    column = table.column(key).combine_chunks()
    list_size = getattr(column.type, "list_size", None)
    if list_size is not None:
        return np.asarray(column.values.to_numpy(zero_copy_only=False), dtype=dtype).reshape(len(column), int(list_size))
    values = column.to_pylist()
    return np.stack(values).astype(dtype)


def _decode_image(cell: Any, root: Path) -> np.ndarray:
    if isinstance(cell, dict):
        if cell.get("bytes"):
            image = cv2.imdecode(np.frombuffer(cell["bytes"], np.uint8), cv2.IMREAD_COLOR)
        elif cell.get("path"):
            path = Path(cell["path"])
            image = cv2.imread(str(path if path.is_absolute() else root / path))
        else:
            image = None
    else:
        image = cv2.imdecode(np.frombuffer(cell, np.uint8), cv2.IMREAD_COLOR) if isinstance(cell, (bytes, bytearray)) else None
    if image is None:
        raise ValueError("Could not decode LeRobot image cell")
    return image[..., ::-1].copy()


class AstribotLeRobotDataset(Dataset):
    def __init__(self, *, root: str | Path, processor, action_tokenizer: ActionTokenizer, dimension_scope: str,
                 norm_path: str | Path, latent_cache_path: str | Path, hidden_size: int, action_horizon: int = 8,
                 action_frame_stride: int = 1, future_frame_stride: int = 1, latent_length: int = 8, image_size_hw=(384, 320), episode_indices=None):
        self.root = Path(root)
        self.processor = processor
        self.action_tokenizer = action_tokenizer
        self.dimension_scope = dimension_scope
        self.action_horizon = int(action_horizon)
        self.action_frame_stride = int(action_frame_stride)
        self.future_frame_stride = int(future_frame_stride)
        self.image_size_hw = tuple(image_size_hw)
        self.normalizer = FeatureNormalizer(norm_path, expected_scope=dimension_scope, expected_metadata={"action_horizon": self.action_horizon, "action_frame_stride": self.action_frame_stride})
        self.fingerprint = dataset_fingerprint(root)
        self.latent_cache = LatentCache(latent_cache_path, latent_length=latent_length, hidden_size=hidden_size, dataset_fingerprint=self.fingerprint, expected_future_frame_stride=self.future_frame_stride)
        self.files = sorted((self.root / "data").glob("**/*.parquet"), key=_sort_key)
        if not self.files:
            raise FileNotFoundError(f"No parquet files under {self.root / 'data'}")
        self.rows: list[tuple[int, int, int]] = []
        self.tables = []
        self.has_index = []
        allowed = set(episode_indices) if episode_indices is not None else None
        for file_index, path in enumerate(self.files):
            schema = set(pq.ParquetFile(path).schema_arrow.names)
            optional = [name for name in ("index", "frame_index", "task", "task_index") if name in schema]
            table = pq.read_table(path, columns=["episode_index", STATE_KEY, ACTION_KEY, *optional], memory_map=True)
            episodes = np.asarray(table.column("episode_index").to_numpy(), dtype=np.int64)
            for local_index, episode in enumerate(episodes):
                if allowed is None or int(episode) in allowed:
                    self.rows.append((file_index, local_index, int(episode)))
            self.has_index.append("index" in schema)
            self.tables.append(table)
        tasks_path = self.root / "meta" / "tasks.parquet"
        self.tasks = pq.read_table(tasks_path).to_pandas() if tasks_path.exists() else None

    def __len__(self): return len(self.rows)

    def _task(self, table, row: int) -> str:
        if "task" in table.column_names:
            return str(table.column("task")[row].as_py())
        if "task_index" in table.column_names and self.tasks is not None:
            index = int(table.column("task_index")[row].as_py())
            return str(self.tasks.iloc[index]["task"])
        return ""

    def _images(self, file_index: int, row: int) -> dict[str, np.ndarray]:
        columns = list(IMAGE_KEYS)
        if self.has_index[file_index]:
            row_id = int(self.tables[file_index].column("index")[row].as_py())
            image_table = pq.read_table(
                self.files[file_index],
                columns=["index", *columns],
                filters=[("index", "=", row_id)],
                memory_map=True,
            )
            if len(image_table) != 1:
                raise KeyError(f"Expected one image row for index={row_id}, got {len(image_table)}")
            image_row = 0
        else:
            image_table = pq.read_table(self.files[file_index], columns=columns, memory_map=True)
            image_row = row
        return {
            key: _decode_image(image_table.column(key)[image_row].as_py(), self.root)
            for key in IMAGE_KEYS
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        file_index, row, episode = self.rows[index]
        table = self.tables[file_index]
        episode_rows = np.asarray(table.column("episode_index").to_numpy(), dtype=np.int64)
        future = [row + i * self.action_frame_stride for i in range(self.action_horizon)]
        valid = np.array([position < len(table) and episode_rows[position] == episode for position in future], dtype=np.float32)
        future = [position if ok else row for position, ok in zip(future, valid)]
        state_raw = np.asarray(table.column(STATE_KEY)[row].as_py(), np.float32)
        action_raw = np.stack([np.asarray(table.column(ACTION_KEY)[position].as_py(), np.float32) for position in future])
        if state_raw.shape != (RAW_DIM,) or action_raw.shape != (self.action_horizon, RAW_DIM):
            raise ValueError(f"Raw Astribot shape mismatch: {state_raw.shape}, {action_raw.shape}")
        state = self.normalizer.normalize(select_astribot_dims(state_raw, self.dimension_scope), STATE_KEY)
        action = self.normalizer.normalize(select_astribot_dims(action_raw, self.dimension_scope), ACTION_KEY)
        action_ids = self.action_tokenizer.action_to_token_ids(np.clip(action, -1, 1)).reshape(action.shape)
        images = self._images(file_index, row)
        layout = preprocess_t_layout(images, self.image_size_hw)
        pil = Image.fromarray((layout.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        state_text = self.action_tokenizer(np.clip(state, -1, 1))
        messages = [{"role": "user", "content": [{"type": "image", "image": pil}, {"type": "text", "text": self._task(table, row) + "\nState: " + state_text}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        frame_index = int(table.column("frame_index")[row].as_py()) if "frame_index" in table.column_names else row
        latent_gt, latent_mask = self.latent_cache.get(episode, frame_index)
        return {
            "input_ids_prompt": inputs.input_ids[0], "pixel_values": inputs.pixel_values,
            "image_grid_thw": inputs.image_grid_thw, "latent_gt_embeds": latent_gt, "latent_mask": latent_mask,
            "action_token_ids": torch.as_tensor(action_ids, dtype=torch.long),
            "action_loss_mask": torch.as_tensor(valid[:, None] * np.ones((1, action.shape[1])), dtype=torch.float32),
            "raw_action_chunk": torch.from_numpy(select_astribot_dims(action_raw, self.dimension_scope).copy()),
            "episode_index": episode, "frame_index": frame_index,
        }
