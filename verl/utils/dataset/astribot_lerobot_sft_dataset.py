"""LeRobot v2.1 astribot dataset for Qwen3-VL oneshot action SFT.

Action layout (16-D): left xyz + left quat (qw,qx,qy,qz) + left gripper +
right xyz + right quat + right gripper. Values are absolute EE poses at 50 Hz.

Gripper convention: larger values mean *open* (not closed). For example, left
gripper q01/q99 ~ 0.98 in dataset statistics indicates the left gripper is
usually open during demos; do not interpret high gripper values as "closed".
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TVF

from verl.workers.actor.action_tokenizer import ActionTokenizer


ASTRIBOT_IMAGE_SIZES: dict[str, tuple[int, int]] = {
    "observation.images.cam_main": (320, 256),
    "observation.images.cam_left_wrist": (160, 128),
    "observation.images.cam_right_wrist": (160, 128),
}


def center_crop_image(img: Image.Image, crop_scale: float = 0.9) -> Image.Image:
    width, height = img.size
    crop_h = int(height * crop_scale)
    crop_w = int(width * crop_scale)
    img = TVF.center_crop(img, [crop_h, crop_w])
    img = TVF.resize(img, [height, width])
    return img


def normalize_vector(
    values: np.ndarray,
    mask: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
) -> np.ndarray:
    return np.where(
        mask,
        np.clip(2 * (values - q01) / (q99 - q01 + 1e-8) - 1, -1, 1),
        values,
    )


def denormalize_vector(
    values: np.ndarray,
    mask: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
) -> np.ndarray:
    return np.where(mask, (values + 1) * 0.5 * (q99 - q01 + 1e-8) + q01, values)


@dataclass(frozen=True)
class FrameIndex:
    episode_index: int
    frame_index: int


class VideoFrameCache:
    def __init__(self, max_readers: int = 8):
        self.max_readers = max_readers
        self._readers: OrderedDict[tuple[int, str], VideoReader] = OrderedDict()

    def read_frame(self, video_path: Path, frame_index: int) -> np.ndarray:
        key = (id(video_path), str(video_path))
        if key not in self._readers:
            self._readers[key] = VideoReader(str(video_path), ctx=cpu(0))
            self._readers.move_to_end(key)
            while len(self._readers) > self.max_readers:
                self._readers.popitem(last=False)
        else:
            self._readers.move_to_end(key)
        frame = self._readers[key][frame_index].asnumpy()
        return frame


class EpisodeParquetCache:
    def __init__(self, max_episodes: int = 4):
        self.max_episodes = max_episodes
        self._cache: OrderedDict[int, pd.DataFrame] = OrderedDict()

    def get(self, data_root: Path, episode_index: int) -> pd.DataFrame:
        if episode_index in self._cache:
            self._cache.move_to_end(episode_index)
            return self._cache[episode_index]
        parquet_path = data_root / f"episode_{episode_index:06d}.parquet"
        df = pd.read_parquet(parquet_path)
        self._cache[episode_index] = df
        self._cache.move_to_end(episode_index)
        while len(self._cache) > self.max_episodes:
            self._cache.popitem(last=False)
        return df


def parse_image_sizes(
    image_keys: list[str],
    image_sizes: dict[str, list[int] | tuple[int, int]] | None,
    image_size: int | None,
) -> dict[str, tuple[int, int]]:
    """Resolve per-camera resize targets as (width, height) for PIL.Image.resize."""
    if image_sizes:
        resolved: dict[str, tuple[int, int]] = {}
        for key in image_keys:
            if key not in image_sizes:
                raise KeyError(f"image_sizes missing entry for camera key: {key}")
            width, height = image_sizes[key]
            resolved[key] = (int(width), int(height))
        return resolved
    if image_size is not None:
        return {key: (int(image_size), int(image_size)) for key in image_keys}
    return {}


class AstribotLeRobotSFTDataset(Dataset):
    VIDEO_ROOT_NAME = "videos_480x720_240x360"

    def __init__(
        self,
        dataset_root: str | Path,
        processor,
        statistics_path: str | Path,
        dataset_name: str,
        *,
        image_keys: list[str],
        action_tokenizer: ActionTokenizer,
        action_token_len: int = 16,
        action_chunks_len: int = 8,
        frame_stride: int = 1,
        action_frame_stride: int = 4,
        use_proprio: bool = True,
        center_crop: bool = True,
        image_size: int | None = None,
        image_sizes: dict[str, list[int] | tuple[int, int]] | None = None,
        episode_indices: list[int] | None = None,
        max_frames: int | None = None,
        subsample_seed: int = 42,
    ):
        self.dataset_root = Path(dataset_root)
        self.processor = processor
        self.dataset_name = dataset_name
        self.image_keys = image_keys
        self.action_tokenizer = action_tokenizer
        self.action_token_len = action_token_len
        self.action_chunks_len = action_chunks_len
        self.frame_stride = frame_stride
        if action_frame_stride < 1:
            raise ValueError(f"action_frame_stride must be >= 1, got {action_frame_stride}")
        self.action_frame_stride = action_frame_stride
        self.use_proprio = use_proprio
        self.center_crop = center_crop
        self.image_sizes = parse_image_sizes(image_keys, image_sizes, image_size)

        with Path(statistics_path).open("r", encoding="utf-8") as f:
            stats = json.load(f)[dataset_name]
        self.action_mask = np.array(stats["action"]["mask"], dtype=bool)
        self.action_q01 = np.array(stats["action"]["q01"], dtype=np.float32)
        self.action_q99 = np.array(stats["action"]["q99"], dtype=np.float32)
        self.state_mask = np.array(stats["state"]["mask"], dtype=bool)
        self.state_q01 = np.array(stats["state"]["q01"], dtype=np.float32)
        self.state_q99 = np.array(stats["state"]["q99"], dtype=np.float32)

        self.tasks = self._load_tasks()
        self.episode_lengths = self._load_episode_lengths()
        all_episodes = sorted(self.episode_lengths.keys())
        if episode_indices is None:
            self.episode_indices = all_episodes
        else:
            self.episode_indices = sorted(episode_indices)

        self.frame_indices: list[FrameIndex] = []
        for episode_index in self.episode_indices:
            length = self.episode_lengths[episode_index]
            for frame_index in range(0, length, self.frame_stride):
                self.frame_indices.append(FrameIndex(episode_index, frame_index))

        if max_frames is not None and len(self.frame_indices) > max_frames:
            rng = np.random.default_rng(subsample_seed)
            chosen = np.sort(rng.choice(len(self.frame_indices), size=max_frames, replace=False))
            self.frame_indices = [self.frame_indices[int(i)] for i in chosen]

        self.video_root = self.dataset_root / self.VIDEO_ROOT_NAME / "chunk-000"
        self.data_root = self.dataset_root / "data" / "chunk-000"
        self._video_cache = VideoFrameCache()
        self._parquet_cache = EpisodeParquetCache()

    def _load_tasks(self) -> dict[int, str]:
        tasks_path = self.dataset_root / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            raise FileNotFoundError(self._missing_meta_message(tasks_path))
        tasks: dict[int, str] = {}
        with tasks_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                tasks[int(row["task_index"])] = row["task"]
        return tasks

    def _load_episode_lengths(self) -> dict[int, int]:
        episodes_path = self.dataset_root / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            raise FileNotFoundError(self._missing_meta_message(episodes_path))
        lengths: dict[int, int] = {}
        with episodes_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                lengths[int(row["episode_index"])] = int(row["length"])
        return lengths

    def _missing_meta_message(self, missing_path: Path) -> str:
        candidates = []
        for sibling in sorted(self.dataset_root.parent.glob(f"{self.dataset_root.name}-*")):
            if (sibling / "meta" / missing_path.name).exists():
                candidates.append(str(sibling))
        hint = ""
        if candidates:
            hint = f" Existing converted repo candidates: {candidates}."
        return (
            f"Missing LeRobot metadata file: {missing_path}. "
            "Check data.dataset_root/--dataset-root or rerun convert_raw_to_lerobot.py "
            "to generate meta/tasks.jsonl and meta/episodes.jsonl."
            f"{hint}"
        )

    def _load_episode_df(self, episode_index: int) -> pd.DataFrame:
        return self._parquet_cache.get(self.data_root, episode_index)

    def _video_path(self, episode_index: int, image_key: str) -> Path:
        return self.video_root / image_key / f"episode_{episode_index:06d}.mp4"

    def _load_image(self, episode_index: int, frame_index: int, image_key: str) -> Image.Image:
        video_path = self._video_path(episode_index, image_key)
        frame = self._video_cache.read_frame(video_path, frame_index)
        image = Image.fromarray(frame).convert("RGB")
        if self.center_crop:
            image = center_crop_image(image)
        resize = self.image_sizes.get(image_key)
        if resize is not None:
            image = image.resize(resize)
        return image

    def _build_action_chunk(self, df: pd.DataFrame, start: int) -> tuple[np.ndarray, np.ndarray]:
        chunk = np.zeros((self.action_chunks_len, self.action_token_len), dtype=np.float32)
        mask = np.zeros((self.action_chunks_len, self.action_token_len), dtype=np.float32)
        for i in range(self.action_chunks_len):
            idx = start + i * self.action_frame_stride
            if idx < len(df):
                chunk[i] = np.asarray(df["action"].iloc[idx], dtype=np.float32)
                mask[i] = 1.0
        return chunk, mask

    def __len__(self) -> int:
        return len(self.frame_indices)

    def build_sample(self, episode_index: int, frame_index: int) -> dict[str, Any]:
        """Build one training sample for an arbitrary episode/frame (same as __getitem__)."""
        df = self._load_episode_df(episode_index)
        row = df.iloc[frame_index]

        state = np.asarray(row["observation.state"], dtype=np.float32)
        task_index = int(row["task_index"])
        task_description = self.tasks[task_index]

        images = [
            self._load_image(episode_index, frame_index, image_key)
            for image_key in self.image_keys
        ]

        action_chunk, action_loss_mask = self._build_action_chunk(df, frame_index)
        normalized_action_chunk = normalize_vector(
            action_chunk.reshape(-1),
            np.tile(self.action_mask, self.action_chunks_len),
            np.tile(self.action_q01, self.action_chunks_len),
            np.tile(self.action_q99, self.action_chunks_len),
        ).reshape(self.action_chunks_len, self.action_token_len)
        action_token_ids = self.action_tokenizer.action_to_token_ids(normalized_action_chunk.reshape(-1))
        action_token_ids = action_token_ids.reshape(self.action_chunks_len, self.action_token_len)

        state_tokens = ""
        if self.use_proprio:
            normalized_state = normalize_vector(state, self.state_mask, self.state_q01, self.state_q99)
            state_tokens = self.action_tokenizer(normalized_state)

        prompt_content: list[dict[str, Any]] = []
        for image in images:
            prompt_content.append({"type": "image", "image": image})
        prompt_content.append({"type": "text", "text": task_description + state_tokens})

        messages = [{"role": "user", "content": prompt_content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids_prompt = inputs.input_ids[0]
        attention_mask_prompt = torch.ones_like(input_ids_prompt)

        return {
            "input_ids_prompt": input_ids_prompt,
            "attention_mask_prompt": attention_mask_prompt,
            "pixel_values": inputs.pixel_values,
            "image_grid_thw": inputs.image_grid_thw,
            "action_token_ids": torch.tensor(action_token_ids, dtype=torch.long),
            "action_loss_mask": torch.tensor(action_loss_mask, dtype=torch.float32),
            "episode_index": episode_index,
            "frame_index": frame_index,
            "task_index": task_index,
            "raw_action_chunk": torch.tensor(action_chunk, dtype=torch.float32),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.frame_indices[index]
        return self.build_sample(sample.episode_index, sample.frame_index)


def split_episode_indices(
    episode_indices: list[int],
    val_episode_ratio: float,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    episodes = list(episode_indices)
    rng.shuffle(episodes)
    val_count = max(1, int(len(episodes) * val_episode_ratio))
    val_episodes = sorted(episodes[:val_count])
    train_episodes = sorted(episodes[val_count:])
    return train_episodes, val_episodes
