from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


class LatentCache:
    """Per-episode future latent cache stored as episode_XXXXXX.npz."""

    def __init__(self, root: str | Path, *, latent_length: int, hidden_size: int,
                 dataset_fingerprint: str | None = None, expected_future_frame_stride: int | None = None, max_cached_episodes: int = 2):
        self.root = Path(root)
        self.latent_length = int(latent_length)
        self.hidden_size = int(hidden_size)
        metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("latent_length") != self.latent_length or metadata.get("hidden_size") != self.hidden_size:
            raise ValueError(f"Latent cache shape metadata mismatch: {metadata}")
        if dataset_fingerprint and metadata.get("dataset_fingerprint") != dataset_fingerprint:
            raise ValueError("Latent cache dataset fingerprint mismatch")
        if expected_future_frame_stride is not None and metadata.get("future_frame_stride") != expected_future_frame_stride:
            raise ValueError("Latent cache future_frame_stride mismatch")
        self.metadata = metadata
        self.max_cached_episodes = int(max_cached_episodes)
        self._episodes: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, episode_index: int, frame_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if episode_index not in self._episodes:
            payload = np.load(self.root / f"episode_{episode_index:06d}.npz")
            self._episodes[episode_index] = (payload["latents"], payload["mask"])
            while len(self._episodes) > self.max_cached_episodes:
                self._episodes.popitem(last=False)
        else:
            self._episodes.move_to_end(episode_index)
        latents, mask = self._episodes[episode_index]
        value = np.asarray(latents[frame_index], dtype=np.float32)
        valid = np.asarray(mask[frame_index], dtype=np.float32)
        if value.shape != (self.latent_length, self.hidden_size) or valid.shape != (self.latent_length,):
            raise ValueError(f"Invalid latent cache entry shape: {value.shape}, {valid.shape}")
        return torch.from_numpy(value.copy()), torch.from_numpy(valid.copy())
