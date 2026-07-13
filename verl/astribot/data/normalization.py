from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


class FeatureNormalizer:
    def __init__(self, source: str | Path | Mapping[str, Any], *, norm_type: str = "bounds_99_woclip", expected_scope: str | None = None, expected_metadata: Mapping[str, Any] | None = None):
        payload = json.loads(Path(source).read_text()) if isinstance(source, (str, Path)) else dict(source)
        self.metadata = dict(payload.get("metadata", {}))
        self.stats = payload.get("norm_stats", payload)
        self.norm_type = norm_type
        if expected_scope and self.metadata.get("dimension_scope") != expected_scope:
            raise ValueError(f"Norm scope mismatch: expected {expected_scope}, got {self.metadata.get('dimension_scope')}")
        for key, expected in (expected_metadata or {}).items():
            if self.metadata.get(key) != expected:
                raise ValueError(f"Norm metadata mismatch for {key}: expected {expected!r}, got {self.metadata.get(key)!r}")

    def _apply(self, value: np.ndarray | torch.Tensor, key: str, inverse: bool):
        array = value.detach().cpu().float().numpy() if isinstance(value, torch.Tensor) else np.asarray(value, dtype=np.float32)
        stats = self.stats[key]
        if self.norm_type == "bounds_99_woclip":
            low, high = np.asarray(stats["q01"], np.float32), np.asarray(stats["q99"], np.float32)
        elif self.norm_type == "minmax":
            low, high = np.asarray(stats["min"], np.float32), np.asarray(stats["max"], np.float32)
        else:
            raise ValueError(f"Unsupported norm_type={self.norm_type}")
        if array.shape[-1] != low.shape[-1]:
            raise ValueError(f"Normalizer dimension mismatch for {key}: {array.shape[-1]} != {low.shape[-1]}")
        result = (array + 1) * 0.5 * (high - low + 1e-6) + low if inverse else (array - low) / (high - low + 1e-6) * 2 - 1
        return torch.as_tensor(result, dtype=value.dtype, device=value.device) if isinstance(value, torch.Tensor) else result.astype(np.float32)

    def normalize(self, value, key: str): return self._apply(value, key, False)
    def denormalize(self, value, key: str): return self._apply(value, key, True)


class RunningStats:
    def __init__(self): self.chunks: list[np.ndarray] = []
    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.size: self.chunks.append(values.reshape(-1, values.shape[-1]))
    def result(self) -> dict[str, list[float]]:
        values = np.concatenate(self.chunks).astype(np.float64)
        return {"mean": values.mean(0).tolist(), "std": values.std(0).tolist(), "min": values.min(0).tolist(), "max": values.max(0).tolist(), "q01": np.quantile(values, .01, axis=0).tolist(), "q99": np.quantile(values, .99, axis=0).tolist()}
