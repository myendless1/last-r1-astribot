from __future__ import annotations

from typing import Mapping, Sequence

import cv2
import numpy as np
import torch

from .embodiment import IMAGE_KEYS


def _as_rgb_uint8(image: np.ndarray | torch.Tensor, color_order: str = "rgb") -> np.ndarray:
    array = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D image, got {array.shape}")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB/BGR image, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if float(np.nanmax(array)) <= 1.5:
            array = array * 255
        array = np.clip(array, 0, 255).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8, copy=False)
    if color_order.lower() == "bgr":
        array = array[..., ::-1]
    elif color_order.lower() != "rgb":
        raise ValueError("color_order must be 'rgb' or 'bgr'")
    return np.ascontiguousarray(array)


def compose_t_layout_rgb(head: np.ndarray, left: np.ndarray, right: np.ndarray, *, color_order: str = "rgb") -> np.ndarray:
    head = _as_rgb_uint8(head, color_order)
    left = _as_rgb_uint8(left, color_order)
    right = _as_rgb_uint8(right, color_order)
    half_width = max(1, head.shape[1] // 2)
    def resize_wrist(image: np.ndarray) -> np.ndarray:
        height = max(1, round(image.shape[0] * half_width / image.shape[1]))
        return cv2.resize(image, (half_width, height), interpolation=cv2.INTER_AREA)
    left, right = resize_wrist(left), resize_wrist(right)
    bottom = np.zeros((max(left.shape[0], right.shape[0]), head.shape[1], 3), dtype=np.uint8)
    bottom[: left.shape[0], :half_width] = left
    bottom[: right.shape[0], half_width : half_width + right.shape[1]] = right
    return np.concatenate([head, bottom], axis=0)


def resize_with_padding(image: np.ndarray, size_hw: Sequence[int]) -> np.ndarray:
    target_h, target_w = map(int, size_hw)
    scale = min(target_h / image.shape[0], target_w / image.shape[1])
    new_w, new_h = max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    output = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y, x = (target_h - new_h) // 2, (target_w - new_w) // 2
    output[y : y + new_h, x : x + new_w] = resized
    return output


def preprocess_t_layout(images: Mapping[str, np.ndarray], size_hw: Sequence[int], *, color_order: str = "rgb") -> torch.Tensor:
    missing = [key for key in IMAGE_KEYS if key not in images]
    if missing:
        raise KeyError(f"Missing Astribot cameras: {missing}")
    composed = compose_t_layout_rgb(*(images[key] for key in IMAGE_KEYS), color_order=color_order)
    return torch.from_numpy(resize_with_padding(composed, size_hw)).permute(2, 0, 1).float() / 255
