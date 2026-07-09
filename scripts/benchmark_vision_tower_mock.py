#!/usr/bin/env python3
"""Validate Conv3d vs Linear/Einsum equivalence for Qwen3-VL patch_embed.proj."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _bootstrap_local_transformers() -> None:
    """Prefer this repo's transformers fork."""
    repo_root = str(_REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    if "transformers.utils.versions" in sys.modules:
        return

    versions_path = _REPO_ROOT / "transformers" / "utils" / "versions.py"
    spec = importlib.util.spec_from_file_location("transformers.utils.versions", versions_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {versions_path}")
    versions_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(versions_mod)
    versions_mod.require_version_core = lambda *args, **kwargs: None
    versions_mod.require_version = lambda *args, **kwargs: None
    sys.modules["transformers.utils.versions"] = versions_mod


_bootstrap_local_transformers()

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

ASTRIBOT_GRIDS_PER_SAMPLE = [
    [1, 16, 20],
    [1, 16, 18],
    [1, 16, 18],
]


@dataclass(frozen=True)
class PatchProjWeights:
    """Conv3d weights copied into tensor layouts used by einsum / linear."""

    conv_weight: torch.Tensor
    conv_bias: torch.Tensor
    einsum_weight: torch.Tensor
    linear_weight: torch.Tensor
    flat_dim: int
    in_channels: int
    flat_spatial: int
    embed_dim: int

    @classmethod
    def from_patch_embed(cls, patch_embed, *, dtype: torch.dtype, device: torch.device) -> PatchProjWeights:
        proj = patch_embed.proj
        in_channels = patch_embed.in_channels
        flat_spatial = patch_embed.temporal_patch_size * patch_embed.patch_size * patch_embed.patch_size
        flat_dim = in_channels * flat_spatial
        embed_dim = patch_embed.embed_dim

        conv_weight = proj.weight.detach().to(device=device, dtype=dtype)
        conv_bias = proj.bias.detach().to(device=device, dtype=dtype)
        einsum_weight = conv_weight.reshape(embed_dim, in_channels, flat_spatial)
        linear_weight = conv_weight.reshape(embed_dim, flat_dim)
        return cls(
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            einsum_weight=einsum_weight,
            linear_weight=linear_weight,
            flat_dim=flat_dim,
            in_channels=in_channels,
            flat_spatial=flat_spatial,
            embed_dim=embed_dim,
        )


def seq_len_from_grids(batch_size: int, grids_per_sample: list[list[int]]) -> int:
    image_grid_thw = torch.tensor(grids_per_sample * batch_size, dtype=torch.long)
    return int((image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).sum().item())


def build_pixel_values(
    seq_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    flat_dim = 3 * 2 * 16 * 16
    return torch.randn(seq_len, flat_dim, device=device, dtype=dtype, generator=generator)


def proj_conv3d(
    patch_embed,
    pixel_values: torch.Tensor,
    weights: PatchProjWeights,
) -> torch.Tensor:
    conv_input = pixel_values.view(
        -1,
        patch_embed.in_channels,
        patch_embed.temporal_patch_size,
        patch_embed.patch_size,
        patch_embed.patch_size,
    )
    out = F.conv3d(
        conv_input,
        weights.conv_weight,
        weights.conv_bias,
        stride=(
            patch_embed.temporal_patch_size,
            patch_embed.patch_size,
            patch_embed.patch_size,
        ),
    )
    return out.view(-1, weights.embed_dim)


def proj_einsum(pixel_values: torch.Tensor, weights: PatchProjWeights) -> torch.Tensor:
    hidden_states = pixel_values.view(-1, weights.in_channels, weights.flat_spatial)
    out = torch.einsum("ncw, hcw->nh", hidden_states, weights.einsum_weight)
    return out + weights.conv_bias.view(1, -1)


def proj_linear(pixel_values: torch.Tensor, weights: PatchProjWeights) -> torch.Tensor:
    hidden_states = pixel_values.view(-1, weights.flat_dim)
    return F.linear(hidden_states, weights.linear_weight, weights.conv_bias)


@dataclass(frozen=True)
class DiffStats:
    max_abs: float
    mean_abs: float
    rmse: float

    @classmethod
    def between(cls, ref: torch.Tensor, other: torch.Tensor) -> DiffStats:
        diff = (ref.float() - other.float()).abs()
        return cls(
            max_abs=float(diff.max().item()),
            mean_abs=float(diff.mean().item()),
            rmse=float(torch.sqrt((diff * diff).mean()).item()),
        )


def compare_dtype(
    patch_embed,
    pixel_values_fp32: torch.Tensor,
    dtype: torch.dtype,
    *,
    device: torch.device,
) -> dict[str, DiffStats]:
    pixel_values = pixel_values_fp32.to(dtype=dtype)
    weights = PatchProjWeights.from_patch_embed(patch_embed, dtype=dtype, device=device)

    with torch.inference_mode():
        ref = proj_conv3d(patch_embed, pixel_values, weights)
        via_einsum = proj_einsum(pixel_values, weights)
        via_linear = proj_linear(pixel_values, weights)

    return {
        "einsum_vs_conv3d": DiffStats.between(ref, via_einsum),
        "linear_vs_conv3d": DiffStats.between(ref, via_linear),
        "einsum_vs_linear": DiffStats.between(via_einsum, via_linear),
    }


def load_patch_embed(model_path: str, device: torch.device) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    visual = Qwen3VLVisionModel._from_config(config.vision_config)
    visual = visual.to(device=device)
    visual.eval()
    return visual.patch_embed


def print_diff_row(label: str, stats: DiffStats) -> None:
    print(f"  {label:<22} max_abs={stats.max_abs:.6e}  mean_abs={stats.mean_abs:.6e}  rmse={stats.rmse:.6e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare patch_embed Conv3d vs einsum/linear numeric error (bf16/fp32)",
    )
    parser.add_argument(
        "--model-path",
        default="/media/damoxing/ckp/qwen_ft/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtypes",
        default="float32,bfloat16",
        help="Comma-separated dtypes to test, e.g. float32,bfloat16",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    dtypes = []
    for name in args.dtypes.split(","):
        key = name.strip().lower()
        if not key:
            continue
        if key not in dtype_map:
            raise ValueError(f"unsupported dtype {name!r}, choose from {list(dtype_map)}")
        dtypes.append((key, dtype_map[key]))
    if not dtypes:
        raise ValueError("no dtypes selected")

    device = torch.device("cuda:0")
    patch_embed = load_patch_embed(args.model_path, device)
    seq_len = seq_len_from_grids(args.batch_size, ASTRIBOT_GRIDS_PER_SAMPLE)
    pixel_values_fp32 = build_pixel_values(seq_len, device=device, dtype=torch.float32, seed=args.seed)

    pe = patch_embed
    flat_spatial = pe.temporal_patch_size * pe.patch_size * pe.patch_size
    flat_dim = pe.in_channels * flat_spatial

    print("=== patch_embed.proj: Conv3d vs einsum/linear accuracy ===")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"model: {args.model_path}")
    print(f"batch_size: {args.batch_size}, seq_len: {seq_len}")
    print(f"pixel_values: ({seq_len}, {flat_dim})")
    print(
        f"conv3d weight: {tuple(pe.proj.weight.shape)} -> "
        f"einsum ({pe.embed_dim}, {pe.in_channels}, {flat_spatial}), "
        f"linear ({pe.embed_dim}, {flat_dim})"
    )
    print(f"seed: {args.seed}")
    print()

    for dtype_name, dtype in dtypes:
        print(f"--- {dtype_name} ---")
        stats = compare_dtype(patch_embed, pixel_values_fp32, dtype, device=device)
        print_diff_row("einsum vs conv3d", stats["einsum_vs_conv3d"])
        print_diff_row("linear vs conv3d", stats["linear_vs_conv3d"])
        print_diff_row("einsum vs linear", stats["einsum_vs_linear"])
        print()


if __name__ == "__main__":
    main()
