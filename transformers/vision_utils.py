"""CPU-side helpers for Qwen3-VL vision tower metadata (static per grid)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def get_vision_bilinear_indices_and_weights(
    grid_thw: torch.Tensor,
    *,
    num_grid_per_side: int,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Bilinear position-embed indices/weights in final packed token order."""
  grid_thw = grid_thw.detach().cpu()
  grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

  idx_list: list[list[int]] = [[] for _ in range(4)]
  weight_list: list[list[float]] = [[] for _ in range(4)]

  for t, h, w in zip(grid_ts.tolist(), grid_hs.tolist(), grid_ws.tolist()):
    h_idxs = torch.linspace(0, num_grid_per_side - 1, int(h))
    w_idxs = torch.linspace(0, num_grid_per_side - 1, int(w))

    h_idxs_floor = h_idxs.int()
    w_idxs_floor = w_idxs.int()
    h_idxs_ceil = (h_idxs.int() + 1).clip(max=num_grid_per_side - 1)
    w_idxs_ceil = (w_idxs.int() + 1).clip(max=num_grid_per_side - 1)

    dh = h_idxs - h_idxs_floor
    dw = w_idxs - w_idxs_floor

    base_h = h_idxs_floor * num_grid_per_side
    base_h_ceil = h_idxs_ceil * num_grid_per_side

    indices = [
      (base_h[None].T + w_idxs_floor[None]).flatten(),
      (base_h[None].T + w_idxs_ceil[None]).flatten(),
      (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
      (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
    ]
    weights = [
      ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
      ((1 - dh)[None].T * dw[None]).flatten(),
      (dh[None].T * (1 - dw)[None]).flatten(),
      (dh[None].T * dw[None]).flatten(),
    ]

    for i in range(4):
      idx_list[i].extend(indices[i].tolist())
      weight_list[i].extend(weights[i].tolist())

  bilinear_indices = torch.tensor(idx_list, dtype=torch.long)
  bilinear_weights = torch.tensor(weight_list, dtype=torch.float32)

  split_sizes = [int(h * w) for h, w in zip(grid_hs.tolist(), grid_ws.tolist())]
  merge_size = spatial_merge_size
  merged_indices = []
  merged_weights = []
  for corner in range(4):
    corner_idx = bilinear_indices[corner]
    corner_w = bilinear_weights[corner]
    parts = corner_idx.split(split_sizes)
    parts_w = corner_w.split(split_sizes)
    reordered_idx = []
    reordered_w = []
    for pe_idx, pe_w, t, h, w in zip(parts, parts_w, grid_ts.tolist(), grid_hs.tolist(), grid_ws.tolist()):
      h, w, t = int(h), int(w), int(t)
      pe_idx = pe_idx.view(h, w)
      pe_w = pe_w.view(h, w)
      pe_idx = (
        pe_idx.view(1, h // merge_size, merge_size, w // merge_size, merge_size)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1)
      )
      pe_w = (
        pe_w.view(1, h // merge_size, merge_size, w // merge_size, merge_size)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1)
      )
      if t > 1:
        pe_idx = pe_idx.repeat(t)
        pe_w = pe_w.repeat(t)
      reordered_idx.append(pe_idx)
      reordered_w.append(pe_w)
    merged_indices.append(torch.cat(reordered_idx))
    merged_weights.append(torch.cat(reordered_w))

  return torch.stack(merged_indices), torch.stack(merged_weights)


def get_vision_position_ids(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
  """Packed 2D coords (seq_len, 2) before rotary freq lookup."""
  grid_thw = grid_thw.detach().cpu()
  merge_size = spatial_merge_size
  total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
  pos_ids = torch.empty((total_tokens, 2), dtype=torch.long)
  offset = 0
  for num_frames, height, width in grid_thw.tolist():
    height, width = int(height), int(width)
    num_frames = int(num_frames)
    merged_h, merged_w = height // merge_size, width // merge_size

    block_rows = torch.arange(merged_h)
    block_cols = torch.arange(merged_w)
    intra_row = torch.arange(merge_size)
    intra_col = torch.arange(merge_size)

    row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
    col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
    row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
    col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
    coords = torch.stack((row_idx, col_idx), dim=-1)
    if num_frames > 1:
      coords = coords.repeat(num_frames, 1)
    n = coords.shape[0]
    pos_ids[offset : offset + n] = coords
    offset += n
  return pos_ids


def get_vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
  grid_thw = grid_thw.detach().cpu()
  cu_seqlens = torch.repeat_interleave(
    grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
  ).cumsum(dim=0, dtype=torch.int32)
  return F.pad(cu_seqlens, (1, 0), value=0)


def get_vision_position_embeddings(
    grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
    rotary_module,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Precompute vision rotary (cos, sin) on CPU."""
  grid_thw = grid_thw.detach().cpu()
  max_hw = int(grid_thw[:, 1:].max().item())
  freq_table = rotary_module(max_hw).detach().cpu()
  pos_ids = get_vision_position_ids(grid_thw, spatial_merge_size)
  rotary_pos_emb = freq_table[pos_ids].flatten(1)
  emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
  return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_bilinear_pos_embed(
    pos_embed_weight: torch.Tensor,
    bilinear_indices: torch.Tensor,
    bilinear_weights: torch.Tensor,
) -> torch.Tensor:
  """Apply precomputed bilinear indices/weights to pos_embed table."""
  weight_dtype = pos_embed_weight.dtype
  idx = bilinear_indices.to(device=pos_embed_weight.device)
  w = bilinear_weights.to(device=pos_embed_weight.device, dtype=weight_dtype)
  parts = pos_embed_weight[idx] * w[:, :, None]
  return parts[0] + parts[1] + parts[2] + parts[3]
