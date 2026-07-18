#!/usr/bin/env python3
"""Benchmark vision-forward scaling vs batch size (CUDA, same setup as SFT trainer)."""

from __future__ import annotations

import os
import time

os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import hydra
import torch
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh

from verl.trainer.vla_fsdp_sft_trainer import (
    VLAFSDPSFTTrainer,
    _cuda_sync,
    split_vla_micro_batches,
)
from verl.utils.dataset.vla_sft_collate import vla_sft_collate_fn
from verl.utils.distributed import initialize_global_process_group


def _device_report(trainer: VLAFSDPSFTTrainer) -> None:
    visual = trainer.model.model.visual
    sample_param = next(visual.parameters())
    print("\n=== Device check ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current CUDA device: {torch.cuda.current_device()} ({torch.cuda.get_device_name()})")
    print(f"visual.proj.weight device: {sample_param.device}")
    if hasattr(trainer.fsdp_model, "compute_device"):
        print(f"fsdp_model.compute_device: {trainer.fsdp_model.compute_device}")


def _make_batch(trainer: VLAFSDPSFTTrainer, batch_size: int) -> dict[str, torch.Tensor]:
    samples = [trainer.train_dataset[i] for i in range(batch_size)]
    batch = vla_sft_collate_fn(
        samples,
        pad_token_id=trainer.pad_token_id,
        max_prompt_length=trainer.config.data.max_prompt_length,
    )
    return {k: v.cuda(non_blocking=True) for k, v in batch.items()}


def _benchmark_batch(
    trainer: VLAFSDPSFTTrainer,
    batch_size: int,
    micro_batch_size: int,
    repeats: int = 5,
) -> dict[str, float]:
    forward_timer = trainer._attach_forward_timer()
    batch = _make_batch(trainer, batch_size)
    info = trainer._batch_profile_info(batch)
    micro_batches = split_vla_micro_batches(batch, micro_batch_size)

    def run_forward_once() -> tuple[float, float]:
        trainer.fsdp_model.train()
        forward_timer.reset()
        forward_total = 0.0
        for micro_batch in micro_batches:
            _cuda_sync()
            t0 = time.perf_counter()
            trainer._compute_loss(micro_batch)
            _cuda_sync()
            forward_total += time.perf_counter() - t0
        vision_total = forward_timer.times.get("fwd_vision", 0.0)
        return forward_total, vision_total

    for _ in range(2):
        run_forward_once()

    forward_times = []
    vision_times = []
    for _ in range(repeats):
        fwd, vis = run_forward_once()
        forward_times.append(fwd)
        vision_times.append(vis)

    forward_timer.detach()
    return {
        "batch_size": batch_size,
        "micro_batch_size": micro_batch_size,
        "num_micro_batches": len(micro_batches),
        "vision_tokens": info["vision_tokens"],
        "tokens_per_sample": info["vision_tokens"] / batch_size,
        "forward_total": sum(forward_times) / len(forward_times),
        "fwd_vision": sum(vision_times) / len(vision_times),
    }


@hydra.main(config_path="../verl/trainer/config", config_name="vla_sft_astribot", version_base=None)
def main(config: DictConfig) -> None:
    local_rank, rank, world_size = initialize_global_process_group()
    del local_rank, rank

    config.data.limit_episodes = 5
    config.trainer.profile_steps = None

    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",))
    trainer = VLAFSDPSFTTrainer(config=config, device_mesh=device_mesh)

    if device_mesh.get_rank() == 0:
        _device_report(trainer)

        batch_sizes = [1, 2, 4, 8]
        print("\n=== Batch scaling (micro_batch_size=batch_size, forward only) ===")
        print(
            f"{'bs':>3} | {'micro':>5} | {'vision_tok':>10} | {'tok/sample':>10} | "
            f"{'fwd_vision(s)':>12} | {'forward(s)':>10} | {'ms/token':>8}"
        )
        print("-" * 82)

        results = []
        for bs in batch_sizes:
            row = _benchmark_batch(trainer, bs, micro_batch_size=bs)
            results.append(row)
            per_token_ms = 1000.0 * row["fwd_vision"] / max(row["vision_tokens"], 1)
            print(
                f"{row['batch_size']:>3} | {row['micro_batch_size']:>5} | {row['vision_tokens']:>10} | "
                f"{row['tokens_per_sample']:>10.1f} | {row['fwd_vision']:>12.3f} | "
                f"{row['forward_total']:>10.3f} | {per_token_ms:>8.4f}"
            )

        print("\n=== Scaling ratios (relative to bs=1) ===")
        base = results[0]
        for row in results:
            tok_ratio = row["vision_tokens"] / base["vision_tokens"]
            vision_ratio = row["fwd_vision"] / base["fwd_vision"]
            print(
                f"bs={row['batch_size']}: vision_tokens x{tok_ratio:.2f}, "
                f"fwd_vision x{vision_ratio:.2f} (ideal linear would match token ratio)"
            )

    torch.distributed.barrier()


if __name__ == "__main__":
    main()
