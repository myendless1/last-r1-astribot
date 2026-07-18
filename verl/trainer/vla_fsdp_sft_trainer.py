# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FSDP SFT trainer for Qwen3-VL action-token warmup."""

from __future__ import annotations

import logging
import os
import json
import time
from pathlib import Path

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
import torch.distributed
from torch import optim
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader, DistributedSampler

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from verl.utils.dataset.astribot_lerobot_sft_dataset import (
    ASTRIBOT_IMAGE_SIZES,
    AstribotLeRobotSFTDataset,
    split_episode_indices,
)
from verl.utils.dataset.vla_sft_collate import vla_sft_collate_fn
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy_vla, init_fn
from verl.utils.torch_functional import get_cosine_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.utils.vla_model_utils import enable_frozen_vision_no_grad, prepare_qwen3vl_processor_and_model
from verl.workers.actor.action_tokenizer import ActionTokenizer

import verl.utils.hdfs_io as hdfs_io

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "INFO"))


def _patch_counts_per_sample(image_grid_thw: torch.Tensor, batch_size: int) -> list[int]:
    num_images = image_grid_thw.shape[0] // batch_size
    counts = []
    for i in range(batch_size):
        grids = image_grid_thw[i * num_images : (i + 1) * num_images]
        counts.append(int((grids[:, 0] * grids[:, 1] * grids[:, 2]).sum().item()))
    return counts


def split_vla_micro_batches(
    batch: dict[str, torch.Tensor],
    micro_batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    batch_size = batch["input_ids"].shape[0]
    num_images = batch["image_grid_thw"].shape[0] // batch_size
    patch_counts = _patch_counts_per_sample(batch["image_grid_thw"], batch_size)
    patch_prefix = [0]
    for count in patch_counts:
        patch_prefix.append(patch_prefix[-1] + count)

    micro_batches = []
    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)
        micro_batches.append(
            {
                "input_ids": batch["input_ids"][start:end],
                "attention_mask": batch["attention_mask"][start:end],
                "labels": batch["labels"][start:end],
                "image_grid_thw": batch["image_grid_thw"][start * num_images : end * num_images],
                "pixel_values": batch["pixel_values"][patch_prefix[start] : patch_prefix[end]],
            }
        )
    return micro_batches


class StepProfiler:
    def __init__(self, warmup_steps: int = 10):
        self.warmup_steps = warmup_steps
        self._step = 0
        self.records: list[dict[str, float]] = []

    def record(self, timings: dict[str, float]) -> None:
        self._step += 1
        if self._step > self.warmup_steps:
            self.records.append(timings)

    def summarize(self) -> dict[str, float]:
        if not self.records:
            return {}
        keys = self.records[0].keys()
        return {key: sum(record[key] for record in self.records) / len(self.records) for key in keys}

    @property
    def num_recorded(self) -> int:
        return len(self.records)


class SubmoduleForwardTimer:
    def __init__(self):
        self._starts: dict[str, float] = {}
        self.times: dict[str, float] = {}
        self._handles: list = []

    def attach(self, module, name: str) -> None:
        def pre_hook(_module, _inputs):
            _cuda_sync()
            self._starts[name] = time.perf_counter()

        def post_hook(_module, _inputs, _output):
            _cuda_sync()
            self.times[name] = self.times.get(name, 0.0) + (time.perf_counter() - self._starts[name])

        self._handles.append(module.register_forward_pre_hook(pre_hook))
        self._handles.append(module.register_forward_hook(post_hook))

    def reset(self) -> None:
        self.times = {}
        self._starts = {}

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class VLAFSDPSFTTrainer:
    def __init__(self, config: DictConfig, device_mesh: DeviceMesh):
        self.config = config
        self.device_mesh = device_mesh
        self.local_model_path = copy_local_path_from_hdfs(
            src=self.config.model.partial_pretrain, verbose=True
        )
        self._normalize_config_bsz()
        self._build_model_and_processor()
        self._build_dataloader()
        self._build_optimizer()

        if self.device_mesh.get_rank() == 0:
            print(OmegaConf.to_yaml(self.config))

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(
                "Per-device batch sizes: "
                f"train={self.config.data.train_batch_size}, "
                f"micro={self.config.data.micro_batch_size}, "
                f"global_train={self.config.data.train_batch_size * dp_size}"
            )

    def _action_slice_bounds(self) -> tuple[int, int]:
        action_len = self.config.data.action_chunks_len * self.config.data.action_token_len
        action_start = self.config.data.max_prompt_length
        return action_start, action_start + action_len

    def _compute_action_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Parallel CE over action slots; action labels are never model inputs."""
        action_start, action_end = self._action_slice_bounds()
        action_labels = labels[:, action_start:action_end].contiguous()
        if logits.shape[:2] != action_labels.shape:
            raise ValueError(
                f"Parallel action shape mismatch: logits={logits.shape}, labels={action_labels.shape}"
            )
        action_logits = logits[..., self.action_0_id : self.action_0_id + 256].float()
        valid = action_labels != -100
        local_labels = torch.where(valid, action_labels - self.action_0_id, action_labels)
        if valid.any() and ((local_labels[valid] < 0) | (local_labels[valid] >= 256)).any():
            raise ValueError("Action labels fall outside the 256-token action vocabulary")
        return torch.nn.functional.cross_entropy(
            action_logits.reshape(-1, 256),
            local_labels.reshape(-1),
            ignore_index=-100,
        )

    def _build_model_and_processor(self):
        freeze_vision = self.config.model.get("freeze_vision", True)
        action_decode_mode = self.config.model.get("action_decode_mode", "parallel")
        if action_decode_mode != "parallel":
            raise ValueError(
                f"Unsupported action_decode_mode={action_decode_mode!r}; LaST-R1 requires 'parallel'"
            )
        attn_implementation = self.config.model.get("attn_implementation", "sdpa") or "sdpa"
        if attn_implementation != "sdpa":
            raise ValueError(
                "Parallel action decoding requires model.attn_implementation=sdpa "
                "so the full-attention action-block mask is applied"
            )

        self.processor, self.model, added_tokens = prepare_qwen3vl_processor_and_model(
            self.local_model_path,
            local_files_only=True,
            num_action_tokens=self.config.model.get("action_special_tokens", 256),
            attn_implementation=attn_implementation,
        )
        if self.device_mesh.get_rank() == 0:
            print(f"Added {added_tokens} action special tokens; vocab={len(self.processor.tokenizer)}")
            if attn_implementation is not None:
                print(f"Using attn_implementation={attn_implementation} (freeze_vision={freeze_vision})")

        if hasattr(self.model, "value_head"):
            del self.model.value_head
        self.model.config.astribot_action_decode_mode = "parallel"

        if freeze_vision:
            self._freeze_vision_encoder()

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        self.pad_token_id = self.processor.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.processor.tokenizer.eos_token_id

        need_to_sub = self.config.model.get("need_to_sub", 3)
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer, need_to_sub=need_to_sub)
        self.action_0_id = self.action_tokenizer.action_0_id

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        auto_wrap_policy = get_fsdp_wrap_policy_vla(
            module=self.model,
            config=self.config.model.fsdp_config.wrap_policy,
        )
        cpu_offload = None
        if self.config.model.fsdp_config.cpu_offload:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        self.fsdp_model = FSDP(
            module=self.model,
            auto_wrap_policy=auto_wrap_policy,
            param_init_fn=init_fn,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_mesh=self.device_mesh,
            sync_module_states=True,
            device_id=torch.cuda.current_device(),
            cpu_offload=cpu_offload,
            use_orig_params=True,
        )
        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

    def _freeze_vision_encoder(self):
        backbone = getattr(self.model, "model", self.model)
        if not hasattr(backbone, "visual"):
            raise AttributeError("Cannot find vision encoder at model.model.visual")
        visual = backbone.visual
        frozen_params = sum(p.numel() for p in visual.parameters())
        for param in visual.parameters():
            param.requires_grad = False
        enable_frozen_vision_no_grad(self.model)
        if self.device_mesh.get_rank() == 0:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(
                f"Froze vision encoder ({frozen_params / 1e9:.2f}B params); "
                f"trainable: {trainable_params / 1e9:.2f}B; "
                f"vision forward uses torch.no_grad()"
            )

    def _make_dataset(
        self,
        episode_indices,
        *,
        max_frames: int | None = None,
        subsample_seed: int | None = None,
    ):
        return AstribotLeRobotSFTDataset(
            dataset_root=self.config.data.dataset_root,
            processor=self.processor,
            statistics_path=self.config.data.statistics_path,
            dataset_name=self.config.data.dataset_name,
            image_keys=list(self.config.data.image_keys),
            action_tokenizer=self.action_tokenizer,
            action_token_len=self.config.data.action_token_len,
            action_chunks_len=self.config.data.action_chunks_len,
            frame_stride=self.config.data.frame_stride,
            action_frame_stride=self.config.data.get("action_frame_stride", 4),
            use_proprio=self.config.data.use_proprio,
            center_crop=self.config.data.center_crop,
            image_sizes=ASTRIBOT_IMAGE_SIZES,
            episode_indices=episode_indices,
            max_frames=max_frames,
            subsample_seed=(
                self.config.trainer.seed if subsample_seed is None else subsample_seed
            ),
        )

    def _build_dataloader(self):
        episodes_path = Path(self.config.data.dataset_root) / "meta" / "episodes.jsonl"
        all_episodes = []
        with episodes_path.open("r", encoding="utf-8") as f:
            for line in f:
                all_episodes.append(int(json.loads(line)["episode_index"]))
        all_episodes = sorted(all_episodes)
        limit_episodes = self.config.data.get("limit_episodes")
        if limit_episodes is not None:
            all_episodes = all_episodes[:limit_episodes]

        train_episodes, val_episodes = split_episode_indices(
            all_episodes,
            val_episode_ratio=self.config.data.val_episode_ratio,
            seed=self.config.trainer.seed,
        )
        if self.device_mesh.get_rank() == 0:
            print(f"Train episodes: {len(train_episodes)}, Val episodes: {len(val_episodes)}")

        self.train_dataset = self._make_dataset(train_episodes)
        val_max_frames = self.config.data.get("val_max_frames")
        self.val_dataset = self._make_dataset(val_episodes, max_frames=val_max_frames)
        if self.device_mesh.get_rank() == 0:
            val_msg = f", val frames capped at {val_max_frames}" if val_max_frames else ""
            print(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}{val_msg}")

        rank = self.device_mesh.get_rank()
        world_size = self.device_mesh.size()
        self.train_sampler = DistributedSampler(
            self.train_dataset,
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            drop_last=True,
        )
        self.val_sampler = DistributedSampler(
            self.val_dataset,
            shuffle=False,
            num_replicas=world_size,
            rank=rank,
            drop_last=False,
        )

        collate = lambda batch: vla_sft_collate_fn(
            batch,
            pad_token_id=self.pad_token_id,
            max_prompt_length=self.config.data.max_prompt_length,
        )
        num_workers = self.config.data.get("num_workers", 2)
        loader_kwargs = {
            "collate_fn": collate,
            "num_workers": num_workers,
            "pin_memory": num_workers > 0,
            "prefetch_factor": 2 if num_workers > 0 else None,
        }
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            sampler=self.train_sampler,
            drop_last=True,
            **loader_kwargs,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.data.micro_batch_size,
            sampler=self.val_sampler,
            drop_last=False,
            **loader_kwargs,
        )

    def _resolve_logger_backends(self) -> list[str]:
        backends = list(self.config.trainer.logger)
        wandb_enabled = os.getenv("WANDB_ENABLED", "true").lower() in ("1", "true", "yes")
        if not wandb_enabled:
            backends = [backend for backend in backends if backend != "wandb"]
        return backends

    def _build_tracking(self) -> Tracking | None:
        if self.device_mesh.get_rank() != 0:
            return None
        backends = self._resolve_logger_backends()
        if self.device_mesh.get_rank() == 0:
            print(f"Logging backends: {backends}")
        return Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=backends,
            local_dir=self.config.trainer.default_local_dir,
            wandb_mode=self.config.trainer.get("wandb_mode", os.getenv("WANDB_MODE", "online")),
            config=OmegaConf.to_container(self.config, resolve=True),
        )

    def _build_optimizer(self):
        self.optimizer = optim.AdamW(
            (p for p in self.fsdp_model.parameters() if p.requires_grad),
            lr=self.config.optim.lr,
            betas=tuple(self.config.optim.betas),
            weight_decay=self.config.optim.weight_decay,
        )
        steps_per_epoch = max(len(self.train_dataloader), 1)
        self.total_steps = steps_per_epoch * self.config.trainer.total_epochs
        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.total_steps,
        )

    def _compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.fsdp_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
                astribot_parallel_action=True,
                prompt_length=self.config.data.max_prompt_length,
                action_length=(
                    self.config.data.action_chunks_len * self.config.data.action_token_len
                ),
                action_slot_id=self.action_0_id,
            )
        loss = self._compute_action_loss(outputs.action_logits, batch["labels"])
        if not torch.isfinite(loss):
            action_start, action_end = self._action_slice_bounds()
            valid = int((batch["labels"][:, action_start:action_end] != -100).sum())
            raise FloatingPointError(
                f"Non-finite loss (valid_action_labels={valid}, "
                f"batch_size={batch['input_ids'].shape[0]})"
            )
        return loss

    def _attach_forward_timer(self) -> SubmoduleForwardTimer:
        timer = SubmoduleForwardTimer()
        backbone = self.model.model
        timer.attach(backbone.visual, "fwd_vision")
        timer.attach(backbone.language_model, "fwd_language_model")
        timer.attach(self.model.lm_head, "fwd_lm_head")
        return timer

    def _batch_profile_info(self, batch: dict[str, torch.Tensor]) -> dict[str, int]:
        grids = batch["image_grid_thw"]
        batch_size = batch["input_ids"].shape[0]
        num_images = grids.shape[0]
        vision_tokens = int((grids[:, 0] * grids[:, 1] * grids[:, 2]).sum().item())
        return {
            "batch_size": batch_size,
            "seq_len": int(batch["input_ids"].shape[1]),
            "num_images": int(num_images),
            "vision_tokens": vision_tokens,
            "pixel_rows": int(batch["pixel_values"].shape[0]),
        }

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        metric, _ = self._training_step_impl(batch, profile=False, forward_timer=None)
        return metric

    def _training_step_impl(
        self,
        batch: dict[str, torch.Tensor],
        *,
        profile: bool,
        forward_timer: SubmoduleForwardTimer | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        self.fsdp_model.train()
        self.optimizer.zero_grad()

        micro_batches = split_vla_micro_batches(batch, self.config.data.micro_batch_size)
        loss = torch.tensor(0.0, device="cuda")
        forward_time = 0.0
        backward_time = 0.0
        forward_breakdown = {
            "fwd_vision": 0.0,
            "fwd_language_model": 0.0,
            "fwd_lm_head": 0.0,
            "fwd_other": 0.0,
        }
        for micro_batch in micro_batches:
            if forward_timer is not None:
                forward_timer.reset()
            if profile:
                _cuda_sync()
                forward_start = time.perf_counter()
            micro_loss = self._compute_loss(micro_batch) / len(micro_batches)
            if profile:
                _cuda_sync()
                backward_start = time.perf_counter()
                micro_forward = backward_start - forward_start
                forward_time += micro_forward
                if forward_timer is not None:
                    vision = forward_timer.times.get("fwd_vision", 0.0)
                    language = forward_timer.times.get("fwd_language_model", 0.0)
                    lm_head = forward_timer.times.get("fwd_lm_head", 0.0)
                    forward_breakdown["fwd_vision"] += vision
                    forward_breakdown["fwd_language_model"] += language
                    forward_breakdown["fwd_lm_head"] += lm_head
                    forward_breakdown["fwd_other"] += max(0.0, micro_forward - vision - language - lm_head)
            micro_loss.backward()
            if profile:
                _cuda_sync()
                backward_time += time.perf_counter() - backward_start
            loss = loss + micro_loss.detach()

        if profile:
            _cuda_sync()
            optim_start = time.perf_counter()
        self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        self.optimizer.step()
        self.lr_scheduler.step()
        if profile:
            _cuda_sync()
            optim_time = time.perf_counter() - optim_start
        else:
            optim_time = 0.0

        metric = {
            "train/loss": loss.item(),
            "train/lr(1e-5)": self.lr_scheduler.get_last_lr()[0] * 1e5,
        }
        timings = {
            "forward": forward_time,
            "backward": backward_time,
            "optim": optim_time,
            **forward_breakdown,
        }
        return metric, timings

    def validation_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.fsdp_model.eval()
        with torch.no_grad():
            losses = []
            micro_batches = split_vla_micro_batches(batch, self.config.data.micro_batch_size)
            for micro_batch in micro_batches:
                losses.append(self._compute_loss(micro_batch))
            loss = torch.stack(losses).mean()
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        return loss

    def _run_validation(self, tracking: Tracking | None, global_step: int) -> float | None:
        test_freq = self.config.trainer.get("test_freq")
        if not test_freq:
            return None

        rank = self.device_mesh.get_rank()
        if rank == 0:
            print(f"Running validation at step {global_step}...")

        self.fsdp_model.eval()
        val_losses = []
        for data in self.val_dataloader:
            batch = {k: v.cuda(non_blocking=True) for k, v in data.items()}
            val_losses.append(self.validation_step(batch))

        val_loss = torch.stack(val_losses).mean().item()
        if tracking is not None:
            tracking.log(data={"val/loss": val_loss}, step=global_step)
        if rank == 0:
            print(f"Validation step {global_step}: val/loss={val_loss:.5f}")

        torch.distributed.barrier()
        self.fsdp_model.train()
        return val_loss

    def save_checkpoint(self, step: int):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.fsdp_model.state_dict()

        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")
        if self.device_mesh.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path, state_dict=state_dict)
            self.processor.save_pretrained(path)
            if self.config.trainer.default_hdfs_dir:
                hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
                hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)
        torch.distributed.barrier()

    def _print_profile_summary(self, summary: dict[str, float], num_recorded: int, warmup_steps: int) -> None:
        top_level = ("data_load", "h2d", "forward", "backward", "optim")
        forward_parts = ("fwd_vision", "fwd_language_model", "fwd_lm_head", "fwd_other")
        total = sum(summary.get(stage, 0.0) for stage in top_level)
        forward_total = summary.get("forward", 0.0)

        print("\n" + "=" * 60)
        print(f"SFT step profiling (avg of last {num_recorded} steps, warmup={warmup_steps})")
        print("=" * 60)
        for stage in top_level:
            seconds = summary.get(stage, 0.0)
            pct = 100.0 * seconds / total if total > 0 else 0.0
            print(f"  {stage:20s}: {seconds:7.3f}s  ({pct:5.1f}%)")
        if forward_total > 0:
            print("  forward breakdown:")
            for stage in forward_parts:
                seconds = summary.get(stage, 0.0)
                pct = 100.0 * seconds / forward_total
                print(f"    {stage:18s}: {seconds:7.3f}s  ({pct:5.1f}% of forward)")
        print("-" * 60)
        print(f"  {'total':20s}: {total:7.3f}s")
        print("=" * 60 + "\n")

    def profile_fit(self, num_steps: int, warmup_steps: int = 10) -> dict[str, float]:
        rank = self.device_mesh.get_rank()
        profiler = StepProfiler(warmup_steps=warmup_steps)
        forward_timer = self._attach_forward_timer()
        data_iter = iter(self.train_dataloader)
        step_start = time.perf_counter()
        printed_batch_info = False

        try:
            for step_idx in range(num_steps):
                try:
                    data = next(data_iter)
                except StopIteration:
                    self.train_sampler.set_epoch(step_idx)
                    data_iter = iter(self.train_dataloader)
                    data = next(data_iter)

                _cuda_sync()
                after_data = time.perf_counter()
                data_load_time = after_data - step_start

                batch = {k: v.cuda(non_blocking=True) for k, v in data.items()}
                _cuda_sync()
                after_h2d = time.perf_counter()
                h2d_time = after_h2d - after_data

                if rank == 0 and not printed_batch_info:
                    info = self._batch_profile_info(batch)
                    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
                    print(
                        "[profile] batch info: "
                        f"gpu={gpu_name}, "
                        f"bs={info['batch_size']}, seq={info['seq_len']}, "
                        f"images={info['num_images']}, vision_tokens={info['vision_tokens']}, "
                        f"pixel_rows={info['pixel_rows']}, "
                        f"micro_batch={self.config.data.micro_batch_size}, "
                        f"freeze_vision={self.config.model.get('freeze_vision', True)}"
                    )
                    printed_batch_info = True

                metric, train_timings = self._training_step_impl(
                    batch, profile=True, forward_timer=forward_timer
                )
                profiler.record(
                    {
                        "data_load": data_load_time,
                        "h2d": h2d_time,
                        **train_timings,
                    }
                )

                if rank == 0:
                    print(
                        f"profile step {step_idx + 1}/{num_steps} "
                        f"(recorded after warmup) - train/loss: {metric['train/loss']:.5f}"
                    )

                _cuda_sync()
                step_start = time.perf_counter()
        finally:
            forward_timer.detach()

        summary = profiler.summarize()
        if rank == 0:
            self._print_profile_summary(summary, profiler.num_recorded, warmup_steps)
        torch.distributed.barrier()
        return summary

    def fit(self):
        profile_steps = self.config.trainer.get("profile_steps")
        if profile_steps:
            warmup_steps = self.config.trainer.get("profile_warmup_steps", 10)
            self.profile_fit(num_steps=profile_steps, warmup_steps=warmup_steps)
            return

        rank = self.device_mesh.get_rank()
        tracking = self._build_tracking()

        global_step = 0
        test_freq = self.config.trainer.get("test_freq")
        last_validated_step = 0
        try:
            for epoch in range(self.config.trainer.total_epochs):
                self.train_sampler.set_epoch(epoch)
                step_start = time.perf_counter()
                for data in self.train_dataloader:
                    batch = {k: v.cuda(non_blocking=True) for k, v in data.items()}
                    metric = self.training_step(batch)
                    _cuda_sync()
                    metric["train/step_time"] = time.perf_counter() - step_start
                    if tracking is not None:
                        tracking.log(data=metric, step=global_step)
                    global_step += 1
                    step_start = time.perf_counter()

                    save_freq = self.config.trainer.get("save_freq")
                    if save_freq and global_step % save_freq == 0:
                        self.save_checkpoint(step=global_step)

                    if test_freq and global_step % test_freq == 0:
                        self._run_validation(tracking, global_step)
                        last_validated_step = global_step

                torch.distributed.barrier()
                self.save_checkpoint(step=global_step)

            if test_freq and global_step > 0 and last_validated_step != global_step:
                self._run_validation(tracking, global_step)
        finally:
            if tracking is not None and "wandb" in tracking.logger:
                tracking.logger["wandb"].finish()


@hydra.main(config_path="config", config_name="vla_sft_astribot", version_base=None)
def main(config: DictConfig):
    local_rank, rank, world_size = initialize_global_process_group()
    del local_rank, rank

    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",))
    trainer = VLAFSDPSFTTrainer(config=config, device_mesh=device_mesh)
    trainer.fit()


if __name__ == "__main__":
    main()
