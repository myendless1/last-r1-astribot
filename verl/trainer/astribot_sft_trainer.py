from __future__ import annotations

from pathlib import Path

import hydra
import pyarrow.parquet as pq
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader, DistributedSampler

from verl.astribot.data.collate import astribot_sft_collate
from verl.astribot.data.lerobot_dataset import AstribotLeRobotDataset
from verl.astribot.modeling.losses import latent_cosine_loss, latent_end_loss, parallel_action_loss
from verl.astribot.protocol import AstribotProtocol
from verl.trainer.vla_fsdp_sft_trainer import VLAFSDPSFTTrainer
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fsdp_utils import get_fsdp_wrap_policy_vla, init_fn
from verl.utils.vla_model_utils import prepare_qwen3vl_processor_and_model
from verl.workers.actor.action_tokenizer import ActionTokenizer


def _episode_split(root: str, ratio: float, seed: int) -> tuple[list[int], list[int]]:
    episodes = set()
    for path in Path(root).glob("data/**/*.parquet"):
        episodes.update(map(int, pq.read_table(path, columns=["episode_index"]).column(0).to_pylist()))
    generator = torch.Generator().manual_seed(seed)
    values = sorted(episodes)
    if len(values) < 2:
        raise ValueError("Astribot training requires at least two episodes for an episode-level train/validation split")
    order = torch.randperm(len(values), generator=generator).tolist()
    shuffled = [values[index] for index in order]
    count = max(1, int(len(values) * ratio))
    return sorted(shuffled[count:]), sorted(shuffled[:count])


class AstribotSFTTrainer(VLAFSDPSFTTrainer):
    def _build_model_and_processor(self):
        self.processor, self.model, _ = prepare_qwen3vl_processor_and_model(
            self.local_model_path, local_files_only=True,
            num_action_tokens=self.config.model.get("action_special_tokens", 256),
            attn_implementation=self.config.model.get("attn_implementation", "sdpa"),
        )
        latent_tokens = ["<latent_start>", "<latent_end>", "<latent_pad>"]
        existing = set(self.processor.tokenizer.additional_special_tokens or [])
        added = self.processor.tokenizer.add_special_tokens({"additional_special_tokens": [token for token in latent_tokens if token not in existing]})
        if added:
            self.model.resize_token_embeddings(len(self.processor.tokenizer))
        if hasattr(self.model, "value_head"):
            del self.model.value_head
        self.pad_token_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.action_0_id = self.action_tokenizer.action_0_id
        self.latent_end_id = self.processor.tokenizer.convert_tokens_to_ids("<latent_end>")
        self.latent_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<latent_pad>")
        self.protocol = AstribotProtocol(
            dimension_scope=self.config.robot.dimension_scope,
            action_horizon=self.config.data.action_horizon,
            latent_length=self.config.model.latent_length,
            image_height=self.config.data.image_size_hw[0], image_width=self.config.data.image_size_hw[1],
            max_prompt_length=self.config.data.max_prompt_length,
            action_frame_stride=self.config.data.action_frame_stride,
            future_frame_stride=self.config.data.future_frame_stride,
        )
        if self.model.config.text_config.hidden_size != self.config.model.latent_hidden_size:
            raise ValueError("Qwen hidden size must equal latent_hidden_size; use Qwen3-VL-4B for DINO top-k 2560")
        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
        policy = get_fsdp_wrap_policy_vla(self.model, self.config.model.fsdp_config.wrap_policy)
        offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params) if self.config.model.fsdp_config.cpu_offload else None
        self.fsdp_model = FSDP(self.model, auto_wrap_policy=policy, param_init_fn=init_fn,
            sharding_strategy=ShardingStrategy.FULL_SHARD, mixed_precision=precision, device_mesh=self.device_mesh,
            sync_module_states=True, device_id=torch.cuda.current_device(), cpu_offload=offload, use_orig_params=True)
        log_gpu_memory_usage("After Astribot FSDP wrapping")

    def _make_dataset(self, episodes):
        return AstribotLeRobotDataset(
            root=self.config.data.dataset_root, processor=self.processor, action_tokenizer=self.action_tokenizer,
            dimension_scope=self.protocol.dimension_scope, norm_path=self.config.data.normalization_path,
            latent_cache_path=self.config.data.latent_cache_path, hidden_size=self.config.model.latent_hidden_size,
            action_horizon=self.protocol.action_horizon, action_frame_stride=self.config.data.action_frame_stride, future_frame_stride=self.config.data.future_frame_stride,
            latent_length=self.protocol.latent_length, image_size_hw=self.config.data.image_size_hw,
            episode_indices=episodes,
        )

    def _build_dataloader(self):
        train_eps, val_eps = _episode_split(self.config.data.dataset_root, self.config.data.val_episode_ratio, self.config.trainer.seed)
        self.train_dataset, self.val_dataset = self._make_dataset(train_eps), self._make_dataset(val_eps)
        rank, world = self.device_mesh.get_rank(), self.device_mesh.size()
        self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        self.val_sampler = DistributedSampler(self.val_dataset, num_replicas=world, rank=rank, shuffle=False)
        collate = lambda items: astribot_sft_collate(items, pad_token_id=self.pad_token_id,
            latent_pad_id=self.latent_pad_id, latent_end_id=self.latent_end_id,
            max_prompt_length=self.config.data.max_prompt_length)
        kwargs = dict(num_workers=self.config.data.num_workers, collate_fn=collate, pin_memory=True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.data.train_batch_size,
            sampler=self.train_sampler, drop_last=True, **kwargs)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.config.data.micro_batch_size,
            sampler=self.val_sampler, **kwargs)

    def _split_micro(self, batch):
        size = batch["input_ids"].shape[0]
        micro = self.config.data.micro_batch_size
        images_per_sample = batch["image_grid_thw"].shape[0] // size
        patch_counts = []
        for index in range(size):
            grids = batch["image_grid_thw"][index * images_per_sample : (index + 1) * images_per_sample]
            patch_counts.append(int(grids.prod(dim=-1).sum().item()))
        patch_offsets = [0]
        for count in patch_counts:
            patch_offsets.append(patch_offsets[-1] + count)
        for start in range(0, size, micro):
            end = min(size, start + micro)
            part = {key: value[start:end] for key, value in batch.items() if key not in {"pixel_values", "image_grid_thw"}}
            part["image_grid_thw"] = batch["image_grid_thw"][start * images_per_sample : end * images_per_sample]
            part["pixel_values"] = batch["pixel_values"][patch_offsets[start] : patch_offsets[end]]
            yield part

    def _compute_loss(self, batch):
        result = self.fsdp_model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"], image_grid_thw=batch["image_grid_thw"],
            astribot_parallel_sft=True, prompt_length=self.config.data.max_prompt_length,
            latent_length=self.protocol.latent_length, action_length=self.protocol.action_length,
            latent_end_id=self.latent_end_id,
        )
        action_logits = result.action_logits[..., self.action_0_id:self.action_0_id + 256]
        latent = latent_cosine_loss(result.predicted_latents, batch["latent_gt_embeds"], batch["latent_mask"])
        end = latent_end_loss(result.latent_end_logits, self.latent_end_id)
        action = parallel_action_loss(action_logits, batch["action_labels"], batch["action_mask"], self.action_0_id)
        return latent * self.config.loss.latent_weight + end * self.config.loss.latent_end_weight + action * self.config.loss.action_weight

    def _training_step_impl(self, batch, *, profile=False, forward_timer=None):
        self.fsdp_model.train(); self.optimizer.zero_grad()
        parts = list(self._split_micro(batch)); total = torch.zeros((), device="cuda")
        for part in parts:
            loss = self._compute_loss(part) / len(parts); loss.backward(); total += loss.detach()
        self.fsdp_model.clip_grad_norm_(self.config.optim.clip_grad)
        self.optimizer.step(); self.lr_scheduler.step()
        return {"train/loss": total.item(), "train/lr(1e-5)": self.lr_scheduler.get_last_lr()[0] * 1e5}, {}

    def validation_step(self, batch):
        self.fsdp_model.eval()
        with torch.no_grad(): loss = torch.stack([self._compute_loss(part) for part in self._split_micro(batch)]).mean()
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        return loss

    def save_checkpoint(self, step: int):
        super().save_checkpoint(step)
        if self.device_mesh.get_rank() == 0:
            path = Path(self.config.trainer.default_local_dir) / f"global_step_{step}"
            self.protocol.save(path / "astribot_protocol.json")
            (path / "resolved_training_config.yaml").write_text(OmegaConf.to_yaml(self.config, resolve=True))


@hydra.main(config_path="config/astribot", config_name="base", version_base=None)
def main(config: DictConfig):
    _, _, world = initialize_global_process_group()
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
    AstribotSFTTrainer(config, mesh).fit()


if __name__ == "__main__": main()
