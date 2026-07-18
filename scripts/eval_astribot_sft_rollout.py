#!/usr/bin/env python3
"""Rolling episode eval: rollout action chunks and render an MP4.

Video layout (per frame t):
  - Top: 3-camera observation at frame t
  - Bottom: 16 subplots with full-episode GT (orange) and prediction (blue), cursor at t
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from verl.utils.dataset.astribot_lerobot_sft_dataset import (
    ASTRIBOT_IMAGE_SIZES,
    AstribotLeRobotSFTDataset,
    denormalize_vector,
)
from verl.utils.dataset.vla_sft_collate import vla_sft_collate_fn
from verl.workers.actor.action_tokenizer import ActionTokenizer

ACTION_DIM_NAMES = [
    "left_x",
    "left_y",
    "left_z",
    "left_qw",
    "left_qx",
    "left_qy",
    "left_qz",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_qw",
    "right_qx",
    "right_qy",
    "right_qz",
    "right_gripper",
]


def parse_torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported eval dtype: {name}")


def load_eval_model(checkpoint: str | Path, *, dtype: torch.dtype):
    checkpoint = resolve_checkpoint(checkpoint)
    config = AutoConfig.from_pretrained(str(checkpoint), local_files_only=True)
    decode_mode = getattr(config, "astribot_action_decode_mode", None)
    if decode_mode != "parallel":
        raise RuntimeError(
            f"Checkpoint {checkpoint} was not trained with parallel action-block decoding "
            f"(astribot_action_decode_mode={decode_mode!r}). Retrain it with the updated SFT trainer."
        )
    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(checkpoint),
        dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    return processor, model, checkpoint


def resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir() and (path / "config.json").exists():
        return path
    if path.is_dir():
        candidates = sorted(path.glob("global_step_*"), key=lambda p: int(p.name.split("_")[-1]))
        if not candidates:
            raise FileNotFoundError(f"No global_step_* checkpoints under {path}")
        return candidates[-1]
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def tokens_to_actions(
    token_ids: np.ndarray,
    action_tokenizer: ActionTokenizer,
    action_mask: np.ndarray,
    action_q01: np.ndarray,
    action_q99: np.ndarray,
    action_chunks_len: int,
    action_token_len: int,
) -> np.ndarray:
    normalized = action_tokenizer.decode_token_ids_to_actions(token_ids.reshape(-1))
    denorm = denormalize_vector(
        normalized,
        np.tile(action_mask, action_chunks_len),
        np.tile(action_q01, action_chunks_len),
        np.tile(action_q99, action_chunks_len),
    )
    return denorm.reshape(action_chunks_len, action_token_len).astype(np.float32)


def predict_action_chunk(
    model,
    batch: dict[str, torch.Tensor],
    *,
    action_0_id: int,
    max_prompt_length: int,
    action_chunks_len: int,
    action_token_len: int,
    action_tokenizer: ActionTokenizer,
    action_mask: np.ndarray,
    action_q01: np.ndarray,
    action_q99: np.ndarray,
) -> np.ndarray:
    """Generate every action token in one full-attention action-block forward."""
    action_len = action_chunks_len * action_token_len
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            astribot_parallel_action=True,
            prompt_length=max_prompt_length,
            action_length=action_len,
            action_slot_id=action_0_id,
        )
    logits = outputs.action_logits[..., action_0_id : action_0_id + 256].float()
    if logits.shape[1] != action_len:
        raise ValueError(f"Expected {action_len} parallel action logits, got {logits.shape}")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite parallel action logits")
    token_ids = (logits.argmax(dim=-1) + action_0_id)[0].cpu().numpy()

    return tokens_to_actions(
        token_ids,
        action_tokenizer,
        action_mask,
        action_q01,
        action_q99,
        action_chunks_len,
        action_token_len,
    )


def load_episode_gt(dataset: AstribotLeRobotSFTDataset, episode_index: int) -> np.ndarray:
    df = dataset._load_episode_df(episode_index)
    return np.stack(df["action"].to_numpy()).astype(np.float32)


def stitch_rollout_predictions(
    *,
    episode_len: int,
    rollout_stride: int,
    action_frame_stride: int,
    action_chunks_len: int,
    rollout_frames: list[int],
    rollout_preds: list[np.ndarray],
) -> np.ndarray:
    pred_full = np.full((episode_len, 16), np.nan, dtype=np.float32)
    for start_frame, pred_chunk in zip(rollout_frames, rollout_preds, strict=True):
        for i in range(action_chunks_len):
            frame_idx = start_frame + i * action_frame_stride
            if frame_idx < episode_len:
                pred_full[frame_idx] = pred_chunk[i]
    return pred_full


def load_observation_strip(
    dataset: AstribotLeRobotSFTDataset,
    episode_index: int,
    frame_index: int,
    target_width: int,
) -> np.ndarray:
    images = [
        dataset._load_image(episode_index, frame_index, key)
        for key in dataset.image_keys
    ]
    total_w = sum(img.width for img in images)
    max_h = max(img.height for img in images)
    canvas = Image.new("RGB", (total_w, max_h))
    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width
    scale = target_width / total_w
    target_h = max(1, int(max_h * scale))
    canvas = canvas.resize((target_width, target_h), Image.BILINEAR)
    return np.asarray(canvas)


def render_action_panel(
    gt_full: np.ndarray,
    pred_full: np.ndarray,
    current_frame: int,
    dim_names: list[str],
    figsize: tuple[float, float],
    dpi: int,
) -> np.ndarray:
    episode_len = gt_full.shape[0]
    frames = np.arange(episode_len)

    fig, axes = plt.subplots(4, 4, figsize=figsize, sharex=True)
    fig.suptitle(f"frame {current_frame} / {episode_len - 1}", fontsize=11)

    for dim in range(16):
        ax = axes[dim // 4, dim % 4]
        ax.plot(frames, gt_full[:, dim], color="#d95f02", linewidth=1.2, label="GT")
        pred_mask = np.isfinite(pred_full[:, dim])
        if pred_mask.any():
            ax.plot(
                frames[pred_mask],
                pred_full[pred_mask, dim],
                color="#1f77b4",
                linewidth=1.2,
                label="pred",
            )
        ax.axvline(current_frame, color="red", linewidth=0.8, alpha=0.8)
        ax.set_title(dim_names[dim], fontsize=8)
        ax.grid(True, alpha=0.25)
        if dim == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.canvas.draw()
    plot_img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return plot_img


def compose_video_frame(
    obs_rgb: np.ndarray,
    plot_rgb: np.ndarray,
    header_h: int = 28,
) -> np.ndarray:
    width = max(obs_rgb.shape[1], plot_rgb.shape[1])
    obs_pad = np.ones((obs_rgb.shape[0], width, 3), dtype=np.uint8) * 255
    obs_pad[:, : obs_rgb.shape[1]] = obs_rgb

    plot_pad = np.ones((plot_rgb.shape[0], width, 3), dtype=np.uint8) * 255
    plot_pad[:, : plot_rgb.shape[1]] = plot_rgb

    header = np.ones((header_h, width, 3), dtype=np.uint8) * 245
    return np.vstack([header, obs_pad, plot_pad])


def resize_rgb(img: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((width, height), Image.BILINEAR))


def pad_video_frame_to_even(img: np.ndarray) -> np.ndarray:
    """yuv420p requires even frame dimensions."""
    pad_h = img.shape[0] % 2
    pad_w = img.shape[1] % 2
    if not (pad_h or pad_w):
        return img
    return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=255)


def write_eval_video(
    *,
    output_path: Path,
    dataset: AstribotLeRobotSFTDataset,
    episode_index: int,
    gt_full: np.ndarray,
    pred_full: np.ndarray,
    fps: int,
    frame_stride: int,
    width: int,
    plot_figsize: tuple[float, float],
    plot_dpi: int,
) -> None:
    episode_len = gt_full.shape[0]
    video_frames = list(range(0, episode_len, frame_stride))
    if video_frames[-1] != episode_len - 1:
        video_frames.append(episode_len - 1)

    sample_obs = load_observation_strip(dataset, episode_index, 0, width)
    sample_plot = render_action_panel(
        gt_full, pred_full, 0, ACTION_DIM_NAMES, plot_figsize, plot_dpi
    )
    sample = pad_video_frame_to_even(compose_video_frame(sample_obs, sample_plot))
    height, width = sample.shape[:2]

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    try:
        for frame_idx in video_frames:
            obs = load_observation_strip(dataset, episode_index, frame_idx, width)
            plot = render_action_panel(
                gt_full, pred_full, frame_idx, ACTION_DIM_NAMES, plot_figsize, plot_dpi
            )
            if plot.shape[1] != width:
                plot = resize_rgb(plot, width, plot.shape[0])
            composed = pad_video_frame_to_even(compose_video_frame(obs, plot))
            writer.append_data(composed)
    finally:
        writer.close()


def save_static_overview(
    gt_full: np.ndarray,
    pred_full: np.ndarray,
    output_path: Path,
) -> None:
    episode_len = gt_full.shape[0]
    frames = np.arange(episode_len)
    fig, axes = plt.subplots(4, 4, figsize=(18, 12), sharex=True)
    fig.suptitle("Full episode GT (orange) vs prediction (blue)", fontsize=13)
    for dim in range(16):
        ax = axes[dim // 4, dim % 4]
        ax.plot(frames, gt_full[:, dim], color="#d95f02", linewidth=1.0, label="GT")
        pred_mask = np.isfinite(pred_full[:, dim])
        if pred_mask.any():
            ax.plot(
                frames[pred_mask],
                pred_full[pred_mask, dim],
                color="#1f77b4",
                linewidth=1.0,
                label="pred",
            )
        ax.set_title(ACTION_DIM_NAMES[dim], fontsize=9)
        ax.grid(True, alpha=0.25)
        if dim == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling astribot SFT evaluation")
    parser.add_argument("--checkpoint", default="/media/damoxing/ckp/last_r1_astribot_sft")
    parser.add_argument(
        "--dataset-root",
        default="/media/damoxing/datasets/vae4d/lerobot-vae4d-org/astribot/centrifuge_multidrop-f2",
    )
    parser.add_argument("--statistics-path", default="assets/astribot_dataset_statistics.json")
    parser.add_argument("--dataset-name", default="astribot_centrifuge_multidrop")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-rollout-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-prompt-length", type=int, default=576)
    parser.add_argument("--action-token-len", type=int, default=16)
    parser.add_argument("--action-chunks-len", type=int, default=8)
    parser.add_argument("--action-frame-stride", type=int, default=4)
    parser.add_argument("--need-to-sub", type=int, default=3)
    parser.add_argument(
        "--inference-mode",
        choices=["parallel"],
        default="parallel",
        help="LaST-R1 one-forward full-attention action-block decoding",
    )
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-frame-stride", type=int, default=2, help="Render every N frames in MP4")
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Also render an MP4. Disabled by default because rendering is slow.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--eval-dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Model dtype for eval. Use fp32 to diagnose bf16 numerical issues.",
    )
    args = parser.parse_args()

    repo_root = _REPO_ROOT
    stats_path = repo_root / args.statistics_path

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)[args.dataset_name]
    action_mask = np.array(stats["action"]["mask"], dtype=bool)
    action_q01 = np.array(stats["action"]["q01"], dtype=np.float32)
    action_q99 = np.array(stats["action"]["q99"], dtype=np.float32)

    eval_dtype = parse_torch_dtype(args.eval_dtype)
    processor, model, ckpt_path = load_eval_model(args.checkpoint, dtype=eval_dtype)
    action_tokenizer = ActionTokenizer(processor.tokenizer, need_to_sub=args.need_to_sub)
    action_0_id = processor.tokenizer.vocab["<action_0>"]

    model = model.to(args.device)
    model.eval()

    image_keys = [
        "observation.images.cam_main",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    dataset = AstribotLeRobotSFTDataset(
        dataset_root=args.dataset_root,
        processor=processor,
        statistics_path=stats_path,
        dataset_name=args.dataset_name,
        image_keys=image_keys,
        action_tokenizer=action_tokenizer,
        action_token_len=args.action_token_len,
        action_chunks_len=args.action_chunks_len,
        action_frame_stride=args.action_frame_stride,
        image_sizes=ASTRIBOT_IMAGE_SIZES,
        episode_indices=[args.episode_index],
    )

    episode_len = dataset.episode_lengths[args.episode_index]
    rollout_stride = args.action_chunks_len * args.action_frame_stride
    output_dir = Path(
        args.output_dir or (repo_root / "eval_outputs" / f"ep{args.episode_index:04d}_{ckpt_path.name}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint: {ckpt_path}")
    print(f"episode {args.episode_index} length={episode_len}, rollout_stride={rollout_stride}")
    print(f"inference_mode={args.inference_mode}")
    print(f"eval_dtype={args.eval_dtype}")

    gt_full = load_episode_gt(dataset, args.episode_index)
    rollout_frames: list[int] = []
    rollout_preds: list[np.ndarray] = []
    metrics: list[dict] = []

    frame = args.start_frame
    step_idx = 0
    while frame < episode_len:
        if args.max_rollout_steps is not None and step_idx >= args.max_rollout_steps:
            break

        sample = dataset.build_sample(args.episode_index, frame)
        gt_chunk = sample["raw_action_chunk"].numpy()
        valid_mask = sample["action_loss_mask"].numpy()

        batch = vla_sft_collate_fn(
            [sample],
            pad_token_id=processor.tokenizer.pad_token_id,
            max_prompt_length=args.max_prompt_length,
        )
        batch = {k: v.to(args.device) for k, v in batch.items()}

        pred_chunk = predict_action_chunk(
            model,
            batch,
            action_0_id=action_0_id,
            max_prompt_length=args.max_prompt_length,
            action_chunks_len=args.action_chunks_len,
            action_token_len=args.action_token_len,
            action_tokenizer=action_tokenizer,
            action_mask=action_mask,
            action_q01=action_q01,
            action_q99=action_q99,
        )

        valid = valid_mask > 0
        mae = float(np.mean(np.abs(pred_chunk[valid] - gt_chunk[valid]))) if valid.any() else float("nan")
        mse = float(np.mean((pred_chunk[valid] - gt_chunk[valid]) ** 2)) if valid.any() else float("nan")

        rollout_frames.append(frame)
        rollout_preds.append(pred_chunk)
        metrics.append({"step": step_idx, "frame_index": frame, "mae": mae, "mse": mse})
        print(f"rollout {step_idx:03d} @ frame {frame:5d} | chunk MAE={mae:.5f}")

        frame += rollout_stride
        step_idx += 1

    pred_full = stitch_rollout_predictions(
        episode_len=episode_len,
        rollout_stride=rollout_stride,
        action_frame_stride=args.action_frame_stride,
        action_chunks_len=args.action_chunks_len,
        rollout_frames=rollout_frames,
        rollout_preds=rollout_preds,
    )

    pred_mask = np.isfinite(pred_full).all(axis=1)
    if pred_mask.any():
        episode_mae = float(np.mean(np.abs(pred_full[pred_mask] - gt_full[pred_mask])))
        episode_mse = float(np.mean((pred_full[pred_mask] - gt_full[pred_mask]) ** 2))
    else:
        episode_mae = episode_mse = float("nan")

    video_path = output_dir / f"rollout_{args.inference_mode}.mp4" if args.save_video else None
    overview_path = output_dir / f"overview_{args.inference_mode}.png"

    print(f"episode MAE={episode_mae:.5f} on {int(pred_mask.sum())} predicted frames")
    if video_path is not None:
        print(f"writing video -> {video_path}")
        write_eval_video(
            output_path=video_path,
            dataset=dataset,
            episode_index=args.episode_index,
            gt_full=gt_full,
            pred_full=pred_full,
            fps=args.video_fps,
            frame_stride=args.video_frame_stride,
            width=args.video_width,
            plot_figsize=(16, 9),
            plot_dpi=100,
        )
    print(f"writing overview -> {overview_path}")
    save_static_overview(gt_full, pred_full, overview_path)

    summary = {
        "checkpoint": str(ckpt_path),
        "episode_index": args.episode_index,
        "episode_length": episode_len,
        "rollout_stride": rollout_stride,
        "inference_mode": args.inference_mode,
        "eval_dtype": args.eval_dtype,
        "num_rollout_steps": len(metrics),
        "episode_mae": episode_mae,
        "episode_mse": episode_mse,
        "video_path": str(video_path) if video_path is not None else None,
        "overview_path": str(overview_path),
        "steps": metrics,
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if video_path is not None:
        print(f"saved video: {video_path}")
    print(f"saved overview: {overview_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
