"""
Evaluate a fine-tuned UniVLA model across config subsets [start_subset, end_subset)
on train/test splits, reusing the vla_align rollout / env / metrics infrastructure.

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=0 python univla_eval.py \
        --pretrained_checkpoint /path/to/univla_finetune_run \
        --action_decoder_path /path/to/action_decoder-XXXX.pt \
        --result_dir ./univla_eval_result \
        --start_subset 0 --end_subset 6

See submit_univla_eval.run for multi-GPU SLURM launch.
"""

import copy
import json
import logging
import os
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["SVT_LOG"] = "0"

import draccus
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# UniVLA imports
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.policy.transformer_utils import MAPBlock

# vla_align imports — reuse env, rollout, and helpers from the lerobot eval codebase
sys.path.insert(0, "/root/vla")
from vla_align.env.config import get_env_cfg, MAX_EPISODE_STEP_WORKSPACE_EVAL
from vla_align.utils.env import build_endless_env
from vla_align.utils.rollout import rollout, calculate_averages
from vla_align.utils.helpers import batch_tensor_to_string
from vla_align.utils.lerobot import (
    image_1_key,
    image_1_robot_state,
    image_1_segmentation_mask_key,
    wrist_image_key,
    task_key,
)


# ============================================================
# ActionDecoder — same architecture as finetune_wiser.py
# ============================================================
ACTION_DIM = 8  # 7 arm joint targets + 1 gripper (Panda WISER)


class ActionDecoder(nn.Module):
    """Mirrors the ActionDecoder from finetune_wiser.py (8-DoF WISER Panda)."""

    def __init__(self, window_size: int = 20, hidden_dim: int = 512):
        super().__init__()
        self.latent_action_pool = MAPBlock(
            n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64
        )
        self.visual_pool = MAPBlock(
            n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, ACTION_DIM * window_size),
            nn.Tanh(),
        )

    def forward(self, latent_action_tokens, visual_embed):
        visual_embed = self.visual_pool(visual_embed)
        latent_action_tokens = latent_action_tokens[:, -4:]
        action_token = self.latent_action_pool(latent_action_tokens, init_embed=visual_embed)
        action = self.proj(action_token)
        return action


# ============================================================
# Config
# ============================================================
@dataclass
class UniVLAEvalConfig:
    # Model paths
    pretrained_checkpoint: str = ""          # Path to fine-tuned UniVLA (merged LoRA) checkpoint
    action_decoder_path: str = ""            # Path to action_decoder-XXXX.pt
    dataset_name: str = "wiser_panda"        # unnorm_key in dataset_statistics.json

    # Action chunk / decoder
    window_size: int = 20                    # Action chunk size (must match training)
    center_crop: bool = True                 # Center crop images (if trained with image aug)

    # Evaluation grid
    start_subset: int = 0                    # First config index (inclusive)
    end_subset: int = 24                     # Last config index (exclusive)
    split: str = "both"                      # "train", "test", or "both"
    eval_rounds: int = 1                     # Rounds per subset
    num_envs: int = 12                       # Parallel envs per subset

    # Output
    result_dir: str = "./univla_eval_result"
    save_video: bool = False                 # Save rollout videos (indices_to_save=None if True)

    # Aggregation-only mode
    aggregate_only: bool = False


# ============================================================
# Image pre-processing (matches UniVLA training / LIBERO eval)
# ============================================================
def _center_crop_image(image_np: np.ndarray, crop_scale: float = 0.9) -> np.ndarray:
    """Center-crop then resize back to original size (PIL-based, no TF dependency)."""
    h, w = image_np.shape[:2]
    new_h = int(h * crop_scale ** 0.5)
    new_w = int(w * crop_scale ** 0.5)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    cropped = image_np[top : top + new_h, left : left + new_w]
    pil_img = Image.fromarray(cropped).resize((w, h), Image.BILINEAR)
    return np.array(pil_img)


# ============================================================
# UniVLA policy wrapper for the rollout() function
# ============================================================
def univla_policy(
    vla,
    processor,
    action_decoder: ActionDecoder,
    norm_stats: dict,
    unnorm_key: str,
    window_size: int = 20,
    center_crop: bool = True,
    device: torch.device = torch.device("cuda"),
):
    """
    Returns a callable with signature ``(obs) -> (action_to_take, expert_action, info_dict)``
    compatible with ``vla_align.utils.rollout.rollout()``.

    Action chunk execution: on every call the wrapper either returns the next
    buffered action or runs a fresh VLA inference when the buffer is exhausted.
    Replan interval = window_size (all chunk actions are consumed before replanning).
    """
    # Per-env action buffer: deque of tensors, each of shape (num_env, ACTION_DIM)
    action_queue: deque = deque()

    # History latent action text tokens per env (list of lists)
    prev_hist_actions: list = []  # will be initialised on first call
    latent_action_detokenize = [f"<ACT_{i}>" for i in range(32)]

    # Action unnorm stats
    action_norm_stats = norm_stats[unnorm_key]["action"]
    mask = np.array(action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool)))
    action_high = np.array(action_norm_stats["q99"])
    action_low = np.array(action_norm_stats["q01"])

    def _obs_to_pil_batch(obs, num_env: int):
        """Convert batched uint8 image tensor (B, H, W, 3) to list of PIL Images."""
        imgs_np = obs[image_1_key].cpu().numpy()  # (B, H, W, 3)
        pil_images = []
        for i in range(num_env):
            img = imgs_np[i]
            if center_crop:
                img = _center_crop_image(img)
            pil_images.append(Image.fromarray(img).convert("RGB"))
        return pil_images

    def forward(obs):
        nonlocal action_queue, prev_hist_actions

        num_env = obs[image_1_key].shape[0]

        # Initialise per-env history on first call
        if len(prev_hist_actions) == 0:
            prev_hist_actions = [[""] for _ in range(num_env)]

        # If buffer is empty, run inference and fill it
        if len(action_queue) == 0:
            inference_start = time.perf_counter()

            pil_images = _obs_to_pil_batch(obs, num_env)
            task_strings = batch_tensor_to_string(obs[task_key])

            all_actions = []  # will be (num_env, window_size, ACTION_DIM)

            # UniVLA generates() only supports batch_size=1, so we loop over envs
            for env_idx in range(num_env):
                # Build prompt with history latent actions
                task_label = task_strings[env_idx]
                hist = prev_hist_actions[env_idx]
                start_idx = min(len(hist), 4)
                prompt_hist = "".join(hist[-start_idx:])

                if len(prompt_hist) > 0:
                    prompt = f"In: What action should the robot take to {task_label.lower()}? History action {prompt_hist}\nOut:"
                else:
                    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

                inputs = processor(prompt, pil_images[env_idx]).to(device, dtype=torch.bfloat16)

                with torch.inference_mode():
                    latent_action, visual_embed, generated_ids = vla.predict_latent_action(
                        **inputs, do_sample=True, temperature=0.75, top_p=0.9
                    )

                # Record history latent action text
                hist_action_str = ""
                for tok_id in generated_ids[0]:
                    hist_action_str += latent_action_detokenize[tok_id.item() - 32001]
                prev_hist_actions[env_idx].append(hist_action_str)

                # Decode continuous actions via ActionDecoder
                with torch.inference_mode():
                    raw_actions = action_decoder(
                        latent_action.float(), visual_embed.float()
                    )  # (1, ACTION_DIM * window_size)
                raw_actions = raw_actions.reshape(window_size, ACTION_DIM).cpu().numpy()

                # Unnormalize
                unnormed = np.where(
                    mask,
                    0.5 * (raw_actions + 1) * (action_high - action_low) + action_low,
                    raw_actions,
                )
                all_actions.append(unnormed)

            # Stack: (num_env, window_size, ACTION_DIM)
            all_actions = np.stack(all_actions, axis=0)

            # Fill the action buffer with per-timestep slices
            for t in range(window_size):
                action_queue.append(
                    torch.from_numpy(all_actions[:, t, :]).float().to(device)
                )

            inference_time = time.perf_counter() - inference_start
            info = {"inference_time": inference_time}
        else:
            info = {"inference_time": 0.0}

        # Pop the next action from the buffer
        action = action_queue.popleft()  # (num_env, ACTION_DIM)
        return action, action, info

    def reset():
        """Clear action buffer and history so a fresh rollout starts cleanly."""
        nonlocal action_queue, prev_hist_actions
        action_queue.clear()
        prev_hist_actions.clear()

    forward.reset = reset
    return forward


# ============================================================
# Main evaluation
# ============================================================
@draccus.wrap()
def run_eval(cfg: UniVLAEvalConfig) -> None:
    result_dir = cfg.result_dir
    if cfg.aggregate_only:
        _aggregate_results(result_dir)
        return

    os.makedirs(result_dir, exist_ok=True)

    # Device
    device = torch.device("cuda")
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # ── Load UniVLA model ──────────────────────────────────
    logging.info("Loading UniVLA model...")
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device).eval()

    # Load dataset statistics for action unnormalization
    stats_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    assert os.path.isfile(stats_path), f"dataset_statistics.json not found at {stats_path}"
    with open(stats_path, "r") as f:
        norm_stats = json.load(f)
    vla.norm_stats = norm_stats
    assert cfg.dataset_name in norm_stats, (
        f"unnorm_key '{cfg.dataset_name}' not in dataset_statistics.json "
        f"(available: {list(norm_stats.keys())})"
    )

    # ── Load ActionDecoder ─────────────────────────────────
    logging.info("Loading ActionDecoder...")
    action_decoder = ActionDecoder(window_size=cfg.window_size).to(device)
    state_dict = torch.load(cfg.action_decoder_path, map_location=device)
    action_decoder.load_state_dict(state_dict)
    action_decoder.eval()

    # ── Build policy wrapper ───────────────────────────────
    wrapped = univla_policy(
        vla=vla,
        processor=processor,
        action_decoder=action_decoder,
        norm_stats=norm_stats,
        unnorm_key=cfg.dataset_name,
        window_size=cfg.window_size,
        center_crop=cfg.center_crop,
        device=device,
    )

    # ── Evaluation loop ───────────────────────────────────
    splits = ["train", "test"] if cfg.split == "both" else [cfg.split]
    logging.info(
        f"Evaluating subsets {cfg.start_subset} to {cfg.end_subset - 1} on splits: {splits}"
    )

    for split in splits:
        for i in range(cfg.start_subset, cfg.end_subset):
            cfg_name = f"config_{i}"
            subset_result_dir = os.path.join(result_dir, f"config_{i}_{split}")

            # Skip if already evaluated
            metrics_file = os.path.join(subset_result_dir, "episode_metrics.json")
            if os.path.exists(metrics_file):
                logging.info(f"Skipping {cfg_name} ({split}) — already evaluated")
                continue

            if os.path.exists(subset_result_dir):
                shutil.rmtree(subset_result_dir)
            os.makedirs(subset_result_dir)

            # Build environment
            scene_cfg = dict(
                robot_init_qpos_noise=0.0,
                cube_size_noise=0.0,
                cfg_name=cfg_name,
                mode=split,
            )
            env_cfg = get_env_cfg(
                num_env=cfg.num_envs,
                max_steps=MAX_EPISODE_STEP_WORKSPACE_EVAL,
                obs_mode="rgb+segmentation",
                scene_cfg_to_overwrite=scene_cfg,
            )
            envs = build_endless_env(env_cfg, record_video=False, data_record_dir="test")

            # Reset policy state (action buffer + history) for each new subset
            wrapped.reset()

            # Rollout
            print("\n" + "=" * 60)
            print(f"Starting Rollout for {cfg_name} ({split})")
            print("=" * 60)

            start_time = time.perf_counter()
            with torch.no_grad():
                performance = rollout(
                    envs,
                    wrapped,
                    round_to_collect=cfg.eval_rounds,
                    demo_saving_dir=subset_result_dir,
                    debug_mode=True,
                    indices_to_save=None if cfg.save_video else [],
                )
            elapsed = time.perf_counter() - start_time

            print("\n" + "=" * 60)
            print(f"Performance for {cfg_name} ({split}) — {elapsed:.1f}s")
            print("=" * 60)
            for key, v in performance.items():
                print(f"  {key}: {v}")

            envs.unwrapped.close()

    logging.info("All subsets evaluated. Done.")


def _aggregate_results(result_dir: str):
    """Compute final aggregated results for train and test splits."""
    for split in ["train", "test"]:
        pattern = os.path.join(result_dir, f"*{split}*")
        final_results = calculate_averages(pattern)
        if final_results:
            out_path = os.path.join(result_dir, f"final_results_{split}.json")
            with open(out_path, "w") as f:
                json.dump(final_results, f, indent=2)
            print(f"Final results saved to {out_path}")
        else:
            print(f"No results found for split '{split}'")


if __name__ == "__main__":
    run_eval()
