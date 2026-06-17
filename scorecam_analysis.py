"""
ScoreCAM Analysis for HRRL-Sensorial Agent.

Gradient-free attention heatmaps: masks the input with each channel's
activation map and measures the change in the network's output score.

1. RND Predictor ScoreCAM — What image regions generate curiosity (intrinsic reward).
2. Actor-Critic ScoreCAM — What regions the policy values most (V(s)).

Usage:
    python scorecam_analysis.py [--experiment hrrl_sensorial_v1] [--warmup_episodes 10]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.rnd_module import RNDModule
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict

from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components
from sf_examples.vizdoom.train_custom_vizdoom_env import register_custom_doom_env

_RND_BATCH = 32

# ═══════════════════════════════════════════════════════════════════════
# Model loading (shared boilerplate)
# ═══════════════════════════════════════════════════════════════════════

def build_cfg(experiment, train_dir):
    register_vizdoom_components()
    register_custom_doom_env()
    with open(os.path.join(train_dir, experiment, "config.json")) as f:
        cfg = AttrDict(json.load(f))
    cfg.num_envs = 1
    cfg.no_render = True
    cfg.eval_deterministic = False
    cfg.policy_index = 0
    cfg.load_checkpoint_kind = "latest"
    cfg.max_num_frames = None
    cfg.max_num_episodes = None
    cfg.cli_args = {}
    return cfg


def load_actor_critic(cfg, env, device):
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(
        Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*"
    )
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    if not checkpoint_dict:
        raise RuntimeError("Could not load checkpoint")
    actor_critic.load_state_dict(checkpoint_dict["model"])
    print(f"  Loaded checkpoint: {checkpoint_dict['env_steps']} env steps")
    return actor_critic


def create_rnd_module(cfg, env, device):
    rnd = RNDModule(cfg, env.observation_space, device)
    rnd.train()
    return rnd


# ═══════════════════════════════════════════════════════════════════════
# Synthetic degradation (Chebyshev glaucoma on 84×84)
# ═══════════════════════════════════════════════════════════════════════

_RANK_84 = None

def _get_rank_84():
    global _RANK_84
    if _RANK_84 is None:
        h, w = 84, 84
        cy, cx = h // 2, w // 2
        y, x = np.indices((h, w))
        _RANK_84 = np.maximum(np.abs(y - cy), np.abs(x - cx))
    return _RANK_84


def apply_degradation(frame_84x84, alpha):
    if alpha <= 0:
        return frame_84x84.copy()
    if alpha >= 1.0:
        return np.zeros_like(frame_84x84)
    total = 84 * 84
    erased = int(alpha * total)
    threshold = int(np.sqrt(erased) / 2)
    result = np.zeros_like(frame_84x84)
    keep = _get_rank_84() >= threshold
    result[keep] = frame_84x84[keep]
    return result


# ═══════════════════════════════════════════════════════════════════════
# ScoreCAM core — gradient-free
# ═══════════════════════════════════════════════════════════════════════

class ScoreCAM:
    """
    Score-weighted Class Activation Mapping (gradient-free).

    For each channel k of the target conv layer:
      1. Upsample activation A_k to input size, normalize to [0, 1]
      2. Mask the input: x_masked = x · A_k_upsampled
      3. Forward pass with masked input → score_k
      4. Weight = softmax(scores)
    CAM = ReLU( Σ weight_k · A_k )
    """

    def __init__(self, get_activations_fn, score_fn):
        self.get_activations_fn = get_activations_fn
        self.score_fn = score_fn

    @torch.no_grad()
    def generate(self, obs_tensor):
        """
        Args:
            obs_tensor: (1, 1, 84, 84) float tensor.
        Returns:
            cam: (84, 84) numpy array in [0, 1].
            baseline_score: scalar (score of the unmasked input).
        """
        # 1. Activation maps from target conv layer
        activations = self.get_activations_fn(obs_tensor)   # (C, H_act, W_act)
        C = activations.shape[0]

        # 2. Upsample each channel to (84, 84)
        acts_up = F.interpolate(
            activations.unsqueeze(0), size=(84, 84),
            mode="bilinear", align_corners=False,
        ).squeeze(0)                                        # (C, 84, 84)

        # Normalize each channel independently to [0, 1]
        for k in range(C):
            a_min, a_max = acts_up[k].min(), acts_up[k].max()
            if (a_max - a_min) > 1e-8:
                acts_up[k] = (acts_up[k] - a_min) / (a_max - a_min)
            else:
                acts_up[k] = 0.0

        # 3. Masked inputs — batch all channels at once
        obs_expanded = obs_tensor.expand(C, -1, -1, -1)    # (C, 1, 84, 84)
        masks = acts_up.unsqueeze(1)                        # (C, 1, 84, 84)
        masked_inputs = obs_expanded * masks                # (C, 1, 84, 84)

        # 4. Score each masked input
        scores = self.score_fn(masked_inputs)               # (C,)

        # Baseline (unmasked)
        baseline = self.score_fn(obs_tensor)                # (1,)

        # 5. Softmax weights
        weights = F.softmax(scores, dim=0)                  # (C,)

        # 6. Weighted combination
        cam = (weights.view(C, 1, 1) * acts_up).sum(dim=0) # (84, 84)
        cam = F.relu(cam)

        cam_np = cam.cpu().numpy()
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()

        return cam_np, baseline.item()


# ═══════════════════════════════════════════════════════════════════════
# RND Predictor ScoreCAM builder
# ═══════════════════════════════════════════════════════════════════════

def build_rnd_scorecam(rnd_module, device):
    """
    Target layer: predictor.feature_extractor[4] (Conv2d 64ch, 7×7).
    Score: MSE(predictor, target) = raw intrinsic reward.
    """

    def get_activations(obs_tensor):
        was_training = rnd_module.obs_rms.training
        rnd_module.obs_rms.eval()
        obs_norm = rnd_module.obs_rms(obs_tensor)
        obs_norm = torch.clamp(obs_norm, -5, 5)
        if was_training:
            rnd_module.obs_rms.train()
        # Forward through conv layers 0-5 (Conv,ReLU × 3)
        x = obs_norm
        for i in range(6):
            x = rnd_module.predictor.feature_extractor[i](x)
        return x.squeeze(0)  # (64, 7, 7)

    def score_fn(masked_batch):
        was_training = rnd_module.obs_rms.training
        rnd_module.obs_rms.eval()
        obs_norm = rnd_module.obs_rms(masked_batch)
        obs_norm = torch.clamp(obs_norm, -5, 5)
        if was_training:
            rnd_module.obs_rms.train()
        pred = rnd_module.predictor(obs_norm)
        target = rnd_module.target(obs_norm)
        return (pred - target).pow(2).mean(dim=-1)  # (N,)

    return ScoreCAM(get_activations, score_fn)


# ═══════════════════════════════════════════════════════════════════════
# Actor-Critic ScoreCAM builder
# ═══════════════════════════════════════════════════════════════════════

def _build_nojit_conv_head(actor_critic, device):
    """
    Non-JIT copy of the actor-critic's conv head for activation extraction.
    The encoder uses torch.jit.script internally, which blocks hooks.
    """
    jit_enc = actor_critic.encoder.basic_encoder.enc
    sd = jit_enc.conv_head.state_dict()

    w0 = sd["0.weight"]  # (32, C_in, 8, 8)
    w2 = sd["2.weight"]  # (64, 32, 4, 4)
    w4 = sd["4.weight"]  # (128, 64, 3, 3)

    conv_head = nn.Sequential(
        nn.Conv2d(w0.shape[1], w0.shape[0], w0.shape[2], stride=4), nn.ReLU(),
        nn.Conv2d(w2.shape[1], w2.shape[0], w2.shape[2], stride=2), nn.ReLU(),
        nn.Conv2d(w4.shape[1], w4.shape[0], w4.shape[2], stride=2), nn.ReLU(),
    )
    conv_head.load_state_dict(sd)
    conv_head.to(device).eval()
    return conv_head


def build_ac_scorecam(actor_critic, cfg, device):
    """
    Target layer: last Conv2d (128 channels, 4×4 for 84×84 input).
    Score: V(s) — state value estimate.
    """
    conv_head = _build_nojit_conv_head(actor_critic, device)
    rnn_size = get_rnn_size(cfg)

    def _normalize_obs(obs_batch):
        obs = obs_batch.clone()
        obs_scale = getattr(cfg, "obs_scale", 1.0)
        if abs(obs_scale - 1.0) > 1e-5:
            obs = obs / obs_scale
        if getattr(cfg, "normalize_input", False):
            normalizer = actor_critic.obs_normalizer
            if normalizer.running_mean_std is not None:
                rms = normalizer.running_mean_std.running_mean_std["obs"]
                mean = rms.running_mean.float()
                var = rms.running_var.float()
                std = torch.sqrt(var + 1e-5)
                obs = (obs - mean) / std
                obs = obs.clamp(-5, 5)
        return obs

    def get_activations(obs_tensor):
        obs_norm = _normalize_obs(obs_tensor)
        acts = conv_head(obs_norm)      # (1, 128, 4, 4)
        return acts.squeeze(0)          # (128, 4, 4)

    def score_fn(masked_batch):
        N = masked_batch.shape[0]
        obs_norm = _normalize_obs(masked_batch)
        enc_out = actor_critic.encoder({"obs": obs_norm})
        rnn_states = torch.zeros(N, rnn_size, dtype=torch.float32, device=device)
        core_out, _ = actor_critic.core(enc_out, rnn_states)
        dec_out = actor_critic.decoder(core_out)
        return actor_critic.critic_linear(dec_out).squeeze(-1)  # (N,)

    return ScoreCAM(get_activations, score_fn)


# ═══════════════════════════════════════════════════════════════════════
# Episode runner — collect frames with metadata
# ═══════════════════════════════════════════════════════════════════════

def warmup_rnd(actor_critic, rnd_module, env, env_info, cfg, device, num_episodes):
    for ep in range(num_episodes):
        obs, _ = env.reset()
        _ = obs.pop("action_mask", None)
        rnn_states = torch.zeros(1, get_rnn_size(cfg), dtype=torch.float32, device=device)
        obs_buffer, dones_buffer = [], []
        n_steps = 0

        with torch.no_grad():
            while True:
                normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
                policy_outputs = actor_critic(normalized_obs, rnn_states)
                actions = policy_outputs["actions"]
                rnn_states = policy_outputs["new_rnn_states"]
                if actions.ndim == 1:
                    actions = unsqueeze_tensor(actions, dim=-1)
                actions = preprocess_actions(env_info, actions)

                obs_buffer.append(obs["obs"][0].float().to(device))
                obs, rew, terminated, truncated, infos = env.step(actions)
                _ = obs.pop("action_mask", None)
                dones = make_dones(terminated, truncated)
                done = dones[0].item()
                dones_buffer.append(done)
                n_steps += 1

                if len(obs_buffer) >= _RND_BATCH or done:
                    batch = torch.stack(obs_buffer)
                    batch_dones = torch.tensor(dones_buffer, dtype=torch.bool, device=device)
                    rnd_module.calculate_rewards({"obs": batch}, dones=batch_dones)
                    with torch.enable_grad():
                        rnd_module.update({"obs": batch}, batch_dones)
                    obs_buffer.clear()
                    dones_buffer.clear()

                if done:
                    break

        print(f"  Warmup {ep+1}/{num_episodes}: {n_steps} steps")


def collect_frames(actor_critic, rnd_module, env, env_info, cfg, device,
                   num_episodes=1, train_rnd=True):
    all_frames = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        _ = obs.pop("action_mask", None)
        rnn_states = torch.zeros(1, get_rnn_size(cfg), dtype=torch.float32, device=device)
        obs_buffer, dones_buffer = [], []
        step_data = []

        with torch.no_grad():
            while True:
                normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
                policy_outputs = actor_critic(normalized_obs, rnn_states)
                actions = policy_outputs["actions"]
                new_rnn_states = policy_outputs["new_rnn_states"]
                if actions.ndim == 1:
                    actions = unsqueeze_tensor(actions, dim=-1)
                actions = preprocess_actions(env_info, actions)

                frame_np = obs["obs"][0, 0].cpu().numpy()
                obs_buffer.append(obs["obs"][0].float().to(device))
                step_data.append({
                    "frame": frame_np,
                    "z_homeostasis": 1.0,
                    "health": 0.0,
                    "intrinsic_reward": 0.0,
                })

                obs, rew, terminated, truncated, infos = env.step(actions)
                _ = obs.pop("action_mask", None)
                dones = make_dones(terminated, truncated)
                done = dones[0].item()
                dones_buffer.append(done)
                rnn_states = new_rnn_states

                info = infos[0] if infos else {}
                extra = info.get("episode_extra_stats", {})
                step_data[-1]["z_homeostasis"] = extra.get("z_homeostasis", 1.0)
                step_data[-1]["health"] = info.get("HEALTH", 0.0)

                if len(obs_buffer) >= _RND_BATCH or done:
                    batch = torch.stack(obs_buffer)
                    batch_dones = torch.tensor(dones_buffer, dtype=torch.bool, device=device)
                    rnd_rewards = rnd_module.calculate_rewards({"obs": batch}, dones=batch_dones)
                    if train_rnd:
                        with torch.enable_grad():
                            rnd_module.update({"obs": batch}, batch_dones)
                    start_idx = len(step_data) - len(obs_buffer)
                    for i in range(len(obs_buffer)):
                        r = rnd_rewards[i].item()
                        if np.isnan(r):
                            r = 0.0
                        step_data[start_idx + i]["intrinsic_reward"] = r
                    obs_buffer.clear()
                    dones_buffer.clear()

                if done:
                    break

        n_medkits = sum(
            1 for i in range(1, len(step_data))
            if step_data[i]["health"] > step_data[i - 1]["health"]
        )
        print(f"  Episode {ep+1}: {len(step_data)} steps, {n_medkits} medkits")
        all_frames.extend(step_data)

    return all_frames


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════

def overlay_heatmap(frame, heatmap, alpha=0.5, cmap="jet"):
    frame_f = frame.astype(np.float32)
    if frame_f.max() > 1:
        frame_f /= 255.0
    frame_rgb = np.stack([frame_f] * 3, axis=-1)
    colormap = plt.colormaps.get_cmap(cmap)
    heatmap_rgb = colormap(heatmap)[:, :, :3]
    blended = (1 - alpha) * frame_rgb + alpha * heatmap_rgb
    return np.clip(blended, 0, 1)


plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ── Figure 1: Controlled degradation ─────────────────────────────────

def plot_controlled(clean_frame, rnd_cam, ac_cam, device, output_dir):
    levels = [
        (0.0,  "Saudável (α=0%)"),
        (0.3,  "Leve (α=30%)"),
        (0.7,  "Severo (α=70%)"),
        (1.0,  "Cego (α=100%)"),
    ]
    n = len(levels)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 11))

    for col, (alpha, label) in enumerate(levels):
        degraded = apply_degradation(clean_frame, alpha)
        obs_t = torch.from_numpy(degraded).float().unsqueeze(0).unsqueeze(0).to(device)

        axes[0, col].imshow(degraded, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(label, fontweight="bold")
        axes[0, col].axis("off")

        cam_rnd, r_int = rnd_cam.generate(obs_t)
        axes[1, col].imshow(overlay_heatmap(degraded, cam_rnd))
        axes[1, col].set_title(f"r_int = {r_int:.4f}", fontsize=10)
        axes[1, col].axis("off")

        cam_ac, v_s = ac_cam.generate(obs_t)
        axes[2, col].imshow(overlay_heatmap(degraded, cam_ac))
        axes[2, col].set_title(f"V(s) = {v_s:.3f}", fontsize=10)
        axes[2, col].axis("off")

    row_labels = ["Observação", "RND ScoreCAM\n(Curiosidade)", "Actor-Critic\nScoreCAM (Valor)"]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=12, fontweight="bold",
                                 rotation=0, labelpad=100, va="center")

    fig.suptitle("ScoreCAM: Atenção Visual vs Nível de Degradação",
                 fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0.12, 0, 1, 0.96])

    path = os.path.join(output_dir, "fig_scorecam_controlled.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 2: Natural gameplay ────────────────────────────────────────

def plot_natural(frames_data, rnd_cam, ac_cam, device, output_dir):
    buckets = {
        "Saudável\n(z > 0.95)":    (0.95, 1.01),
        "Leve\n(0.6 < z < 0.8)":   (0.60, 0.80),
        "Severo\n(0.3 < z < 0.5)":  (0.30, 0.50),
        "Crítico\n(z < 0.15)":      (0.00, 0.15),
    }

    selected = []
    for label, (lo, hi) in buckets.items():
        candidates = [d for d in frames_data if lo <= d["z_homeostasis"] < hi]
        if candidates:
            candidates.sort(key=lambda d: d["intrinsic_reward"])
            selected.append((label, candidates[len(candidates) // 2]))
        else:
            print(f"  Warning: no frames for '{label.split(chr(10))[0]}'")

    if len(selected) < 2:
        print("  Not enough degradation variety. Skipping.")
        return

    n = len(selected)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 11))
    if n == 1:
        axes = axes.reshape(3, 1)

    for col, (label, data) in enumerate(selected):
        frame = data["frame"]
        z = data["z_homeostasis"]
        r_int = data["intrinsic_reward"]
        obs_t = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0).to(device)

        axes[0, col].imshow(frame, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(f"{label}\nz={z:.2f}  r_int={r_int:.3f}",
                               fontweight="bold", fontsize=10)
        axes[0, col].axis("off")

        cam_rnd, _ = rnd_cam.generate(obs_t)
        axes[1, col].imshow(overlay_heatmap(frame, cam_rnd))
        axes[1, col].axis("off")

        cam_ac, v_s = ac_cam.generate(obs_t)
        axes[2, col].imshow(overlay_heatmap(frame, cam_ac))
        axes[2, col].set_title(f"V(s)={v_s:.3f}", fontsize=9)
        axes[2, col].axis("off")

    row_labels = ["Frame do\nGameplay", "RND ScoreCAM\n(Curiosidade)", "Actor-Critic\nScoreCAM (Valor)"]
    for row, lbl in enumerate(row_labels):
        axes[row, 0].set_ylabel(lbl, fontsize=12, fontweight="bold",
                                 rotation=0, labelpad=100, va="center")

    fig.suptitle("ScoreCAM em Frames Naturais do Gameplay",
                 fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0.12, 0, 1, 0.96])

    path = os.path.join(output_dir, "fig_scorecam_natural.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 3: Heatmap strip ──────────────────────────────────────────

def plot_heatmap_strip(clean_frame, rnd_cam, ac_cam, device, output_dir):
    levels = [
        (0.0, "α = 0%"), (0.2, "α = 20%"), (0.4, "α = 40%"),
        (0.6, "α = 60%"), (0.8, "α = 80%"), (1.0, "α = 100%"),
    ]
    n = len(levels)
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))

    for col, (alpha, label) in enumerate(levels):
        degraded = apply_degradation(clean_frame, alpha)
        obs_t = torch.from_numpy(degraded).float().unsqueeze(0).unsqueeze(0).to(device)
        cam_rnd, r_int = rnd_cam.generate(obs_t)
        cam_ac, v_s = ac_cam.generate(obs_t)

        axes[0, col].imshow(degraded, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(label, fontweight="bold", fontsize=11)
        axes[0, col].axis("off")

        axes[1, col].imshow(cam_rnd, cmap="jet", vmin=0, vmax=1)
        axes[1, col].set_title(f"r={r_int:.4f}", fontsize=9)
        axes[1, col].axis("off")

        im = axes[2, col].imshow(cam_ac, cmap="jet", vmin=0, vmax=1)
        axes[2, col].set_title(f"V={v_s:.3f}", fontsize=9)
        axes[2, col].axis("off")

    axes[0, 0].set_ylabel("Obs", fontsize=11, fontweight="bold",
                           rotation=0, labelpad=40, va="center")
    axes[1, 0].set_ylabel("RND\nScoreCAM", fontsize=11, fontweight="bold",
                           rotation=0, labelpad=55, va="center")
    axes[2, 0].set_ylabel("V(s)\nScoreCAM", fontsize=11, fontweight="bold",
                           rotation=0, labelpad=55, va="center")

    fig.colorbar(im, ax=axes[2, :].tolist(), shrink=0.8, label="Ativação ScoreCAM")
    fig.suptitle("ScoreCAM: Progressão da Degradação Visual",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, "fig_scorecam_heatmap_strip.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ScoreCAM analysis for HRRL-Sensorial")
    parser.add_argument("--experiment", default="hrrl_sensorial_v1")
    parser.add_argument("--train_dir", default="train_dir")
    parser.add_argument("--output_dir", default="figures")
    parser.add_argument("--warmup_episodes", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print("HRRL-Sensorial — ScoreCAM Analysis")
    print("=" * 60)

    # 1. Load
    print("\n[1/5] Loading model...")
    cfg = build_cfg(args.experiment, args.train_dir)
    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode=None,
    )
    env_info = extract_env_info(env, cfg)
    actor_critic = load_actor_critic(cfg, env, device)
    rnd_module = create_rnd_module(cfg, env, device)

    # 2. Warmup
    print(f"\n[2/5] RND warmup ({args.warmup_episodes} episodes)...")
    warmup_rnd(actor_critic, rnd_module, env, env_info, cfg, device, args.warmup_episodes)

    # 3. Collect frames
    print(f"\n[3/5] Collecting frames ({args.eval_episodes} episodes)...")
    all_frames = collect_frames(
        actor_critic, rnd_module, env, env_info, cfg, device,
        num_episodes=args.eval_episodes, train_rnd=True,
    )
    env.close()
    print(f"  Total frames: {len(all_frames)}")
    z_vals = [d["z_homeostasis"] for d in all_frames]
    print(f"  z_homeostasis: min={min(z_vals):.3f}, mean={np.mean(z_vals):.3f}, max={max(z_vals):.3f}")

    # 4. Build ScoreCAM
    print("\n[4/5] Building ScoreCAM modules...")
    rnd_cam = build_rnd_scorecam(rnd_module, device)
    ac_cam = build_ac_scorecam(actor_critic, cfg, device)

    clean_candidates = [d for d in all_frames if d["z_homeostasis"] > 0.98]
    clean_frame = (clean_candidates[len(clean_candidates) // 2]["frame"]
                   if clean_candidates else all_frames[0]["frame"])
    print(f"  Reference frame: z={'> 0.98' if clean_candidates else 'fallback'}")

    # 5. Figures
    print("\n[5/5] Generating ScoreCAM figures...")

    print("  [Fig 1] Controlled degradation...")
    plot_controlled(clean_frame, rnd_cam, ac_cam, device, args.output_dir)

    print("  [Fig 2] Natural gameplay frames...")
    plot_natural(all_frames, rnd_cam, ac_cam, device, args.output_dir)

    print("  [Fig 3] Heatmap strip...")
    plot_heatmap_strip(clean_frame, rnd_cam, ac_cam, device, args.output_dir)

    print("\nDone! Figures saved to:", args.output_dir)


if __name__ == "__main__":
    main()
