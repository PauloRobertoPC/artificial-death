"""
GradCAM / Saliency Analysis for HRRL-Sensorial Agent.

Generates attention heatmaps showing what the neural networks focus on
at different levels of visual degradation (glaucoma):

1. RND Predictor GradCAM — What image regions generate curiosity (intrinsic reward).
2. Actor-Critic Input Saliency — What regions the policy uses for decision-making.

Validates the thesis mechanism:
  - Healthy vision → attention distributed on scene features (medkits, walls, corridors)
  - Degraded vision → attention concentrated on residual visible edges
  - Blind (100%) → diffuse / no meaningful attention

Usage:
    python gradcam_analysis.py [--experiment hrrl_sensorial_v1] [--warmup_episodes 10]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
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

def _chebyshev_rank_84():
    """Precompute Chebyshev distance rank matrix for 84×84."""
    h, w = 84, 84
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    return np.maximum(np.abs(y - cy), np.abs(x - cx))

_RANK_84 = _chebyshev_rank_84()


def apply_degradation(frame_84x84, alpha):
    """
    Apply Chebyshev-distance glaucoma degradation to an 84×84 frame.

    Args:
        frame_84x84: (84, 84) numpy array, uint8 or float
        alpha: degradation level, 0.0 (healthy) → 1.0 (blind)

    Returns:
        Degraded frame, same shape and dtype.
    """
    if alpha <= 0:
        return frame_84x84.copy()
    if alpha >= 1.0:
        return np.zeros_like(frame_84x84)

    total = 84 * 84
    erased = int(alpha * total)
    threshold = int(np.sqrt(erased) / 2)

    result = np.zeros_like(frame_84x84)
    keep = _RANK_84 >= threshold
    result[keep] = frame_84x84[keep]
    return result


# ═══════════════════════════════════════════════════════════════════════
# RND Predictor GradCAM
# ═══════════════════════════════════════════════════════════════════════

class RNDGradCAM:
    """
    GradCAM for the RND predictor's last convolutional layer.

    Target scalar: MSE(predictor_features, target_features) = intrinsic reward.
    Highlights which spatial regions of the observation contribute most
    to the prediction error (i.e., what generates curiosity).
    """

    def __init__(self, rnd_module):
        self.rnd = rnd_module
        self._activations = None
        self._gradients = None

        # Hook on last Conv2d in predictor.feature_extractor
        # Architecture: [0]Conv2d [1]ReLU [2]Conv2d [3]ReLU [4]Conv2d [5]ReLU [6]Flatten
        target_layer = self.rnd.predictor.feature_extractor[4]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self._activations = output.detach().clone()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach().clone()

    def generate(self, obs_tensor):
        """
        Compute GradCAM heatmap for a single observation.

        Args:
            obs_tensor: (1, 1, 84, 84) float tensor on correct device.

        Returns:
            cam: (84, 84) numpy array, values in [0, 1].
            intrinsic_reward: scalar float (the normalized MSE).
        """
        self.rnd.predictor.zero_grad()

        # Normalize observation through RND's obs_rms (out-of-place, no stat update)
        was_training = self.rnd.obs_rms.training
        self.rnd.obs_rms.eval()
        with torch.no_grad():
            obs_norm = self.rnd.obs_rms(obs_tensor)
            obs_norm = torch.clamp(obs_norm, -5, 5)
        if was_training:
            self.rnd.obs_rms.train()

        # Detach and enable grad for backprop through predictor only
        obs_input = obs_norm.detach().requires_grad_(True)

        # Forward: predictor (hooks capture activations)
        pred_features = self.rnd.predictor(obs_input)

        # Forward: target (frozen, no grad)
        with torch.no_grad():
            target_features = self.rnd.target(obs_input)

        # MSE = intrinsic reward (before normalization)
        mse = F.mse_loss(pred_features, target_features.detach())
        raw_reward = mse.item()

        # Backward
        mse.backward()

        # GradCAM: weights = Global Average Pooling of gradients
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(cam, size=(84, 84), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, raw_reward


# ═══════════════════════════════════════════════════════════════════════
# Actor-Critic Input Saliency
# ═══════════════════════════════════════════════════════════════════════

def compute_saliency(actor_critic, obs_tensor, cfg, device):
    """
    Compute input gradient saliency for the value function V(s).

    Manually reproduces observation normalization to preserve gradient flow
    (the default ObservationNormalizer wraps normalization in torch.no_grad).

    Args:
        actor_critic: loaded model (eval mode).
        obs_tensor: (1, 1, 84, 84) float tensor.
        cfg: experiment config.
        device: torch device.

    Returns:
        saliency: (84, 84) numpy array in [0, 1], or None on failure.
        value: scalar V(s) estimate, or None.
    """
    try:
        # Manual normalization (replicate ObservationNormalizer without no_grad)
        obs = obs_tensor.clone().to(device)

        # 1. Scale (divide by obs_scale, typically 255)
        obs_scale = getattr(cfg, "obs_scale", 1.0)
        if abs(obs_scale - 1.0) > 1e-5:
            obs = obs / obs_scale

        # 2. Running mean/std normalization
        if getattr(cfg, "normalize_input", False):
            normalizer = actor_critic.obs_normalizer
            if normalizer.running_mean_std is not None:
                rms = normalizer.running_mean_std.running_mean_std["obs"]
                mean = rms.running_mean.float()
                var = rms.running_var.float()
                std = torch.sqrt(var + 1e-5)
                obs = (obs - mean) / std
                obs = obs.clamp(-5, 5)

        # Enable gradient on normalized input
        obs_input = obs.detach().requires_grad_(True)

        # cuDNN RNN backward requires training mode
        was_training = actor_critic.training
        actor_critic.train()

        # Forward pass: encoder → core (GRU with zero state) → decoder → V(s)
        rnn_size = get_rnn_size(cfg)
        rnn_states = torch.zeros(1, rnn_size, dtype=torch.float32, device=device)

        enc_out = actor_critic.encoder({"obs": obs_input})
        core_out, _ = actor_critic.core(enc_out, rnn_states)
        dec_out = actor_critic.decoder(core_out)
        value = actor_critic.critic_linear(dec_out)

        # Backward
        value.backward()

        # Restore original mode
        if not was_training:
            actor_critic.eval()

        # Saliency = |gradient| w.r.t. input
        saliency = obs_input.grad.abs().squeeze().cpu().numpy()
        if saliency.max() > 0:
            saliency = saliency / saliency.max()

        return saliency, value.item()

    except Exception as e:
        print(f"  Warning: Actor-Critic saliency failed: {e}")
        return None, None


# ═══════════════════════════════════════════════════════════════════════
# Episode runner — collect frames with metadata
# ═══════════════════════════════════════════════════════════════════════

def warmup_rnd(actor_critic, rnd_module, env, env_info, cfg, device, num_episodes):
    """Train RND predictor online for warmup episodes."""
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
    """
    Run episodes and collect frames with metadata.

    Returns list of dicts with keys:
        frame: (84, 84) numpy uint8
        z_homeostasis: float 0-1
        health: float
        intrinsic_reward: float
        rnn_states: (1, rnn_size) tensor (for actor-critic saliency)
    """
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

                # Save current observation and RNN state
                frame_np = obs["obs"][0, 0].cpu().numpy()
                obs_buffer.append(obs["obs"][0].float().to(device))

                # Save metadata (intrinsic_reward filled later)
                step_data.append({
                    "frame": frame_np,
                    "rnn_states": rnn_states.clone(),
                    "z_homeostasis": 1.0,
                    "health": 0.0,
                    "intrinsic_reward": 0.0,
                })

                # Step environment
                obs, rew, terminated, truncated, infos = env.step(actions)
                _ = obs.pop("action_mask", None)
                dones = make_dones(terminated, truncated)
                done = dones[0].item()
                dones_buffer.append(done)
                rnn_states = new_rnn_states

                # Extract info
                info = infos[0] if infos else {}
                extra = info.get("episode_extra_stats", {})
                step_data[-1]["z_homeostasis"] = extra.get("z_homeostasis", 1.0)
                step_data[-1]["health"] = info.get("HEALTH", 0.0)

                # Process RND batch
                if len(obs_buffer) >= _RND_BATCH or done:
                    batch = torch.stack(obs_buffer)
                    batch_dones = torch.tensor(dones_buffer, dtype=torch.bool, device=device)
                    rnd_rewards = rnd_module.calculate_rewards({"obs": batch}, dones=batch_dones)

                    if train_rnd:
                        with torch.enable_grad():
                            rnd_module.update({"obs": batch}, batch_dones)

                    # Distribute rewards
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
# Visualization helpers
# ═══════════════════════════════════════════════════════════════════════

def overlay_heatmap(frame, heatmap, alpha=0.5, cmap="jet"):
    """
    Overlay a heatmap on a grayscale frame.

    Args:
        frame: (84, 84) uint8 or float array.
        heatmap: (84, 84) float array in [0, 1].
        alpha: blending factor for heatmap.
        cmap: matplotlib colormap name.

    Returns:
        (84, 84, 3) float array in [0, 1].
    """
    frame_f = frame.astype(np.float32)
    if frame_f.max() > 1:
        frame_f = frame_f / 255.0

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


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Controlled degradation comparison
# ═══════════════════════════════════════════════════════════════════════

def plot_controlled_gradcam(clean_frame, rnd_cam, actor_critic, cfg,
                            rnd_module, device, output_dir):
    """
    Same scene, 4 degradation levels.
    Row 0: Observation
    Row 1: RND GradCAM overlay
    Row 2: Actor-Critic saliency overlay
    """
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

        # --- Row 0: observation ---
        axes[0, col].imshow(degraded, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(label, fontweight="bold")
        axes[0, col].axis("off")

        # --- Row 1: RND GradCAM ---
        cam, r_int = rnd_cam.generate(obs_t)
        blended = overlay_heatmap(degraded, cam)
        axes[1, col].imshow(blended)
        axes[1, col].set_title(f"r_int = {r_int:.4f}", fontsize=10)
        axes[1, col].axis("off")

        # --- Row 2: Actor-Critic saliency ---
        saliency, v_s = compute_saliency(actor_critic, obs_t, cfg, device)
        if saliency is not None:
            blended_sal = overlay_heatmap(degraded, saliency)
            axes[2, col].imshow(blended_sal)
            axes[2, col].set_title(f"V(s) = {v_s:.3f}", fontsize=10)
        else:
            axes[2, col].imshow(degraded, cmap="gray")
            axes[2, col].set_title("(falhou)", fontsize=10)
        axes[2, col].axis("off")

    # Row labels
    row_labels = ["Observação", "RND Predictor\nGradCAM", "Actor-Critic\nSaliência"]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(
            label, fontsize=12, fontweight="bold",
            rotation=0, labelpad=90, va="center",
        )

    fig.suptitle(
        "GradCAM: Atenção Visual vs Nível de Degradação (Glaucoma)",
        fontsize=15, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0.1, 0, 1, 0.96])

    path = os.path.join(output_dir, "fig_gradcam_controlled.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Natural gameplay frames at different degradation levels
# ═══════════════════════════════════════════════════════════════════════

def plot_natural_gradcam(frames_data, rnd_cam, actor_critic, cfg, device, output_dir):
    """
    Pick representative frames from actual gameplay at different
    homeostasis levels and show GradCAM / saliency.
    """
    buckets = {
        "Saudável\n(z > 0.95)":   (0.95, 1.01),
        "Leve\n(0.6 < z < 0.8)":  (0.60, 0.80),
        "Severo\n(0.3 < z < 0.5)":(0.30, 0.50),
        "Crítico\n(z < 0.15)":    (0.00, 0.15),
    }

    selected = []
    for label, (lo, hi) in buckets.items():
        candidates = [d for d in frames_data if lo <= d["z_homeostasis"] < hi]
        if candidates:
            # Pick the candidate with median intrinsic reward
            candidates.sort(key=lambda d: d["intrinsic_reward"])
            selected.append((label, candidates[len(candidates) // 2]))
        else:
            short_label = label.split("\n")[0]
            print(f"  Warning: no frames for bucket '{short_label}' — skipping")

    if len(selected) < 2:
        print("  Not enough degradation variety in gameplay. Skipping natural GradCAM.")
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

        # Row 0: observation
        axes[0, col].imshow(frame, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(f"{label}\nz={z:.2f}  r_int={r_int:.3f}",
                               fontweight="bold", fontsize=10)
        axes[0, col].axis("off")

        # Row 1: RND GradCAM
        cam, _ = rnd_cam.generate(obs_t)
        blended = overlay_heatmap(frame, cam)
        axes[1, col].imshow(blended)
        axes[1, col].axis("off")

        # Row 2: Actor-Critic saliency
        saliency, v_s = compute_saliency(actor_critic, obs_t, cfg, device)
        if saliency is not None:
            blended_sal = overlay_heatmap(frame, saliency)
            axes[2, col].imshow(blended_sal)
            axes[2, col].set_title(f"V(s)={v_s:.3f}" if v_s else "", fontsize=9)
        else:
            axes[2, col].imshow(frame, cmap="gray")
        axes[2, col].axis("off")

    row_labels = ["Frame do\nGameplay", "RND Predictor\nGradCAM", "Actor-Critic\nSaliência"]
    for row, lbl in enumerate(row_labels):
        axes[row, 0].set_ylabel(
            lbl, fontsize=12, fontweight="bold",
            rotation=0, labelpad=90, va="center",
        )

    fig.suptitle(
        "GradCAM em Frames Naturais do Gameplay",
        fontsize=15, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0.1, 0, 1, 0.96])

    path = os.path.join(output_dir, "fig_gradcam_natural.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: RND GradCAM heatmap strip (raw, without overlay)
# ═══════════════════════════════════════════════════════════════════════

def plot_raw_heatmaps(clean_frame, rnd_cam, device, output_dir):
    """Raw GradCAM heatmaps side by side with a shared colorbar."""
    levels = [
        (0.0, "α = 0%"),
        (0.2, "α = 20%"),
        (0.4, "α = 40%"),
        (0.6, "α = 60%"),
        (0.8, "α = 80%"),
        (1.0, "α = 100%"),
    ]
    n = len(levels)

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

    for col, (alpha, label) in enumerate(levels):
        degraded = apply_degradation(clean_frame, alpha)
        obs_t = torch.from_numpy(degraded).float().unsqueeze(0).unsqueeze(0).to(device)
        cam, r_int = rnd_cam.generate(obs_t)

        axes[0, col].imshow(degraded, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(label, fontweight="bold", fontsize=11)
        axes[0, col].axis("off")

        im = axes[1, col].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[1, col].set_title(f"r = {r_int:.4f}", fontsize=9)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Observação", fontsize=11, fontweight="bold",
                           rotation=0, labelpad=60, va="center")
    axes[1, 0].set_ylabel("GradCAM\n(RND)", fontsize=11, fontweight="bold",
                           rotation=0, labelpad=60, va="center")

    fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.8, label="Ativação GradCAM")
    fig.suptitle("RND GradCAM: Progressão da Degradação Visual",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, "fig_gradcam_heatmap_strip.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GradCAM / Saliency analysis for HRRL-Sensorial agent"
    )
    parser.add_argument("--experiment", default="hrrl_sensorial_v1")
    parser.add_argument("--train_dir", default="train_dir")
    parser.add_argument("--output_dir", default="figures")
    parser.add_argument("--warmup_episodes", type=int, default=10,
                        help="Episodes for RND predictor warmup")
    parser.add_argument("--eval_episodes", type=int, default=3,
                        help="Episodes to collect frames from")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print("HRRL-Sensorial — GradCAM / Saliency Analysis")
    print("=" * 60)

    # ── 1. Load model ──
    print("\n[1/5] Loading model...")
    cfg = build_cfg(args.experiment, args.train_dir)
    env = make_env_func_batched(
        cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode=None,
    )
    env_info = extract_env_info(env, cfg)
    actor_critic = load_actor_critic(cfg, env, device)
    rnd_module = create_rnd_module(cfg, env, device)

    # ── 2. Warmup RND predictor ──
    print(f"\n[2/5] RND warmup ({args.warmup_episodes} episodes)...")
    warmup_rnd(actor_critic, rnd_module, env, env_info, cfg, device, args.warmup_episodes)

    # ── 3. Collect evaluation frames ──
    print(f"\n[3/5] Collecting frames ({args.eval_episodes} episodes)...")
    all_frames = collect_frames(
        actor_critic, rnd_module, env, env_info, cfg, device,
        num_episodes=args.eval_episodes, train_rnd=True,
    )
    env.close()
    print(f"  Total frames: {len(all_frames)}")

    # Distribution of homeostasis levels
    z_vals = [d["z_homeostasis"] for d in all_frames]
    print(f"  z_homeostasis: min={min(z_vals):.3f}, mean={np.mean(z_vals):.3f}, max={max(z_vals):.3f}")

    # ── 4. Setup GradCAM ──
    print("\n[4/5] Setting up GradCAM hooks...")
    rnd_cam = RNDGradCAM(rnd_module)

    # Find a clean reference frame (z ≈ 1.0, after medkit)
    clean_candidates = [d for d in all_frames if d["z_homeostasis"] > 0.98]
    if clean_candidates:
        clean_frame = clean_candidates[len(clean_candidates) // 2]["frame"]
    else:
        clean_frame = all_frames[0]["frame"]
    print(f"  Clean reference frame selected (z={1.0 if clean_candidates else 'fallback'})")

    # ── 5. Generate figures ──
    print("\n[5/5] Generating figures...")

    # Figure 1: Controlled degradation
    print("  [Fig 1] Controlled degradation comparison...")
    plot_controlled_gradcam(clean_frame, rnd_cam, actor_critic, cfg,
                            rnd_module, device, args.output_dir)

    # Figure 2: Natural gameplay frames
    print("  [Fig 2] Natural gameplay frames...")
    plot_natural_gradcam(all_frames, rnd_cam, actor_critic, cfg,
                         device, args.output_dir)

    # Figure 3: Raw heatmap strip
    print("  [Fig 3] Raw heatmap progression...")
    plot_raw_heatmaps(clean_frame, rnd_cam, device, args.output_dir)

    print("\nDone! Figures saved to:", args.output_dir)


if __name__ == "__main__":
    main()
