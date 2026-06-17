"""
GradCAM / Saliency frame-by-frame for a full episode.

Generates one image per step showing [Observation | AC Saliency | RND GradCAM]
with metadata overlay. Medkit events get a green title. Critical frames (low z)
are generated at every step; the rest every N steps.

Output: figures/gradcam_episode/step_XXXX_z0.XX.png

Usage:
    python gradcam_episode.py [--experiment hrrl_sensorial_v1] [--every_n 5]
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
# Model loading
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
# RND GradCAM (gradient-based)
# ═══════════════════════════════════════════════════════════════════════

class RNDGradCAM:
    """GradCAM on RND predictor's last Conv2d. Target: MSE(pred, target)."""

    def __init__(self, rnd_module):
        self.rnd = rnd_module
        self._activations = None
        self._gradients = None
        target_layer = self.rnd.predictor.feature_extractor[4]
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, output):
        self._activations = output.detach().clone()

    def _save_grad(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach().clone()

    def generate(self, obs_tensor):
        self.rnd.predictor.zero_grad()
        was_t = self.rnd.obs_rms.training
        self.rnd.obs_rms.eval()
        with torch.no_grad():
            obs_norm = torch.clamp(self.rnd.obs_rms(obs_tensor), -5, 5)
        if was_t:
            self.rnd.obs_rms.train()

        obs_input = obs_norm.detach().requires_grad_(True)
        pred = self.rnd.predictor(obs_input)
        with torch.no_grad():
            target = self.rnd.target(obs_input)
        mse = F.mse_loss(pred, target.detach())
        r_int = mse.item()
        mse.backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(84, 84), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, r_int


# ═══════════════════════════════════════════════════════════════════════
# Actor-Critic Input Saliency (gradient of V(s) w.r.t. input)
# ═══════════════════════════════════════════════════════════════════════

def compute_saliency(actor_critic, obs_tensor, cfg, device):
    """Input gradient saliency for V(s). Returns (heatmap, value) or (None, None)."""
    try:
        obs = obs_tensor.clone().to(device)
        obs_scale = getattr(cfg, "obs_scale", 1.0)
        if abs(obs_scale - 1.0) > 1e-5:
            obs = obs / obs_scale
        if getattr(cfg, "normalize_input", False):
            n = actor_critic.obs_normalizer
            if n.running_mean_std is not None:
                rms = n.running_mean_std.running_mean_std["obs"]
                mean = rms.running_mean.float()
                std = torch.sqrt(rms.running_var.float() + 1e-5)
                obs = ((obs - mean) / std).clamp(-5, 5)

        obs_input = obs.detach().requires_grad_(True)

        was_training = actor_critic.training
        actor_critic.train()

        rnn_size = get_rnn_size(cfg)
        rnn_states = torch.zeros(1, rnn_size, dtype=torch.float32, device=device)
        enc_out = actor_critic.encoder({"obs": obs_input})
        core_out, _ = actor_critic.core(enc_out, rnn_states)
        dec_out = actor_critic.decoder(core_out)
        value = actor_critic.critic_linear(dec_out)
        value.backward()

        if not was_training:
            actor_critic.eval()

        sal = obs_input.grad.abs().squeeze().cpu().numpy()
        if sal.max() > 0:
            sal /= sal.max()
        return sal, value.item()

    except Exception as e:
        print(f"  Warning: AC saliency failed: {e}")
        return None, None


# ═══════════════════════════════════════════════════════════════════════
# RND warmup
# ═══════════════════════════════════════════════════════════════════════

def warmup_rnd(actor_critic, rnd_module, env, env_info, cfg, device, num_episodes):
    for ep in range(num_episodes):
        obs, _ = env.reset()
        _ = obs.pop("action_mask", None)
        rnn_states = torch.zeros(1, get_rnn_size(cfg), dtype=torch.float32, device=device)
        obs_buf, done_buf = [], []
        n = 0
        with torch.no_grad():
            while True:
                norm_obs = prepare_and_normalize_obs(actor_critic, obs)
                out = actor_critic(norm_obs, rnn_states)
                actions = out["actions"]
                rnn_states = out["new_rnn_states"]
                if actions.ndim == 1:
                    actions = unsqueeze_tensor(actions, dim=-1)
                actions = preprocess_actions(env_info, actions)
                obs_buf.append(obs["obs"][0].float().to(device))
                obs, _, term, trunc, _ = env.step(actions)
                _ = obs.pop("action_mask", None)
                done = make_dones(term, trunc)[0].item()
                done_buf.append(done)
                n += 1
                if len(obs_buf) >= _RND_BATCH or done:
                    b = torch.stack(obs_buf)
                    bd = torch.tensor(done_buf, dtype=torch.bool, device=device)
                    rnd_module.calculate_rewards({"obs": b}, dones=bd)
                    with torch.enable_grad():
                        rnd_module.update({"obs": b}, bd)
                    obs_buf.clear()
                    done_buf.clear()
                if done:
                    break
        print(f"  Warmup {ep+1}/{num_episodes}: {n} steps")


# ═══════════════════════════════════════════════════════════════════════
# Run full episode
# ═══════════════════════════════════════════════════════════════════════

def run_full_episode(actor_critic, rnd_module, env, env_info, cfg, device):
    obs, _ = env.reset()
    _ = obs.pop("action_mask", None)
    rnn_states = torch.zeros(1, get_rnn_size(cfg), dtype=torch.float32, device=device)
    obs_buf, done_buf = [], []
    steps = []

    with torch.no_grad():
        while True:
            norm_obs = prepare_and_normalize_obs(actor_critic, obs)
            out = actor_critic(norm_obs, rnn_states)
            actions = out["actions"]
            rnn_states = out["new_rnn_states"]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)

            frame = obs["obs"][0, 0].cpu().numpy()
            obs_buf.append(obs["obs"][0].float().to(device))
            steps.append({
                "frame": frame,
                "z": 1.0, "health": 0.0, "r_int": 0.0, "medkit": False,
            })

            obs, _, term, trunc, infos = env.step(actions)
            _ = obs.pop("action_mask", None)
            done = make_dones(term, trunc)[0].item()
            done_buf.append(done)

            info = infos[0] if infos else {}
            extra = info.get("episode_extra_stats", {})
            steps[-1]["z"] = extra.get("z_homeostasis", 1.0)
            steps[-1]["health"] = info.get("HEALTH", 0.0)

            if len(obs_buf) >= _RND_BATCH or done:
                b = torch.stack(obs_buf)
                bd = torch.tensor(done_buf, dtype=torch.bool, device=device)
                rr = rnd_module.calculate_rewards({"obs": b}, dones=bd)
                with torch.enable_grad():
                    rnd_module.update({"obs": b}, bd)
                si = len(steps) - len(obs_buf)
                for i in range(len(obs_buf)):
                    r = rr[i].item()
                    steps[si + i]["r_int"] = 0.0 if np.isnan(r) else r
                obs_buf.clear()
                done_buf.clear()

            if done:
                break

    for i in range(1, len(steps)):
        if steps[i]["health"] > steps[i - 1]["health"]:
            steps[i]["medkit"] = True

    return steps


# ═══════════════════════════════════════════════════════════════════════
# Image generation
# ═══════════════════════════════════════════════════════════════════════

def overlay(frame, heatmap, alpha=0.5):
    f = frame.astype(np.float32)
    if f.max() > 1:
        f /= 255.0
    rgb = np.stack([f] * 3, axis=-1)
    cmap = plt.colormaps.get_cmap("jet")
    hm = cmap(heatmap)[:, :, :3]
    return np.clip((1 - alpha) * rgb + alpha * hm, 0, 1)


def save_frame_image(step_idx, data, sal_ac, v_s, cam_rnd, r_int_gc, out_dir):
    frame = data["frame"]
    z = data["z"]
    health = data["health"]
    r_int = data["r_int"]
    is_medkit = data["medkit"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(frame, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Observação", fontsize=10)

    if sal_ac is not None:
        axes[1].imshow(overlay(frame, sal_ac))
        axes[1].set_title(f"AC Saliência  V(s)={v_s:.3f}", fontsize=10)
    else:
        axes[1].imshow(frame, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("AC Saliência (falhou)", fontsize=10)

    axes[2].imshow(overlay(frame, cam_rnd))
    axes[2].set_title(f"RND GradCAM  r_int={r_int_gc:.4f}", fontsize=10)

    for ax in axes:
        ax.axis("off")

    status = "MEDKIT!" if is_medkit else ("CRITICO" if z < 0.3 else "")
    color = "green" if is_medkit else ("red" if z < 0.3 else "black")
    fig.suptitle(
        f"Step {step_idx:04d}  |  z={z:.3f}  |  health={health:.0f}  |  "
        f"r_int={r_int:.4f}  {status}",
        fontsize=12, fontweight="bold", color=color, y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    tag = "medkit" if is_medkit else ("crit" if z < 0.3 else "ok")
    fname = f"step_{step_idx:04d}_z{z:.2f}_{tag}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GradCAM per-frame for full episode")
    parser.add_argument("--experiment", default="hrrl_sensorial_v1")
    parser.add_argument("--train_dir", default="train_dir")
    parser.add_argument("--output_dir", default="figures/gradcam_episode")
    parser.add_argument("--warmup_episodes", type=int, default=10)
    parser.add_argument("--every_n", type=int, default=5,
                        help="Generate every N steps (critical/medkit always included)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print("HRRL-Sensorial — GradCAM Episode (per-frame)")
    print("=" * 60)

    # 1. Load
    print("\n[1/4] Loading model...")
    cfg = build_cfg(args.experiment, args.train_dir)
    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode=None,
    )
    env_info = extract_env_info(env, cfg)
    actor_critic = load_actor_critic(cfg, env, device)
    rnd_module = create_rnd_module(cfg, env, device)

    # 2. Warmup
    print(f"\n[2/4] RND warmup ({args.warmup_episodes} episodes)...")
    warmup_rnd(actor_critic, rnd_module, env, env_info, cfg, device, args.warmup_episodes)

    # 3. Run episode
    print("\n[3/4] Running episode...")
    steps = run_full_episode(actor_critic, rnd_module, env, env_info, cfg, device)
    env.close()

    n_medkits = sum(1 for s in steps if s["medkit"])
    n_critical = sum(1 for s in steps if s["z"] < 0.3)
    print(f"  {len(steps)} steps, {n_medkits} medkits, {n_critical} critical frames (z < 0.3)")

    # 4. Generate GradCAM images
    print("\n[4/4] Generating GradCAM images...")
    rnd_cam = RNDGradCAM(rnd_module)

    # Select frames
    indices = set()
    for i, s in enumerate(steps):
        if s["medkit"]:
            for j in range(max(0, i - 3), min(len(steps), i + 4)):
                indices.add(j)
        if s["z"] < 0.5:
            indices.add(i)
        if i % args.every_n == 0:
            indices.add(i)
    indices.add(0)
    indices.add(len(steps) - 1)
    indices = sorted(indices)

    print(f"  Processing {len(indices)} / {len(steps)} frames "
          f"(every {args.every_n} + all critical + medkit±3)...")

    for count, idx in enumerate(indices):
        data = steps[idx]
        obs_t = torch.from_numpy(data["frame"]).float().unsqueeze(0).unsqueeze(0).to(device)

        cam_rnd, r_int_gc = rnd_cam.generate(obs_t)
        sal_ac, v_s = compute_saliency(actor_critic, obs_t, cfg, device)

        save_frame_image(idx, data, sal_ac, v_s, cam_rnd, r_int_gc, args.output_dir)

        if (count + 1) % 50 == 0 or count == len(indices) - 1:
            print(f"    {count + 1}/{len(indices)} frames saved")

    print(f"\nDone! {len(indices)} images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
