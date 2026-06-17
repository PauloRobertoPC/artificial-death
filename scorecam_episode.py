"""
ScoreCAM frame-by-frame for a full episode.

Generates one image per step showing [Observation | AC ScoreCAM | RND ScoreCAM]
with metadata overlay. Medkit events get a green border. Critical frames (low z)
are generated at every step; the rest every N steps.

Output: figures/scorecam_episode/step_XXXX_z0.XX.png

Usage:
    python scorecam_episode.py [--experiment hrrl_sensorial_v1] [--every_n 5]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
# ScoreCAM (compact — same logic as scorecam_analysis.py)
# ═══════════════════════════════════════════════════════════════════════

class ScoreCAM:
    def __init__(self, get_activations_fn, score_fn):
        self.get_activations_fn = get_activations_fn
        self.score_fn = score_fn

    @torch.no_grad()
    def generate(self, obs_tensor):
        activations = self.get_activations_fn(obs_tensor)
        C = activations.shape[0]
        acts_up = F.interpolate(
            activations.unsqueeze(0), size=(84, 84),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        for k in range(C):
            a_min, a_max = acts_up[k].min(), acts_up[k].max()
            if (a_max - a_min) > 1e-8:
                acts_up[k] = (acts_up[k] - a_min) / (a_max - a_min)
            else:
                acts_up[k] = 0.0
        obs_expanded = obs_tensor.expand(C, -1, -1, -1)
        masked_inputs = obs_expanded * acts_up.unsqueeze(1)
        scores = self.score_fn(masked_inputs)
        baseline = self.score_fn(obs_tensor)
        weights = F.softmax(scores, dim=0)
        cam = (weights.view(C, 1, 1) * acts_up).sum(dim=0)
        cam = F.relu(cam)
        cam_np = cam.cpu().numpy()
        if cam_np.max() > 0:
            cam_np /= cam_np.max()
        return cam_np, baseline.item()


def build_rnd_scorecam(rnd_module, device):
    def get_activations(obs_tensor):
        was_t = rnd_module.obs_rms.training
        rnd_module.obs_rms.eval()
        obs_norm = torch.clamp(rnd_module.obs_rms(obs_tensor), -5, 5)
        if was_t: rnd_module.obs_rms.train()
        x = obs_norm
        for i in range(6):
            x = rnd_module.predictor.feature_extractor[i](x)
        return x.squeeze(0)

    def score_fn(masked_batch):
        was_t = rnd_module.obs_rms.training
        rnd_module.obs_rms.eval()
        obs_norm = torch.clamp(rnd_module.obs_rms(masked_batch), -5, 5)
        if was_t: rnd_module.obs_rms.train()
        pred = rnd_module.predictor(obs_norm)
        target = rnd_module.target(obs_norm)
        return (pred - target).pow(2).mean(dim=-1)

    return ScoreCAM(get_activations, score_fn)


def build_ac_scorecam(actor_critic, cfg, device):
    # Non-JIT conv head copy
    jit_enc = actor_critic.encoder.basic_encoder.enc
    sd = jit_enc.conv_head.state_dict()
    w0, w2, w4 = sd["0.weight"], sd["2.weight"], sd["4.weight"]
    conv_head = nn.Sequential(
        nn.Conv2d(w0.shape[1], w0.shape[0], w0.shape[2], stride=4), nn.ReLU(),
        nn.Conv2d(w2.shape[1], w2.shape[0], w2.shape[2], stride=2), nn.ReLU(),
        nn.Conv2d(w4.shape[1], w4.shape[0], w4.shape[2], stride=2), nn.ReLU(),
    )
    conv_head.load_state_dict(sd)
    conv_head.to(device).eval()
    rnn_size = get_rnn_size(cfg)

    def _norm(obs_batch):
        obs = obs_batch.clone()
        s = getattr(cfg, "obs_scale", 1.0)
        if abs(s - 1.0) > 1e-5:
            obs = obs / s
        if getattr(cfg, "normalize_input", False):
            n = actor_critic.obs_normalizer
            if n.running_mean_std is not None:
                rms = n.running_mean_std.running_mean_std["obs"]
                obs = (obs - rms.running_mean.float()) / torch.sqrt(rms.running_var.float() + 1e-5)
                obs = obs.clamp(-5, 5)
        return obs

    def get_activations(obs_tensor):
        return conv_head(_norm(obs_tensor)).squeeze(0)

    def score_fn(masked_batch):
        N = masked_batch.shape[0]
        enc_out = actor_critic.encoder({"obs": _norm(masked_batch)})
        rnn_st = torch.zeros(N, rnn_size, dtype=torch.float32, device=device)
        core_out, _ = actor_critic.core(enc_out, rnn_st)
        dec_out = actor_critic.decoder(core_out)
        return actor_critic.critic_linear(dec_out).squeeze(-1)

    return ScoreCAM(get_activations, score_fn)


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
                if actions.ndim == 1: actions = unsqueeze_tensor(actions, dim=-1)
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
                    obs_buf.clear(); done_buf.clear()
                if done: break
        print(f"  Warmup {ep+1}/{num_episodes}: {n} steps")


# ═══════════════════════════════════════════════════════════════════════
# Run episode and collect ALL frames
# ═══════════════════════════════════════════════════════════════════════

def run_full_episode(actor_critic, env, env_info, cfg, device, rnd_module=None):
    """Run one episode, return per-step data with frames and (optional) RND rewards."""
    obs, _ = env.reset()
    _ = obs.pop("action_mask", None)
    rnn_states = torch.zeros(1, get_rnn_size(cfg), dtype=torch.float32, device=device)
    obs_buf, done_buf = [], []
    steps = []
    use_rnd = rnd_module is not None

    with torch.no_grad():
        while True:
            norm_obs = prepare_and_normalize_obs(actor_critic, obs)
            out = actor_critic(norm_obs, rnn_states)
            actions = out["actions"]
            rnn_states = out["new_rnn_states"]
            if actions.ndim == 1: actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)

            frame = obs["obs"][0, 0].cpu().numpy()
            if use_rnd:
                obs_buf.append(obs["obs"][0].float().to(device))
            steps.append({
                "frame": frame,
                "z": 1.0, "health": 0.0, "r_int": 0.0, "medkit": False,
            })

            obs, _, term, trunc, infos = env.step(actions)
            _ = obs.pop("action_mask", None)
            done = make_dones(term, trunc)[0].item()
            if use_rnd:
                done_buf.append(done)

            info = infos[0] if infos else {}
            extra = info.get("episode_extra_stats", {})
            steps[-1]["z"] = extra.get("z_homeostasis", 1.0)
            steps[-1]["health"] = info.get("HEALTH", 0.0)

            if use_rnd and (len(obs_buf) >= _RND_BATCH or done):
                b = torch.stack(obs_buf)
                bd = torch.tensor(done_buf, dtype=torch.bool, device=device)
                rr = rnd_module.calculate_rewards({"obs": b}, dones=bd)
                with torch.enable_grad():
                    rnd_module.update({"obs": b}, bd)
                si = len(steps) - len(obs_buf)
                for i in range(len(obs_buf)):
                    r = rr[i].item()
                    steps[si + i]["r_int"] = 0.0 if np.isnan(r) else r
                obs_buf.clear(); done_buf.clear()

            if done: break

    # Detect medkit events
    for i in range(1, len(steps)):
        if steps[i]["health"] > steps[i - 1]["health"]:
            steps[i]["medkit"] = True

    return steps


# ═══════════════════════════════════════════════════════════════════════
# Image generation
# ═══════════════════════════════════════════════════════════════════════

def overlay(frame, heatmap, alpha=0.5):
    f = frame.astype(np.float32)
    if f.max() > 1: f /= 255.0
    rgb = np.stack([f] * 3, axis=-1)
    cmap = plt.colormaps.get_cmap("jet")
    hm = cmap(heatmap)[:, :, :3]
    return np.clip((1 - alpha) * rgb + alpha * hm, 0, 1)


def save_frame_image(step_idx, data, cam_ac, v_s, out_dir, cam_rnd=None, r_int_sc=None):
    """Save a single frame as [Obs | AC ScoreCAM] or [Obs | AC ScoreCAM | RND ScoreCAM]."""
    frame = data["frame"]
    z = data["z"]
    health = data["health"]
    r_int = data["r_int"]
    is_medkit = data["medkit"]

    n_panels = 3 if cam_rnd is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    # Panel 1: raw observation
    axes[0].imshow(frame, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Observação", fontsize=10)

    # Panel 2: Actor-Critic ScoreCAM
    axes[1].imshow(overlay(frame, cam_ac))
    axes[1].set_title(f"AC ScoreCAM  V(s)={v_s:.3f}", fontsize=10)

    # Panel 3: RND ScoreCAM (optional)
    if cam_rnd is not None:
        axes[2].imshow(overlay(frame, cam_rnd))
        axes[2].set_title(f"RND ScoreCAM  r_int={r_int_sc:.4f}", fontsize=10)

    for ax in axes:
        ax.axis("off")

    # Medkit border (green) or critical border (red)
    if is_medkit:
        for ax in axes:
            for spine in ax.spines.values():
                spine.set_edgecolor("lime")
                spine.set_linewidth(4)
                spine.set_visible(True)
            ax.set_frame_on(True)

    # Suptitle with metadata
    status = "MEDKIT!" if is_medkit else ("CRITICO" if z < 0.3 else "")
    color = "green" if is_medkit else ("red" if z < 0.3 else "black")
    r_int_str = f"r_int={r_int:.4f}  " if cam_rnd is not None else ""
    fig.suptitle(
        f"Step {step_idx:04d}  |  z={z:.3f}  |  health={health:.0f}  |  "
        f"{r_int_str}{status}",
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
    parser = argparse.ArgumentParser(description="ScoreCAM per-frame for full episode")
    parser.add_argument("--experiment", default="hrrl_sensorial_v1")
    parser.add_argument("--train_dir", default="train_dir")
    parser.add_argument("--output_dir", default="figures/scorecam_episode")
    parser.add_argument("--warmup_episodes", type=int, default=10)
    parser.add_argument("--every_n", type=int, default=5,
                        help="Generate ScoreCAM every N steps (critical/medkit frames always included)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print("HRRL-Sensorial — ScoreCAM Episode (per-frame)")
    print("=" * 60)

    # 1. Load
    print("\n[1/4] Loading model...")
    cfg = build_cfg(args.experiment, args.train_dir)
    use_rnd = getattr(cfg, "with_curiosity", False)
    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode=None,
    )
    env_info = extract_env_info(env, cfg)
    actor_critic = load_actor_critic(cfg, env, device)

    rnd_module = None
    if use_rnd:
        rnd_module = create_rnd_module(cfg, env, device)

    # 2. Warmup
    if use_rnd:
        print(f"\n[2/4] RND warmup ({args.warmup_episodes} episodes)...")
        warmup_rnd(actor_critic, rnd_module, env, env_info, cfg, device, args.warmup_episodes)
    else:
        print("\n[2/4] RND warmup skipped (with_curiosity=False)")

    # 3. Run episode
    print("\n[3/4] Running episode...")
    steps = run_full_episode(actor_critic, env, env_info, cfg, device, rnd_module=rnd_module)
    env.close()

    n_medkits = sum(1 for s in steps if s["medkit"])
    n_critical = sum(1 for s in steps if s["z"] < 0.3)
    print(f"  {len(steps)} steps, {n_medkits} medkits, {n_critical} critical frames (z < 0.3)")

    # 4. Generate ScoreCAM images
    print("\n[4/4] Generating ScoreCAM images...")
    rnd_cam = build_rnd_scorecam(rnd_module, device) if use_rnd else None
    ac_cam = build_ac_scorecam(actor_critic, cfg, device)

    # Select which frames to process
    indices = set()
    for i, s in enumerate(steps):
        # Always include: medkit events ± 3 steps
        if s["medkit"]:
            for j in range(max(0, i - 3), min(len(steps), i + 4)):
                indices.add(j)
        # Always include: critical (z < 0.5)
        if s["z"] < 0.5:
            indices.add(i)
        # Regular interval
        if i % args.every_n == 0:
            indices.add(i)
    # Always first and last
    indices.add(0)
    indices.add(len(steps) - 1)

    indices = sorted(indices)
    print(f"  Processing {len(indices)} / {len(steps)} frames "
          f"(every {args.every_n} + all critical + medkit±3)...")

    for count, idx in enumerate(indices):
        data = steps[idx]
        obs_t = torch.from_numpy(data["frame"]).float().unsqueeze(0).unsqueeze(0).to(device)

        cam_ac, v_s = ac_cam.generate(obs_t)
        cam_rnd, r_int_sc = None, None
        if rnd_cam is not None:
            cam_rnd, r_int_sc = rnd_cam.generate(obs_t)

        save_frame_image(idx, data, cam_ac, v_s, args.output_dir, cam_rnd=cam_rnd, r_int_sc=r_int_sc)

        if (count + 1) % 50 == 0 or count == len(indices) - 1:
            print(f"    {count + 1}/{len(indices)} frames saved")

    print(f"\nDone! {len(indices)} images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
