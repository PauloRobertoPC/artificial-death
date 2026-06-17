from typing import Dict, Any, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.algo.utils.curiosity_interface import CuriosityModule
from sample_factory.algo.utils.running_mean_std import RunningMeanStd, RunningMeanStdInPlace

class RNDEncoder(nn.Module):
    """
    Encoder de CNN baseado na arquitetura 'Nature CNN' (Mnih et al. 2015).
    Usado amplamente em RND para Atari/VizDoom.
    """
    def __init__(self, obs_space: ObsSpace, latent_dim: int = 512):
        super().__init__()
        # Pega dimensões da observação (C, H, W)
        # Quando DAE online está ativo, obs_space tem keys obs_noisy/obs_clean; usar obs_noisy.
        obs_key = "obs_noisy" if "obs_noisy" in obs_space.spaces else "obs"
        shape = obs_space[obs_key].shape
        c = shape[0]
        
        # Arquitetura de 3 camadas (Nature CNN)
        # Kernel 8x8 (stride 4) -> Kernel 4x4 (stride 2) -> Kernel 3x3 (stride 1)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Cálculo dinâmico do tamanho da saída da CNN
        # Isso garante que funcione tanto para 84x84 quanto para 240x320 sem quebrar
        with torch.no_grad():
            dummy_obs = torch.zeros(1, *shape)
            output_dim = self.feature_extractor(dummy_obs).shape[1]

        # Camadas finais
        self.head = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim) # Normalização extra para estabilidade
        )

    def forward(self, obs_tensor: Tensor) -> Tensor:
        x = self.feature_extractor(obs_tensor)
        return self.head(x)


class RNDModule(CuriosityModule, nn.Module):
    """
    Módulo RND completo.
    """
    def __init__(self, cfg: Config, obs_space: ObsSpace, device: torch.device):
        CuriosityModule.__init__(self)
        nn.Module.__init__(self)

        self.cfg = cfg
        self.device = device
        
        # 1. Normalizador de observações (Input)
        # Atenção: Passamos input_shape e forçamos .to(device)
        obs_key = "obs_noisy" if "obs_noisy" in obs_space.spaces else "obs"
        self.obs_rms = RunningMeanStd(input_shape=obs_space[obs_key].shape).to(device)

        # 2. Redes Neurais (Predictor e Target)
        self.predictor = RNDEncoder(obs_space).to(device)
        self.target = RNDEncoder(obs_space).to(device)

        # Congela a rede Target (ela é fixa e aleatória)
        for p in self.target.parameters():
            p.requires_grad = False

        # 3. Otimizador
        self.rnd_optimizer = torch.optim.Adam(
            self.predictor.parameters(), 
            lr=getattr(cfg, 'rnd_lr', 1e-4)
        )

        # 4. Normalizador de recompensa intrínseca (Output)
        # Normalizador de recompensa intrínseca (Output)
        # norm_only=True garante que não subtraímos a média, mantendo a reward positiva.
        self.intrinsic_reward_rms = RunningMeanStdInPlace(input_shape=(1,), norm_only=True).to(device)        

    def _normalize_obs(self, obs: Tensor) -> Tensor:
        """Normaliza e clipa a observação."""
        # Garante que está na GPU correta
        if obs.device != self.obs_rms.running_mean.device:
            obs = obs.to(self.obs_rms.running_mean.device)

        normalized = self.obs_rms(obs)
        normalized = torch.clamp(normalized, -5, 5)
        return normalized

    def _flatten_5d(self, obs: Tensor, dones: Optional[Tensor] = None):
        """Achata [B, T+1, ...] → [B*(T+1), ...] e retorna shape original.
        dones tem shape [B, T] (sem T+1), então é padded com False antes de achatar."""
        orig_shape = None
        if obs.ndim == 5:
            orig_shape = obs.shape[:2]  # [B, T+1]
            obs = obs.flatten(0, 1)     # [B*(T+1), C, H, W]
            if dones is not None:
                B, T = dones.shape[0], dones.shape[1]
                T1 = orig_shape[1]  # T+1
                if T < T1:
                    # Pad: obs[T+1] não é terminal (é next_obs para bootstrap)
                    pad = torch.zeros(B, T1 - T, dtype=dones.dtype, device=dones.device)
                    dones = torch.cat([dones, pad], dim=1)
                dones = dones.flatten(0, 1)  # [B*(T+1)]
        return obs, dones, orig_shape

    def _build_valid_mask(self, dones: Optional[Tensor], n: int) -> Optional[Tensor]:
        """Cria máscara booleana de obs válidas (não-terminais). True = válido."""
        if dones is None:
            return None
        mask = ~dones.bool()
        if mask.ndim > 1:
            mask = mask.squeeze(-1)
        return mask[:n]

    def calculate_rewards(self, obs_dict: Dict[str, Tensor], dones: Optional[Tensor] = None) -> Tensor:
        """
        Calcula a recompensa intrínseca. Filtra obs terminais (telas pretas).
        Entrada 5D [Batch, Time, C, H, W] é achatada automaticamente.
        """
        obs = obs_dict["obs"]
        obs, dones_flat, orig_shape = self._flatten_5d(obs, dones)

        n = obs.shape[0]
        valid_mask = self._build_valid_mask(dones_flat, n)

        with torch.no_grad():
            # Filtra obs válidas para não poluir obs_rms com telas pretas
            if valid_mask is not None and valid_mask.sum() > 0 and valid_mask.sum() < n:
                obs_valid = obs[valid_mask]
            else:
                obs_valid = obs

            normalized_obs = self._normalize_obs(obs_valid)

            target_feature = self.target(normalized_obs)
            predictor_feature = self.predictor(normalized_obs)
            valid_reward = F.mse_loss(predictor_feature, target_feature, reduction='none').mean(dim=-1)

            # Normaliza reward in-place (divide por running std, mantém positivo)
            reward_for_norm = valid_reward.unsqueeze(-1)
            self.intrinsic_reward_rms(reward_for_norm)
            valid_reward = reward_for_norm.squeeze(-1)

            # Reconstrói tensor completo: obs terminais recebem reward = 0
            if valid_mask is not None and valid_mask.sum() < n:
                full_reward = torch.zeros(n, device=valid_reward.device)
                full_reward[valid_mask] = valid_reward
            else:
                full_reward = valid_reward

        if orig_shape is not None:
            full_reward = full_reward.view(orig_shape)

        return full_reward

    def update(self, obs_dict: Dict[str, Tensor], dones: Tensor) -> Tensor:
        """
        Treina o Predictor apenas em obs válidas (não-terminais).
        obs_rms NÃO é atualizado aqui (já foi em calculate_rewards).
        """
        obs = obs_dict["obs"]
        obs, dones_flat, _ = self._flatten_5d(obs, dones)

        n = obs.shape[0]
        valid_mask = self._build_valid_mask(dones_flat, n)

        # Filtra obs válidas
        if valid_mask is not None and valid_mask.sum() > 0 and valid_mask.sum() < n:
            obs = obs[valid_mask]

        # obs_rms em eval mode: não atualiza stats (já atualizado em calculate_rewards)
        was_training = self.obs_rms.training
        self.obs_rms.eval()
        normalized_obs = self._normalize_obs(obs)
        if was_training:
            self.obs_rms.train()

        predictor_feature = self.predictor(normalized_obs)
        with torch.no_grad():
            target_feature = self.target(normalized_obs)

        loss = F.mse_loss(predictor_feature, target_feature)

        self.rnd_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=1.0)
        self.rnd_optimizer.step()

        return loss

    def get_checkpoint_dict(self) -> Dict[str, Any]:
        return {
            "predictor": self.predictor.state_dict(),
            "target": self.target.state_dict(), 
            "optimizer": self.rnd_optimizer.state_dict(),
            "obs_rms": self.obs_rms.state_dict(),
            "reward_rms": self.intrinsic_reward_rms.state_dict(),
        }

    def load_checkpoint_dict(self, checkpoint_dict: Dict[str, Any]) -> None:
        if "predictor" in checkpoint_dict:
            self.predictor.load_state_dict(checkpoint_dict["predictor"])
            self.target.load_state_dict(checkpoint_dict["target"])
            self.rnd_optimizer.load_state_dict(checkpoint_dict["optimizer"])
            self.obs_rms.load_state_dict(checkpoint_dict["obs_rms"])
            self.intrinsic_reward_rms.load_state_dict(checkpoint_dict["reward_rms"])