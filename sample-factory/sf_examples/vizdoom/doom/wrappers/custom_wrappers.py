# Arquivo: sample-factory/sf_examples/vizdoom/doom/wrappers/custom_wrappers.py
import cv2
import gymnasium as gym
import numpy as np

class VizDoomGrayscaleWrapper(gym.ObservationWrapper):
    """
    Converte a observação do VizDoom para Escala de Cinza e Redimensiona.
    Reduz drasticamente o uso de memória e acelera o RND.
    """
    def __init__(self, env, new_shape=(84, 84)):
        super().__init__(env)
        self.new_shape = new_shape
        
        # Define o novo espaço de observação: 1 canal (Cinza)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(1, new_shape[0], new_shape[1]), 
            dtype=np.uint8
        )

    def observation(self, obs):
        # VizDoom pode retornar (C, H, W) ou (H, W, C).
        # O OpenCV precisa de (H, W, C) C-contíguo.

        if obs.shape[0] == 3: # Se for (3, H, W)
             img = np.ascontiguousarray(np.moveaxis(obs, 0, -1)) # (H, W, 3) contíguo
        else:
             img = np.ascontiguousarray(obs)

        # Converte para Cinza
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Redimensiona
        resized = cv2.resize(gray, (self.new_shape[1], self.new_shape[0]), interpolation=cv2.INTER_NEAREST)

        # Adiciona dimensão de canal: (1, H, W) para o Sample Factory
        return resized[None, :, :]


class GaussianNoiseWrapper(gym.ObservationWrapper):
    """Adds additive Gaussian noise independent of the homeostatic state."""

    def __init__(self, env, sigma: float = 0.0, seed: int = 42):
        super().__init__(env)
        self.sigma = sigma
        self._base_seed = int(seed)
        self._rng = np.random.default_rng(seed=seed)

    def set_eval_noise_seed(self, ep_seed: int):
        """Reinicializa o RNG do ruído de forma determinística para AVALIAÇÃO.

        Uso EXCLUSIVO de eval: o caller (scripts de eval) chama isto antes de
        cada episódio com ep_seed fixo, tornando a sequência de ruído reprodutível
        (mesma ep_seed -> mesmo ruído). NÃO é chamado no treino, então o
        comportamento de treino fica byte-idêntico ao original (o RNG continua
        criado uma vez no __init__ e avançando naturalmente durante o treino).
        Deriva de base ^ ep_seed para que ep_seeds distintas deem ruídos distintos.
        """
        self._rng = np.random.default_rng(seed=self._base_seed ^ int(ep_seed))

    def observation(self, obs):
        if self.sigma <= 0:
            return obs
        noise = self._rng.normal(0, self.sigma, obs.shape)
        noisy = obs.astype(np.int16) + noise.astype(np.int16)
        return np.clip(noisy, 0, 255).astype(np.uint8)


class DAEObsWrapper(gym.ObservationWrapper):
    """Wraps observation into dict {obs_noisy, obs_clean} for online DAE training.

    obs_clean = observation after GlaucomaWrapper (endogenous signal intact).
    obs_noisy = obs_clean + GaussianNoise(sigma) (exogenous corruption added here).
    The policy forward pass uses obs_noisy; obs_clean is the AE reconstruction target.
    Replaces GaussianNoiseWrapper when --use_online_dae=True."""

    def __init__(self, env, sigma: float = 25.0, seed: int = 42):
        super().__init__(env)
        self.sigma = sigma
        self._base_seed = int(seed)
        self._rng = np.random.default_rng(seed=seed)
        single = env.observation_space
        self.observation_space = gym.spaces.Dict({
            "obs_noisy": single,
            "obs_clean": single,
        })

    def set_eval_noise_seed(self, ep_seed: int):
        """Ver GaussianNoiseWrapper.set_eval_noise_seed: reseed determinístico do
        RNG do ruído para AVALIAÇÃO (uso exclusivo de eval; não chamado no treino,
        que fica byte-idêntico ao original)."""
        self._rng = np.random.default_rng(seed=self._base_seed ^ int(ep_seed))

    def observation(self, obs):
        noise = self._rng.normal(0, self.sigma, obs.shape)
        noisy = np.clip(obs.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
        return {"obs_noisy": noisy, "obs_clean": obs}