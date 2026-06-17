import collections

try:
    import cv2  # opcional — usado apenas para degradation_type='blur'
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

import gymnasium as gym
import numpy as np

# Intensidade máxima dos modos não-blackout. Calibrados empiricamente para que
# o efeito visual seja comparável ao blackout total quando α = 1 (visão crítica).
_MAX_BLUR_SIGMA = 8.0      # cv2.GaussianBlur(σ=8) em imagem 240×320 satura legibilidade
_MAX_NOISE_SIGMA = 64.0    # ruído N(0, 64²) em uint8 [0,255] domina o sinal


class HomeostaticGlaucomaWrapper(gym.Wrapper):
    """
    Versão FINAL 'BLINDADA' para Tese HRRL.
    Estratégia: Criação de nova imagem preta e colagem da área visível
    para evitar Ghosting/Transparência do buffer do VizDoom.

    Suporta dois modos de acoplamento (Ablação 3 — Acoplamento Causal):
      - alpha_mode="coupled" (default): a máscara visual usa o estado interno
        atual de pixels_erased — comportamento original do paper.
      - alpha_mode="delayed": a máscara visual usa o valor de pixels_erased
        defasado em `alpha_delay_steps` passos. O estado interno continua
        atualizando em sincronia com a coleta, mas a restauração visual só
        aparece k passos depois — desacopla coleta e spike de novidade.

    Suporta variantes de geometria e tipo de degradação (Ablação 1 e 2):
      - degradation_geometry ∈ {chebyshev, euclidean, dropout}: define a
        REGIÃO ocluída quando degradation_type='blackout'. Ignorado para
        blur/noise globais.
      - degradation_type ∈ {blackout, blur, noise}:
          * blackout (default): pixels da região viram 0 (comportamento do paper)
          * blur: cv2.GaussianBlur global com σ proporcional a α (escala 0→σ_max)
          * noise: imagem += N(0, σ²) global com σ proporcional a α
    """

    def __init__(self, env: gym.Env,
                 steps_until_decay: int = 100,
                 decay_speed: int = 5,
                 alpha_mode: str = "coupled",
                 alpha_delay_steps: int = 0,
                 degradation_geometry: str = "chebyshev",
                 degradation_type: str = "blackout"):

        super().__init__(env)
        self.env = env
        self.steps_until_decay = steps_until_decay
        self.decay_speed = decay_speed

        # Ablação 3 — Acoplamento Causal
        assert alpha_mode in ("coupled", "delayed"), \
            f"alpha_mode inválido: {alpha_mode!r}. Use 'coupled' ou 'delayed'."
        assert alpha_delay_steps >= 0, "alpha_delay_steps deve ser >= 0."
        self.alpha_mode = alpha_mode
        self.alpha_delay_steps = alpha_delay_steps

        # Ablação 1 & 2 — Geometria e Tipo de Degradação
        assert degradation_geometry in ("chebyshev", "euclidean", "dropout"), \
            f"degradation_geometry inválido: {degradation_geometry!r}."
        assert degradation_type in ("blackout", "blur", "noise"), \
            f"degradation_type inválido: {degradation_type!r}."
        if degradation_type == "blur" and not _HAS_CV2:
            raise RuntimeError("degradation_type='blur' requer opencv-python instalado.")
        self.degradation_geometry = degradation_geometry
        self.degradation_type = degradation_type

        # --- CONFIGURAÇÃO DE ESPAÇO ---
        h, w, c = env.observation_space.shape
        # Garante que o ambiente original é HWC (padrão VizDoom)
        assert h > c and w > c, "Erro: O ambiente original não parece ser (H, W, C)."

        self.h_orig, self.w_orig, self.c_orig = h, w, c

        # Define o espaço de saída como CHW (padrão Sample Factory/PyTorch)
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(c, h, w),
            dtype=np.uint8
        )

        # Estado Interno
        self.hunger_steps = 0
        self.pixels_erased = 0
        self.prev_health = None
        self.total_pixels = h * w

        # Buffer para alpha_mode="delayed": guarda os últimos (k+1) valores
        # de pixels_erased. O elemento [0] é o valor de k passos atrás
        # (head = oldest, tail = current). Em modo coupled, fica vazio.
        self._pixels_erased_buffer = collections.deque(maxlen=self.alpha_delay_steps + 1)

        # Telemetria de coletas (Bloco C1/C3 do experiments_plan)
        self._medkit_count = 0
        self._first_collection_step = -1
        self._z_at_collection_sum = 0.0
        self._z_at_first_collection = -1.0
        self._step_idx = 0

        # --- MATRIZ DE RANK (geometria de degradação) ---
        # Chebyshev: quadrados concêntricos (norma L_inf)
        # Euclidean: círculos concêntricos (norma L_2)
        # Dropout: não usa rank_matrix; sorteio aleatório por pixel
        if self.degradation_geometry == "chebyshev":
            self.rank_matrix = self._precompute_chebyshev_rank(h, w)
        elif self.degradation_geometry == "euclidean":
            self.rank_matrix = self._precompute_euclidean_rank(h, w)
        else:  # dropout
            self.rank_matrix = None
        self.history_homeostasis = []
        # RNG para dropout (sample por pixel), independente do RNG do treino
        # para reprodutibilidade dentro do wrapper.
        self._dropout_rng = np.random.default_rng(seed=12345)

    def reset(self, **kwargs):
        self.hunger_steps = 0
        self.pixels_erased = 0
        self.prev_health = None
        self.history_homeostasis = []

        # Reseta buffer de delay: pré-popula com zeros (estado de visão saudável)
        self._pixels_erased_buffer.clear()
        for _ in range(self.alpha_delay_steps + 1):
            self._pixels_erased_buffer.append(0)

        # Reseta telemetria de coletas
        self._medkit_count = 0
        self._first_collection_step = -1
        self._z_at_collection_sum = 0.0
        self._z_at_first_collection = -1.0
        self._step_idx = 0

        obs, info = self.env.reset(**kwargs)

        # Processamento centralizado
        obs_chw = self._process_obs_robust(obs)

        return obs_chw, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated | truncated
        self._step_idx += 1

        # --- LÓGICA FISIOLÓGICA DA TESE ---
        curr_health = info.get("HEALTH", 0.0)
        if self.prev_health is None: self.prev_health = curr_health

        ate_kit = curr_health > self.prev_health
        self.prev_health = curr_health

        if ate_kit:
            self.hunger_steps = 0
            self.pixels_erased = 0 # Cura total da visão
        else:
            self.hunger_steps += 1
            if self.hunger_steps > self.steps_until_decay:
                self.pixels_erased += self.decay_speed

        self.pixels_erased = min(self.pixels_erased, self.total_pixels)

        # Atualiza buffer de delay (no modo coupled o buffer é mantido em
        # sincronia mas não consultado).
        self._pixels_erased_buffer.append(self.pixels_erased)

        # --- PROCESSAMENTO VISUAL BLINDADO ---
        # No modo "coupled" usa o estado atual; no modo "delayed" usa o valor
        # de k passos atrás. O head do deque é o elemento mais antigo dentro
        # da janela de tamanho (k+1).
        if self.alpha_mode == "delayed":
            pixels_erased_view = self._pixels_erased_buffer[0]
        else:
            pixels_erased_view = self.pixels_erased
        obs_chw = self._process_obs_robust(obs, pixels_erased_view)

        # --- TELEMETRIA ---
        # z_homeostasis reflete o ESTADO INTERNO (não o defasado) — mantém
        # comparabilidade com runs históricas onde os dois são idênticos.
        current_homeostasis = 1.0 - (self.pixels_erased / self.total_pixels)
        self.history_homeostasis.append(current_homeostasis)

        if "episode_extra_stats" not in info: info["episode_extra_stats"] = {}
        info["episode_extra_stats"]["z_homeostasis"] = current_homeostasis
        info["episode_extra_stats"]["z_hunger_max"] = self.hunger_steps
        # z_view = o que o agente realmente vê (= z_homeostasis no modo coupled)
        info["episode_extra_stats"]["z_view"] = 1.0 - (pixels_erased_view / self.total_pixels)

        # Métricas de coleta (Bloco C1/C3 do experiments_plan)
        if ate_kit:
            self._medkit_count += 1
            self._z_at_collection_sum += current_homeostasis
            if self._first_collection_step < 0:
                self._first_collection_step = self._step_idx
                self._z_at_first_collection = current_homeostasis

        if done:
            avg = np.mean(self.history_homeostasis) if self.history_homeostasis else 1.0
            info["episode_extra_stats"]["z_homeostasis_avg"] = avg
            info["episode_extra_stats"]["z_final_homeostasis"] = current_homeostasis
            info["episode_extra_stats"]["medkit_collected_count"] = self._medkit_count
            info["episode_extra_stats"]["time_to_first_medkit"] = float(self._first_collection_step)
            info["episode_extra_stats"]["z_at_first_collection"] = self._z_at_first_collection
            if self._medkit_count > 0:
                info["episode_extra_stats"]["z_at_collection_mean"] = self._z_at_collection_sum / self._medkit_count
            else:
                info["episode_extra_stats"]["z_at_collection_mean"] = -1.0

        # Zera recompensa do jogo para garantir aprendizado via RND
        blind_reward = 0.0

        return obs_chw, blind_reward, terminated, truncated, info

    def _process_obs_robust(self, obs_hwc, pixels_erased=None):
        """Despacha o tipo de degradação a aplicar.

        Entrada:
            obs_hwc: uint8 (H, W, C) do VizDoom.
            pixels_erased: intensidade de degradação acumulada. Se None,
                usa self.pixels_erased.
        Saída: obs_chw (uint8, C-contíguo).

        Despacho:
            type='blackout' → _apply_blackout (varia com self.degradation_geometry)
            type='blur'     → _apply_blur_global (geometria ignorada)
            type='noise'    → _apply_noise_global (geometria ignorada)
        """
        if pixels_erased is None:
            pixels_erased = self.pixels_erased

        # Caso degenerado: nenhuma degradação acumulada, só transpõe e retorna.
        if pixels_erased <= 0:
            return np.ascontiguousarray(np.transpose(obs_hwc, (2, 0, 1)))

        if self.degradation_type == "blackout":
            return self._apply_blackout(obs_hwc, pixels_erased)
        elif self.degradation_type == "blur":
            return self._apply_blur_global(obs_hwc, pixels_erased)
        elif self.degradation_type == "noise":
            return self._apply_noise_global(obs_hwc, pixels_erased)
        else:
            raise ValueError(f"degradation_type inválido: {self.degradation_type!r}")

    def _apply_blackout(self, obs_hwc, pixels_erased):
        """Zera pixels da região indicada por self.degradation_geometry."""
        h, w, _ = obs_hwc.shape
        if self.degradation_geometry == "chebyshev":
            # Quadrado preto crescente do centro (norma L_inf)
            threshold = int(np.sqrt(pixels_erased) / 2)
            mask_to_keep = self.rank_matrix >= threshold
        elif self.degradation_geometry == "euclidean":
            # Círculo preto crescente do centro (norma L_2). Mesma "quantidade
            # de pixels apagados" mapeada para o raio de uma circunferência:
            # área ≈ π·r² → r = sqrt(pixels_erased / π).
            radius = np.sqrt(pixels_erased / np.pi)
            mask_to_keep = self.rank_matrix >= radius
        else:  # dropout
            # Sorteia uma fração α = pixels_erased / total_pixels de pixels
            # para apagar. Resorteia a cada step (estocástico). Independente da posição.
            p_erase = pixels_erased / self.total_pixels
            mask_to_erase = self._dropout_rng.random(size=(h, w)) < p_erase
            mask_to_keep = ~mask_to_erase

        # Imagem nova totalmente preta + colagem da parte visível
        final_obs_hwc = np.zeros(obs_hwc.shape, dtype=np.uint8)
        final_obs_hwc[mask_to_keep] = obs_hwc[mask_to_keep]
        return np.ascontiguousarray(np.transpose(final_obs_hwc, (2, 0, 1)))

    def _apply_blur_global(self, obs_hwc, pixels_erased):
        """Gaussian blur global na imagem inteira com σ proporcional a α."""
        alpha = pixels_erased / self.total_pixels  # [0, 1]
        sigma = max(0.1, alpha * _MAX_BLUR_SIGMA)
        # cv2.GaussianBlur com ksize=(0,0) deduz o kernel a partir do sigma.
        # Trabalha em HWC nativo.
        blurred = cv2.GaussianBlur(obs_hwc, (0, 0), sigmaX=sigma, sigmaY=sigma)
        return np.ascontiguousarray(np.transpose(blurred, (2, 0, 1)))

    def _apply_noise_global(self, obs_hwc, pixels_erased):
        """Adiciona ruído gaussiano N(0, σ²) à imagem inteira; σ ∝ α."""
        alpha = pixels_erased / self.total_pixels  # [0, 1]
        sigma = alpha * _MAX_NOISE_SIGMA
        if sigma <= 0:
            return np.ascontiguousarray(np.transpose(obs_hwc, (2, 0, 1)))
        # Ruído por pixel; tipo float pra evitar overflow, depois clip e cast.
        noise = self._dropout_rng.normal(loc=0.0, scale=sigma, size=obs_hwc.shape)
        noisy = obs_hwc.astype(np.int16) + noise.astype(np.int16)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(np.transpose(noisy, (2, 0, 1)))

    def _precompute_chebyshev_rank(self, h, w):
        """Gera matriz de ranks baseada na Distância de Chebyshev (Quadrados Concêntricos)."""
        center_y, center_x = h // 2, w // 2
        y_indices, x_indices = np.indices((h, w))
        y_dist = np.abs(y_indices - center_y)
        x_dist = np.abs(x_indices - center_x)
        # O rank é a maior distância em qualquer eixo (define um quadrado)
        rank_matrix = np.maximum(y_dist, x_dist)
        return rank_matrix

    def _precompute_euclidean_rank(self, h, w):
        """Gera matriz de ranks baseada na Distância Euclidiana (Círculos Concêntricos)."""
        center_y, center_x = h // 2, w // 2
        y_indices, x_indices = np.indices((h, w))
        # rank = sqrt((y-cy)² + (x-cx)²) — distância L_2 ao centro
        rank_matrix = np.sqrt(
            (y_indices - center_y) ** 2 + (x_indices - center_x) ** 2
        )
        return rank_matrix