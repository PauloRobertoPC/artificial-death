import os
from os.path import join, dirname, abspath

# Importa Gymnasium para corrigir o espaço de ação manualmente
import gymnasium as gym
from gymnasium.spaces import Discrete

# Importações do Sample Factory
from sample_factory.envs.env_utils import register_env
from sf_examples.vizdoom.doom.doom_gym import VizdoomEnv


# Importação da Tese
from sf_examples.vizdoom.doom.wrappers.scenario_wrappers.homeostatic_wrapper import HomeostaticGlaucomaWrapper

# --- NOVO IMPORT (Otimização Visual) ---
# Certifique-se de que o arquivo custom_wrappers.py está nessa pasta
from sf_examples.vizdoom.doom.wrappers.custom_wrappers import VizDoomGrayscaleWrapper, GaussianNoiseWrapper, DAEObsWrapper

def add_custom_args(parser):
    """
    Adiciona os hiperparâmetros da Tese HRRL-Sensorial à linha de comando.
    """
    # --- Parâmetros Fisiológicos (Glaucoma) ---
    parser.add_argument("--steps_until_decay", type=int, default=0,
                        help="Janela de tolerância (passos) antes da cegueira iniciar (Alostase)")
    parser.add_argument("--decay_speed", type=int, default=0,
                        help="Quantos pixels são apagados por passo após o fim da tolerância")
    parser.add_argument("--render_mode", type=str, default=None,
                        help="Modo de renderização (human para ver a janela)")
    
    # --- Parâmetros Psicológicos (Curiosidade RND) ---
    # Estes argumentos são necessários para o seu custom_learner.py funcionar
    parser.add_argument("--with_curiosity", type=lambda x: str(x).lower() == 'true', default=False,
                        help="Ativa o módulo de curiosidade customizado")
    parser.add_argument("--curiosity_module_type", type=str, default='rnd',
                        help="Tipo do módulo (ex: rnd)")
    parser.add_argument("--intrinsic_reward_coeff", type=float, default=1.0,
                        help="Peso da recompensa intrínseca na otimização")
    parser.add_argument("--rnd_lr", type=float, default=1e-4,
                        help="Learning rate específico para a rede do RND")
    parser.add_argument("--rnd_ext_coef", type=float, default=0.0,
                        help="Coeficiente para a recompensa extrínseca (0.0 para ignorar recompensa do jogo)")


    # --- Generalização: cenário VizDoom ---
    parser.add_argument("--scenario_cfg", type=str, default="health_gathering.cfg",
                        choices=["health_gathering.cfg", "health_gathering_supreme.cfg",
                                 "health_gathering_supreme3x.cfg", "health_gathering_fixed.cfg",
                                 "health_gathering_fixed_respawn.cfg", "health_gathering_rand_static.cfg",
                                 "health_gathering_dense.cfg"],
                        help="Qual arquivo .cfg de cenário VizDoom carregar. "
                             "'health_gathering.cfg' (default) = baseline do paper (medkits em "
                             "posição ALEATÓRIA, spawn contínuo). "
                             "'health_gathering_fixed.cfg' = 16 medkits em CÍRCULO FIXO sem "
                             "reposição (determinístico; para eval reprodutível e estudo de "
                             "generalização de rota). "
                             "'health_gathering_fixed_respawn.cfg' = círculo fixo COM respawn "
                             "cíclico (1/30 tics em slot vazio), piso FLOOR7_1, dano via ACS "
                             "(tunável). Experimento de generalização de coleta. "
                             "'health_gathering_rand_static.cfg' = mapa aleatório original mas com "
                             "piso FLOOR7_1 + dano ACS (pareado com fixed_respawn p/ isolar layout). "
                             "'health_gathering_supreme.cfg' = versão difícil para testar generalização.")

def make_custom_env(full_env_name, cfg=None, env_config=None, render_mode=None):
    # scenario directory
    script_dir = dirname(abspath(__file__))
    scenarios_dir = join(script_dir, "doom", "scenarios")

    # choosing .cfg scenario
    scenario_cfg_name = getattr(cfg, 'scenario_cfg', 'health_gathering.cfg') if cfg else 'health_gathering.cfg'
    config_path = join(scenarios_dir, scenario_cfg_name)

    if not os.path.exists(config_path):
        fallback_path = join(os.getcwd(), "sf_examples", "vizdoom", "doom", "scenarios", scenario_cfg_name)
        if os.path.exists(fallback_path):
            config_path = fallback_path
        else:
            raise FileNotFoundError(f"configuration file does not exists: {config_path}")

    # render_mode = getattr(cfg, 'render_mode')
    env = VizdoomEnv(
        action_space=Discrete(1+3),
        config_file=config_path,
        skip_frames=4,
        render_mode=render_mode
    )

    # Discrete(N_botões + 1): ação 0 = idle, ações 1..N = um botão cada.
    # health_gathering.cfg tem 3 botões (TURN_LEFT, TURN_RIGHT, MOVE_FORWARD).
    if isinstance(env.action_space, str):
        env.action_space = gym.spaces.Discrete(4)


    # env = SharkWrapper(env)

    # 4. Injeta Fisiologia (condicional para ablação)
    enable_homeo = getattr(cfg, 'enable_homeostasis', True) if cfg else True
    alpha_mode = getattr(cfg, 'alpha_mode', 'coupled') if cfg else 'coupled'
    alpha_delay_steps = getattr(cfg, 'alpha_delay_steps', 0) if cfg else 0
    deg_geom = getattr(cfg, 'degradation_geometry', 'chebyshev') if cfg else 'chebyshev'
    deg_type = getattr(cfg, 'degradation_type', 'blackout') if cfg else 'blackout'

    if enable_homeo:
        steps = cfg.steps_until_decay if cfg else 25
        decay = cfg.decay_speed if cfg else 300
        env = HomeostaticGlaucomaWrapper(
            env, steps_until_decay=steps, decay_speed=decay,
            alpha_mode=alpha_mode, alpha_delay_steps=alpha_delay_steps,
            degradation_geometry=deg_geom, degradation_type=deg_type,
        )
    else:
        # Wrapper presente mas degradação nunca ativa:
        # steps_until_decay=999999 (>> 2100 do episódio), decay_speed=0
        # Mantém: transposta HWC→CHW, reward=0, telemetria (z=1.0 constante)
        env = HomeostaticGlaucomaWrapper(
            env, steps_until_decay=999999, decay_speed=0,
            alpha_mode=alpha_mode, alpha_delay_steps=alpha_delay_steps,
            degradation_geometry=deg_geom, degradation_type=deg_type,
        )

    # 5. --- OTIMIZAÇÃO VISUAL (Grayscale Wrapper) ---
    # Reduz de (3, 240, 320) -> (1, 84, 84)
    # Isso acelera o treino e permite usar a arquitetura Nature CNN no RND
    env = VizDoomGrayscaleWrapper(env, new_shape=(84, 84))

    # 6. --- RUÍDO EXÓGENO / DAE ONLINE ---
    # use_online_dae=True → DAEObsWrapper (obs dict para treino auxiliar)
    # use_online_dae=False + noise_sigma>0 → GaussianNoiseWrapper (eval-only, retrocompatível)
    #
    # Seed do ruído: derivado de cfg.seed e env_config.env_id para que:
    #   - cada env paralelo veja ruído estatisticamente independente (via env_id único);
    #   - experimentos com --seed diferente produzam sequências de ruído distintas;
    #   - C3 e C5 com mesmo --seed vejam exatamente o mesmo ruído (controle contrafactual).
    # Fallback para seed=42 quando cfg ou env_config não estão disponíveis (modo eval/debug).
    use_online_dae = getattr(cfg, 'use_online_dae', False) if cfg else False
    noise_sigma = getattr(cfg, 'exogenous_noise_sigma', 0.0) if cfg else 0.0
    if noise_sigma > 0 or use_online_dae:
        _base_seed = int(getattr(cfg, 'seed', 42) or 42) if cfg else 42
        _env_id = int(env_config.env_id) if env_config is not None and hasattr(env_config, 'env_id') else 0
        _noise_seed = _base_seed * 10000 + _env_id
        _sigma = noise_sigma if noise_sigma > 0 else 25.0
        if use_online_dae:
            env = DAEObsWrapper(env, sigma=_sigma, seed=_noise_seed)
        else:
            env = GaussianNoiseWrapper(env, sigma=_sigma, seed=_noise_seed)

    return env

def register_custom_doom_env():
    register_env("my_health_gathering_homeostatic", make_custom_env)
