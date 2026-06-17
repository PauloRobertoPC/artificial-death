import functools
import os
import sys
from os.path import join, dirname, abspath

# Importa Gymnasium para corrigir o espaço de ação manualmente
import gymnasium as gym

# Importações do Sample Factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components
from sf_examples.vizdoom.doom.doom_gym import VizdoomEnv
from sf_examples.vizdoom.doom.wrappers.scenario_wrappers.SharkWrapper import SharkWrapper


# Importação da Tese
from sf_examples.vizdoom.doom.wrappers.scenario_wrappers.homeostatic_wrapper import HomeostaticGlaucomaWrapper

# --- NOVO IMPORT (Otimização Visual) ---
# Certifique-se de que o arquivo custom_wrappers.py está nessa pasta
from sf_examples.vizdoom.doom.wrappers.custom_wrappers import VizDoomGrayscaleWrapper, GaussianNoiseWrapper, DAEObsWrapper

def add_thesis_args(parser):
    """
    Adiciona os hiperparâmetros da Tese HRRL-Sensorial à linha de comando.
    """
    # --- Parâmetros Fisiológicos (Glaucoma) ---
    parser.add_argument("--steps_until_decay", type=int, default=25,
                        help="Janela de tolerância (passos) antes da cegueira iniciar (Alostase)")
    parser.add_argument("--decay_speed", type=int, default=300,
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

    # --- Controle de Ablação ---
    parser.add_argument("--enable_homeostasis", type=lambda x: str(x).lower() == 'true',
                        default=True,
                        help="Ativa degradação visual homeostática (glaucoma). False desativa para ablação.")

    # --- Ablação 3: Acoplamento Causal ---
    parser.add_argument("--alpha_mode", type=str, default="coupled",
                        choices=["coupled", "delayed"],
                        help="Regra de atualização do alpha visual. 'coupled' = comportamento atual "
                             "(estado interno = visualização). 'delayed' = mascara visual usa o valor "
                             "de alpha defasado por --alpha_delay_steps passos.")
    parser.add_argument("--alpha_delay_steps", type=int, default=0,
                        help="Defasagem k em passos da alpha visual em relação à alpha interna. "
                             "Aplicado apenas se --alpha_mode=delayed. 0 reproduz comportamento coupled.")

    # --- Ablação 1: Geometria da Degradação ---
    parser.add_argument("--degradation_geometry", type=str, default="chebyshev",
                        choices=["chebyshev", "euclidean", "dropout"],
                        help="Forma da região degradada para degradation_type=blackout. "
                             "'chebyshev' = quadrados concêntricos (default, paper). "
                             "'euclidean' = círculos concêntricos. "
                             "'dropout' = pixels aleatórios com prob alpha. "
                             "Ignorado para degradation_type=blur/noise (que são globais).")

    # --- Ablação 2: Tipo de Degradação ---
    parser.add_argument("--degradation_type", type=str, default="blackout",
                        choices=["blackout", "blur", "noise"],
                        help="O que aplicar aos pixels degradados. "
                             "'blackout' = pixels viram 0 (default, paper). "
                             "'blur' = GaussianBlur global, σ ∝ alpha. "
                             "'noise' = ruído gaussiano aditivo global, σ ∝ alpha.")

    # --- Experimento DAE: ruído exógeno de sensor ---
    parser.add_argument("--exogenous_noise_sigma", type=float, default=0.0,
                        help="Sigma do ruído gaussiano exógeno aplicado após grayscale. "
                             "0.0 = desativado (retrocompatível com todos os experimentos anteriores).")

    # --- Experimento DAE Online ---
    parser.add_argument("--use_online_dae", type=lambda x: str(x).lower() == 'true',
                        default=False,
                        help="Ativa DAE online. Substitui GaussianNoiseWrapper por DAEObsWrapper "
                             "(obs dict com obs_noisy + obs_clean para treino auxiliar do AE).")
    parser.add_argument("--dae_arch", type=str, default="shared",
                        choices=["shared", "dual"],
                        help="Arquitetura do DAE online: encoder compartilhado com policy (shared) "
                             "ou encoder dedicado independente (dual).")
    parser.add_argument("--ae_loss_coeff", type=float, default=0.1,
                        help="Peso da perda de reconstrução do AE no gradiente do encoder compartilhado.")
    parser.add_argument("--ae_lr", type=float, default=1e-4,
                        help="Learning rate do otimizador dedicado do AE (ae_optimizer).")
    parser.add_argument("--dae_debug_metrics", type=lambda x: str(x).lower() == 'true',
                        default=False,
                        help="Ativa métricas de diagnóstico do colapso tardio (puramente "
                             "observacionais, não alteram o treino): normas de gradiente do "
                             "encoder separadas por objetivo (APPO vs AE) e cosseno entre elas, "
                             "drift representacional do encoder vs snapshot de referência. "
                             "Off por padrão; runs de produção ficam byte-idênticos.")

    # --- Ablação de fase (C1): desligar o DAE online após K passos ---
    parser.add_argument("--dae_phase_cutoff_steps", type=int, default=0,
                        help="Se > 0, a tarefa auxiliar do DAE online é desligada "
                             "(nenhum ae_optimizer.step) após este nº de env steps. "
                             "Testa a hipótese de ancoragem de fase inicial: se o DAE só "
                             "importa cedo, cutoff pequeno reproduz o C3 completo. "
                             "0 = sem cutoff (DAE ativo o treino todo, comportamento padrão).")

    # --- Baseline de regularização genérica (C4): LayerNorm no encoder conv ---
    parser.add_argument("--encoder_use_layernorm", type=lambda x: str(x).lower() == 'true',
                        default=False,
                        help="Insere GroupNorm(1, C) (equivalente a LayerNorm por canal, "
                             "estável no regime correlacionado do RL) após cada conv do "
                             "encoder, antes da ativação. Baseline de regularização genérica "
                             "para isolar se o ganho do DAE online é específico ou genérico. "
                             "Off por padrão; não toca observação, RND nem Chebyshev.")

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

def make_homeostatic_env(full_env_name, cfg=None, env_config=None, render_mode=None):
    """
    Fábrica do Ambiente da Tese.
    """
    # 1. Resolução robusta de caminhos
    script_dir = dirname(abspath(__file__))
    scenarios_dir = join(script_dir, "doom", "scenarios")

    # Permite escolher entre health_gathering.cfg (default) e health_gathering_supreme.cfg
    # via --scenario_cfg, mantendo retrocompat com todos os experimentos anteriores.
    scenario_cfg_name = getattr(cfg, 'scenario_cfg', 'health_gathering.cfg') if cfg else 'health_gathering.cfg'
    config_path = join(scenarios_dir, scenario_cfg_name)

    if not os.path.exists(config_path):
        fallback_path = join(os.getcwd(), "sf_examples", "vizdoom", "doom", "scenarios", scenario_cfg_name)
        if os.path.exists(fallback_path):
            config_path = fallback_path
        else:
            raise FileNotFoundError(f"Configuração não encontrada em: {config_path}")

    # 2. Define render_mode
    r_mode = render_mode
    if cfg and hasattr(cfg, 'render_mode') and cfg.render_mode is not None:
        r_mode = cfg.render_mode

    # 3. Cria ambiente VizDoom Base
    # Recolocamos action_space="discrete" pois é obrigatório no construtor
    env = VizdoomEnv(
        action_space="discrete", 
        config_file=config_path,
        render_mode=r_mode
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
    register_env("my_health_gathering_homeostatic", make_homeostatic_env)

def main():
    register_vizdoom_components()
    parser, cfg = parse_sf_args()
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    
    add_thesis_args(parser)
    
    cfg = parse_full_cfg(parser)
    register_custom_doom_env()
    
    status = run_rl(cfg)
    return status

if __name__ == "__main__":
    sys.exit(main())