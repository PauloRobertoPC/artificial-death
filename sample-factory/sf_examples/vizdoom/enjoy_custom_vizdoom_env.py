import sys

# Importações do Sample Factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.enjoy import enjoy
from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components

# --- CORREÇÃO DE IMPORTAÇÃO ---
# Estamos importando 'add_thesis_args', que é o nome correto definido no seu script de treino.
from sf_examples.vizdoom.train_custom_vizdoom_env import (
    register_custom_doom_env, 
    add_thesis_args  # <--- Nome correto aqui
)

def main():
    """
    Script dedicado para visualizar (enjoy) o ambiente da Tese.
    """
    # 1. Registra componentes básicos do VizDoom
    register_vizdoom_components()

    # 2. Registra o SEU ambiente customizado
    register_custom_doom_env()

    # 3. Configura os argumentos
    parser, cfg = parse_sf_args(evaluation=True)
    
    # Adiciona os argumentos da tese
    add_thesis_args(parser) # <--- Chamada correta aqui
    
    # Processa a configuração final
    cfg = parse_full_cfg(parser)

    # 4. Executa a visualização
    status = enjoy(cfg)
    return status

if __name__ == "__main__":
    sys.exit(main())