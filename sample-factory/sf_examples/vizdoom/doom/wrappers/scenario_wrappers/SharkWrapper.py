import gymnasium as gym
import numpy as np
import vizdoom.vizdoom as vizdoom  # Import necessário para ler variáveis do jogo

class SharkWrapper(gym.Wrapper):
    """
    Wrapper 'Tubarão': Força o movimento para frente.
    
    DIFERENCIAL: Este wrapper executa a ação DIRETAMENTE no game engine,
    bypassando a verificação de tipo do doom_gym.py que estava causando o erro.
    
    Ações (Discretas 3):
    0: Só Frente
    1: Frente + Esquerda
    2: Frente + Direita
    """
    def __init__(self, env):
        super().__init__(env)
        # O agente vê apenas 3 opções discretas
        self.action_space = gym.spaces.Discrete(3)
        
        # Encontra o ambiente VizDoom real "embaixo" dos wrappers
        self.doom_env = env.unwrapped
        
        # Descobre quantos botões o cenário tem configurado no .cfg
        # (Ex: Move_Forward, Turn_Left, Turn_Right = 3 botões)
        self.num_buttons = self.doom_env.game.get_available_buttons_size()

    def step(self, action_idx):
        # 1. TRADUÇÃO: Índice -> Vetor de Botões
        # Cria um vetor de zeros com o tamanho exato que o jogo espera
        buttons = [0.0] * self.num_buttons
        
        # Lógica do Tubarão (Sempre Frente)
        # Assume que Botão 0 é MOVE_FORWARD (padrão health_gathering)
        buttons[0] = 1.0 
        
        # Assume que Botão 1 é TURN_LEFT
        if action_idx == 1: 
            if len(buttons) > 1: buttons[1] = 1.0
            
        # Assume que Botão 2 é TURN_RIGHT
        if action_idx == 2: 
            if len(buttons) > 2: buttons[2] = 1.0
            
        # 2. EXECUÇÃO DIRETA (Bypass doom_gym.py)
        # Chamamos o motor C++ diretamente. Isso evita o erro 'ValueError'
        # skip_frames é a config de frameskip do seu ambiente (geralmente 4)
        skip = self.doom_env.skip_frames
        reward = self.doom_env.game.make_action(buttons, skip)
        
        # 3. RECONSTRUÇÃO DO ESTADO
        # Como bypassamos o step() padrão, precisamos pegar o obs manualmente
        state = self.doom_env.game.get_state()
        done = self.doom_env.game.is_episode_finished()
        
        info = {}
        
        if state:
            # Pega a imagem (HWC)
            obs = state.screen_buffer
            
            # RECRÍTICO: Extrair 'HEALTH' para o GlaucomaWrapper funcionar!
            # Tenta pegar a variável de vida do jogo
            try:
                # GameVariable.HEALTH é o padrão interno do VizDoom
                health_val = self.doom_env.game.get_game_variable(vizdoom.GameVariable.HEALTH)
                info["HEALTH"] = health_val
            except Exception:
                pass # Se falhar, o Glaucoma usará o último valor conhecido
        else:
            # Fim de episódio, retorna tela preta segura
            obs = np.zeros(self.env.observation_space.shape, dtype=np.uint8)

        truncated = False # SampleFactory gerencia timeout externamente geralmente

        return obs, reward, done, truncated, info