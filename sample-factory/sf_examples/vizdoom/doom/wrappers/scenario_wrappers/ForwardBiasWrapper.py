import gymnasium as gym
import numpy as np

class ForwardBiasWrapper(gym.ActionWrapper):
    """
    Obriga o agente a ter um viés de movimento para frente.
    Remove a opção de 'Girar Parado'.
    
    Mapeamento de Ações (Discrete 3):
    0 -> Move Forward (Apenas Frente)
    1 -> Move Forward + Turn Left (Curva Aberta Esquerda)
    2 -> Move Forward + Turn Right (Curva Aberta Direita)
    """
    def __init__(self, env):
        super().__init__(env)
        # Assume que o ambiente original tem botões: [MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, ...]
        # Vamos redefinir o espaço para apenas 3 ações discretas
        self.action_space = gym.spaces.Discrete(3)
        
    def action(self, action_idx):
        # Cria um vetor de ações binárias (padrão VizDoom)
        # Supondo que a ordem dos botões no cfg seja:
        # [MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]
        
        # Inicializa tudo com 0.0
        # O tamanho depende de quantos botões você configurou no .cfg do Doom
        # Geralmente 3 para movimento básico
        env_action = np.zeros(3, dtype=np.float32) 
        
        # SEMPRE aplica movimento para frente (Botão 0)
        env_action[0] = 1.0 
        
        if action_idx == 0:
            # Apenas Frente
            pass 
        elif action_idx == 1:
            # Frente + Esquerda
            env_action[1] = 1.0
        elif action_idx == 2:
            # Frente + Direita
            env_action[2] = 1.0
            
        return env_action