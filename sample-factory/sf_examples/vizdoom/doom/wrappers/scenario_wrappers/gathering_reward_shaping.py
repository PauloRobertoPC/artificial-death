import gymnasium as gym
import numpy as np

class DoomGatheringRewardShaping(gym.Wrapper):
    """
    Wrapper modificado para Tese de Controle (Sobrevivência Cega).
    
    Responsabilidades:
    1. LOBOTOMIA: Força a recompensa a ser sempre 0.0, independentemente da saúde.
    2. TELEMETRIA: Coleta estatísticas vitais (Média, Final, Máxima) para o Tensorboard.
    """

    def __init__(self, env):
        super().__init__(env)
        # Apenas o histórico é necessário agora. Removemos _prev_health.
        self.health_history = []

    def reset(self, **kwargs):
        # Limpa o histórico no início de cada episódio
        self.health_history = []
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated | truncated

        # --- 1. LÓGICA DE TELEMETRIA (MONITORAMENTO) ---
        # Captura a saúde atual (se disponível no .cfg)
        curr_health = info.get("HEALTH", 0.0)
        self.health_history.append(curr_health)
        
        # --- 2. LÓGICA DE RECOMPENSA (LOBOTOMIA) ---
        # Ignoramos a reward original (+1 ou -1) e retornamos 0.0 absoluto.
        # O agente não recebe sinal explícito se suas ações são boas ou ruins.
        modified_reward = 0.0

        # --- 3. ENVIO PARA O TENSORBOARD ---
        if done and len(self.health_history) > 0:
            # Calcula estatísticas finais
            avg_health = np.mean(self.health_history)
            final_health = curr_health
            max_health = np.max(self.health_history)

            # Injeta no dicionário 'episode_extra_stats'
            if "episode_extra_stats" not in info:
                info["episode_extra_stats"] = {}
            
            # O prefixo 'z_' agrupa essas métricas no final da lista do Tensorboard
            info["episode_extra_stats"]["z_avg_health"] = avg_health
            info["episode_extra_stats"]["z_final_health"] = final_health
            info["episode_extra_stats"]["z_max_health"] = max_health

        return observation, modified_reward, terminated, truncated, info