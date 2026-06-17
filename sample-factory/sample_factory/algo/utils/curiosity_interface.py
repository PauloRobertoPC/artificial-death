"""
Define a interface abstrata para Módulos de Curiosidade.

Este arquivo estabelece o contrato que qualquer algoritmo de recompensa intrínseca (como RND, ICM, etc.)
deve seguir para se integrar de forma 'plug-and-play' com a classe Learner do Sample Factory.

Princípios de Design:
- Desacoplamento: O Learner não deve ter conhecimento da implementação específica do módulo de curiosidade.
- Responsabilidade Única: A interface define claramente as responsabilidades: inicializar, calcular recompensas,
  atualizar o estado interno e persistir o estado.
- Contrato de Dados Forte: Especifica as assinaturas exatas dos métodos, incluindo os parâmetros,
  garantindo consistência em todas as implementações.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

import torch
from torch import Tensor

class CuriosityModule(ABC):
    """Interface base para módulos de curiosidade."""

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_rewards(self, obs_dict: Dict[str, Tensor], dones: Tensor) -> Tensor:
        """Retorna tensor de recompensas (batch_size,). dones marca transições terminais a ignorar."""
        pass

    @abstractmethod
    def update(self, obs_dict: Dict[str, Tensor], dones: Tensor) -> Tensor:
        """Treina o modelo e retorna o loss (escalar)."""
        pass

    @abstractmethod
    def get_checkpoint_dict(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def load_checkpoint_dict(self, checkpoint_dict: Dict[str, Any]) -> None:
        pass