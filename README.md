# DRL Curiosity 🧠🎮

Projeto de **Aprendizado por Reforço Profundo (Deep Reinforcement Learning)**
utilizando **[Sample Factory](https://github.com/alex-petrenko/sample-factory)**,
**VizDoom**, **Gymnasium** e **PyTorch**.

O agente é treinado no cenário `health_gathering` com **degradação visual simulada**:
a visão do agente se deteriora quanto mais tempo ele passa sem coletar um medkit.
O aprendizado é guiado por **curiosidade intrínseca (RND — Random Network Distillation)**.
Scripts de **Grad-CAM / Score-CAM** permitem inspecionar o que as redes neurais
"olham" em diferentes níveis de degradação visual.

---

## 🚀 Tecnologias Utilizadas

- [Python 3.12+](https://www.python.org/)
- [PyTorch 2.6.0](https://pytorch.org/)
- [Sample Factory 2.1.1 (customizado/vendored)](https://github.com/alex-petrenko/sample-factory)
- [VizDoom 1.2.4](https://vizdoom.farama.org/)
- [Gymnasium 0.29.1](https://gymnasium.farama.org/)
- [OpenCV 4.11.0.86](https://opencv.org/)
- [NumPy 1.26.4](https://numpy.org/)
- [TensorBoard 2.19.0](https://www.tensorflow.org/tensorboard)

---

## ⚙️ Instalação

### 1. Dependências de sistema (Ubuntu / Linux Mint / Debian)

O VizDoom é compilado a partir do código-fonte na instalação, então algumas
bibliotecas nativas são necessárias:

```bash
sudo apt update
sudo apt install -y build-essential cmake git \
  libboost-all-dev libsdl2-dev libopenal-dev libopenal1
```

### 2. Instale o `uv`

Este projeto usa o **[uv](https://github.com/astral-sh/uv)** para gerenciar o
ambiente e as dependências. Caso não tenha:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# reinicie o shell, ou: source $HOME/.local/bin/env
```

### 3. Crie o ambiente

A partir da raiz do projeto:

```bash
uv sync
```

O `uv` provisiona o Python 3.12 automaticamente, instala todas as dependências e
vincula a cópia local do `sample-factory/` em modo editável. A primeira execução
compila o VizDoom e pode levar alguns minutos.

> Se o `uv sync` reclamar que o lockfile está desatualizado, rode `uv lock` uma
> vez e depois `uv sync` novamente.

---

## ✅ Teste rápido (sem treinar)

Verifica se o ambiente, os wrappers e o VizDoom carregam corretamente:

```bash
uv run python -c "
from sf_examples.vizdoom.train_custom_vizdoom_env import register_custom_doom_env
register_custom_doom_env()
print('ambiente registrado OK')
"
```

Se imprimir `ambiente registrado OK`, está tudo certo para treinar.

---

## 🏋️ Treinamento

Execução curta, para validar o loop de treino de ponta a ponta (~1-2 min):

```bash
uv run python sample-factory/sf_examples/vizdoom/train_custom_vizdoom_env.py \
  --env my_health_gathering_homeostatic \
  --experiment smoke_test \
  --train_for_env_steps 20000 \
  --num_workers 4 --num_envs_per_worker 2 \
  --steps_until_decay 25 --decay_speed 300 \
  --with_curiosity true --curiosity_module_type rnd
```

Para um treino completo, basta remover o limite `--train_for_env_steps` (ou
defini-lo bem alto). Os resultados (checkpoints, logs do TensorBoard) vão para
`train_dir/<experiment>/`.

**Flags úteis:**

| Flag | Descrição |
|------|-----------|
| `--steps_until_decay` | Passos sem medkit antes de a visão começar a degradar |
| `--decay_speed` | Pixels apagados por passo após o início da degradação |
| `--with_curiosity true --curiosity_module_type rnd` | Ativa a recompensa intrínseca RND |
| `--intrinsic_reward_coeff`, `--rnd_lr`, `--rnd_ext_coef` | Hiperparâmetros do RND |

Acompanhe o treino com:

```bash
uv run tensorboard --logdir train_dir
```

---

## 🏋️ Experimentos

```bash
uv run python sample-factory/sample_factory/launcher/run.py  --run=sf_examples.vizdoom.experiments.extrinsic --backend=processes --max_parallel=1  --pause_between=1 --experiments_per_gpu=1 --num_gpus=1
```

---

## 🎯 Avaliação

Avalia um agente já treinado:

```bash
uv run python sample-factory/sf_examples/vizdoom/enjoy_custom_vizdoom_env.py \
  --env my_health_gathering_homeostatic \
  --experiment smoke_test \
  --no_render --max_num_episodes 5
```

---

## 🔍 Análise de atenção (Grad-CAM / Score-CAM)

Estes scripts carregam um checkpoint treinado de `train_dir/<experiment>/` e
geram mapas de calor de atenção — portanto, treine um agente antes.

```bash
uv run python gradcam_analysis.py  --experiment smoke_test
uv run python scorecam_analysis.py --experiment smoke_test
```

`gradcam_episode.py` / `scorecam_episode.py` geram mapas de calor passo a passo
ao longo de um episódio completo.

---

## 🏋️ Play as human

Execução curta, para validar o loop de treino de ponta a ponta (~1-2 min):

```bash
uv run python sample-factory/sf_examples/vizdoom/play_human.py \
  --env my_health_gathering_homeostatic \
  --scenario_cfg health_gathering.cfg \
  --steps_until_decay 0 --decay_speed 10 \
  --game_layout 0
```

---

## 📁 Estrutura do projeto

```
sample-factory/        Sample Factory vendored + customizado (motor de RL e ambiente)
  sf_examples/vizdoom/
    train_custom_vizdoom_env.py   Entry point: registro do ambiente + config de treino
    enjoy_custom_vizdoom_env.py   Entry point de avaliação
    doom/wrappers/                Wrappers customizados do ambiente
gradcam_analysis.py    Mapas de atenção Grad-CAM (run completa)
gradcam_episode.py     Mapas de atenção Grad-CAM (passo a passo)
scorecam_analysis.py   Mapas de atenção Score-CAM (run completa)
scorecam_episode.py    Mapas de atenção Score-CAM (passo a passo)
pyproject.toml         Dependências; fixa o sample-factory local como editável
```

TESTAR AMBIENTE COM O PLAY HUMAN
    TESTAR WRAPPERS
        TESTAR SAVE TRAJECTORY WRAPPER
REMOVER HOMEOSTASE E DEIXAR SOMENTE RND
ENTENDER COMO PEGAR OS DADOS E FAZER VISUALIZAÇÕES
PENSAR E FAZER EXPERIMENTOS
