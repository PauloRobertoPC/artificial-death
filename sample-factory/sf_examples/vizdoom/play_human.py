
import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.train import run_rl
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components
from sf_examples.vizdoom.doom.doom_gym import VizdoomEnv

from utils import add_custom_args, make_custom_env

def main():
    register_vizdoom_components()
    parser, cfg = parse_sf_args()
    add_doom_env_args(parser)
    doom_override_defaults(parser)

    add_custom_args(parser)

    cfg = parse_full_cfg(parser)

    # Force rendering so the game window is visible
    cfg.render_mode = "human"

    env = make_custom_env(
        full_env_name="my_health_gathering_homeostatic",
        cfg=cfg,
        env_config=None,
        render_mode="human",
    )

    return_value = 0
    try:
        return_value = VizdoomEnv.play_human_mode(env)
    finally:
        env.close()
    return return_value

if __name__ == "__main__":
    sys.exit(main())
