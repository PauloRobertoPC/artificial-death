import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.train import run_rl
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components

from utils import add_custom_args, register_custom_doom_env

def main():
    register_vizdoom_components()
    parser, cfg = parse_sf_args()
    add_doom_env_args(parser)
    doom_override_defaults(parser)

    add_custom_args(parser)

    cfg = parse_full_cfg(parser)
    register_custom_doom_env()

    status = run_rl(cfg)
    return status

if __name__ == "__main__":
    sys.exit(main())
