from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("seed", [42]),
    ]
)

_experiments = [
    Experiment(
        "sla1",
        "python sample-factory/sf_examples/vizdoom/train_custom_vizdoom_env.py "
        "--env my_health_gathering_homeostatic "
        "--experiment smoke_test "
        "--train_for_env_steps 20000 "
        "--num_workers 4 "
        "--num_envs_per_worker 2 "
        "--steps_until_decay 25 "
        "--decay_speed 300 "
        "--with_curiosity true "
        "--curiosity_module_type rnd",
        _params.generate_params(randomize=False),
    )
]

RUN_DESCRIPTION = RunDescription(
    "sla2",
    experiments=_experiments,
)

