"""Run an experiment."""
import os
from runpy import run_path

import wandb

# don't need docstrings in scripts
# pylint: disable=missing-function-docstring,missing-return-doc,missing-return-type-doc
# ruff: noqa: D103


def run_experiment(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config) as run:  # type: ignore
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = run.config
        print(f"{config=}")
        for param in config.keys():
            print(f"setting sweep param into env: {param} = {config[param]}")
            # os.environ[param] = config[param]
            os.environ[param] = str(config[param])

        # run the experiment in interactive_econ.py
        # import examples.interactive_econ  # pylint: disable=import-outside-toplevel,unused-import
        run_path("interactive_econ.py")


if __name__ == "__main__":
    run_experiment()
