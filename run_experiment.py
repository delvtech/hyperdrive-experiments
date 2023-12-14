"""Run an experiment."""
import os
import sys
from pathlib import Path

import wandb

# don't need docstrings in scripts
# pylint: disable=missing-function-docstring,missing-return-doc,missing-return-type-doc
YOUR_AGENT0_INSTALL_FOLDER = "elfpy"

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
        # sys.path.append(str(Path(__file__).parent / YOUR_AGENT0_INSTALL_FOLDER / "lib"))
        print(f"{sys.path=}")
        import examples.interactive_econ  # pylint: disable=import-outside-toplevel,unused-import


if __name__ == "__main__":
    run_experiment()
