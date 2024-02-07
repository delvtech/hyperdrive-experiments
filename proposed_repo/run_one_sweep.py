"""Sweep over a given experiment."""
from __future__ import annotations

import argparse
import importlib
from copy import deepcopy
from typing import Callable

import experiments
from experiments import SWEEP_CONFIG

import wandb


def run_one_sweep(
    sweep_id: str = "",
    experiment_fn: Callable | None = None,
    count: int = 1,
    sweep_config: dict | None = None,
    entity: str = "delvtech",
    project: str = "agent0_sweep",
):
    """Create a wandb sweep for a given experiment."""
    if experiment_fn is None:
        experiment_fn = experiments.random_trades

    if sweep_config is None:
        sweep_config = deepcopy(SWEEP_CONFIG)
        sweep_config["wandb_init_mode"] = "online"

    # Login (will be a noop if already logged in)
    wandb.login()

    # Setup sweep
    if sweep_id == "":
        sweep_id = wandb.sweep(sweep=sweep_config, entity=entity, project=project)

    # Spawn agent & run the experiment function
    wandb.agent(sweep_id=sweep_id, function=experiment_fn, entity=entity, project=project, count=count)

    # Cleanup
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a wandb sweep for the random experiment.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="random_trades",
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--sweepid",
        type=str,
        default="",
        help="Sweep ID from a wandb sweep controller.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="agent0_sweep",
        help="Name your project.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="delvtech",
        help="wandb entity for grouping projects.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="How many sweeps to do for this process.",
    )

    args = parser.parse_args()
    experiment_name = args.experiment
    sweep_id = args.sweepid
    project = args.project
    entity = args.entity
    count = args.count

    # TODO: Remove the proposed_repo path once we adopt this fully
    experiment_module = importlib.import_module(name="proposed_repo.experiments." + experiment_name)
    experiment_fn = getattr(experiment_module, experiment_name)

    run_one_sweep(sweep_id=sweep_id, experiment_fn=experiment_fn, count=count, entity=entity, project=project)
