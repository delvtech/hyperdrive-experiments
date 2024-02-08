"""Run a bunch of sweeps on different subprocesses.

This example script could be extended to run sweeps in parallel on different CPUs, spot instances, etc.
"""

from __future__ import annotations

import argparse
import importlib
from copy import deepcopy
from multiprocessing.pool import Pool

from experiments import SWEEP_CONFIG

import wandb


def run_one_sweep(
    sweep_id: str = "",
    experiment: str = "",
    count: int = 1,
    sweep_config: dict | None = None,
    entity: str = "delvtech",
    project: str = "agent0_sweep",
):
    """Create a wandb sweep for a given experiment."""
    # set defaults
    if sweep_config is None:
        sweep_config = deepcopy(SWEEP_CONFIG)
        sweep_config["wandb_init_mode"] = "online"
    if experiment == "":
        experiment = "random_trades"

    # load experiment function object
    experiment_module = importlib.import_module(name="sweeps.experiments." + experiment)
    experiment_fn = getattr(experiment_module, experiment)
    sweep_config["experiment"] = experiment

    # login (will be a noop if already logged in)
    wandb.login()

    # setup sweep
    if sweep_id == "":
        sweep_id = wandb.sweep(sweep=sweep_config, entity=entity, project=project)

    # spawn agent & run the experiment function
    wandb.agent(sweep_id=sweep_id, function=experiment_fn, entity=entity, project=project, count=count)

    # cleanup
    wandb.finish()


if __name__ == "__main__":
    # parallel sweep parameters
    parser = argparse.ArgumentParser(description="Create a wandb sweep for the random experiment.")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="How many processes to spin up.",
    )
    parser.add_argument(
        "--count-per-process",
        type=int,
        default=1,
        help="How many sweep runs to run in each process.",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default="",
        help="Sweep ID from a wandb sweep controller.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="random_trades",
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="agent0_many_sweeps",
        help="Name your project.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="delvtech",
        help="wandb entity for grouping projects.",
    )

    args = parser.parse_args()
    num_processes = args.num_processes
    count_per_process = args.count_per_process
    experiment = args.experiment
    sweep_id = args.sweep_id
    project = args.project
    entity = args.entity

    # get default config, which can be modified here
    sweep_config = deepcopy(SWEEP_CONFIG)

    # login to wandb
    wandb.login()

    # create a sweep ID
    if sweep_id == "":
        sweep_id = wandb.sweep(sweep=sweep_config, entity=entity, project=project)

    # create a pool of processes which will carry out sweeps
    pool = Pool()
    for _ in range(num_processes):
        async_result = pool.apply_async(
            func=run_one_sweep,
            kwds={
                "sweep_id": sweep_id,
                "experiment": experiment,
                "count": count_per_process,
                "sweep_config": sweep_config,
                "entity": entity,
                "project": project,
            },
        )

    # prevent any more tasks from being submitted to the pool
    pool.close()

    # wait for the worker processes to exit
    pool.join()
