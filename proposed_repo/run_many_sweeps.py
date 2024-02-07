"""Run a bunch of sweeps on different subprocesses.

This example script could be extended to run sweeps in parallel on different CPUs, spot instances, etc.
"""

import argparse
from copy import deepcopy
from multiprocessing.pool import Pool

from experiments import SWEEP_CONFIG
from run_one_sweep import run_one_sweep

import wandb

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
        "--sweeps-per-process",
        type=int,
        default=1,
        help="How many sweeps to run in each process.",
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
    sweeps_per_process = args.sweeps_per_process
    experiment = args.experiment
    project = args.project
    entity = args.entity

    # get default config, which can be modified here
    sweep_config = deepcopy(SWEEP_CONFIG)

    # login to wandb
    wandb.login()

    # create a sweep ID
    sweep_id = wandb.sweep(sweep=sweep_config, entity=entity, project=project)

    # create a pool of processes which will carry out sweeps
    pool = Pool()
    for proc_id in range(num_processes):
        async_result = pool.apply_async(
            func=run_one_sweep,
            kwds={
                "sweep_id": sweep_id,
                "experiment": experiment,
                "count": sweeps_per_process,
                "sweep_config": sweep_config,
                "entity": entity,
                "project": project,
            },
        )

    # prevent any more tasks from being submitted to the pool
    pool.close()

    # wait for the worker processes to exit
    pool.join()
