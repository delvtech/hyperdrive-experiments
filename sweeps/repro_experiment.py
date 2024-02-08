"""Reproduce an experiment from a wandb run ID."""

from __future__ import annotations

import argparse
import importlib
from dataclasses import asdict

from experiments.utils import convert_run_config, load_config

import wandb


def repro_experiment(entity, project, run_id):
    """Reproduce an experiment from the entity, project, and run_id."""
    # create experiment config object
    exp_config = convert_run_config(load_config(entity, project, run_id))
    exp_config.repro_run_id = run_id

    experiment_module = importlib.import_module(name="sweeps.experiments." + exp_config.experiment)
    getattr(experiment_module, exp_config.experiment)(asdict(exp_config))
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one experiment.")
    parser.add_argument(
        "--entity",
        type=str,
        default="delvtech",
        help="wandb entity for grouping projects.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="agent0_many_sweeps",
        help="Name your project.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="uxuuqui1",
        help="Run ID provided by wandb.",
    )
    args = parser.parse_args()
    repro_experiment(args.entity, args.project, args.run_id)
