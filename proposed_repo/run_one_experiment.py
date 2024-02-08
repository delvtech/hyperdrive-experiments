"""Run a single experiment once."""

from __future__ import annotations

import argparse
import importlib
import os
from dataclasses import asdict

import experiments
import wandb
from fixedpointmath import FixedPoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one experiment.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="random_trades",
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="If set, will use wandb.",
    )
    args = parser.parse_args()
    experiment = args.experiment
    use_wandb = args.wandb

    # Experiment details
    exp_config = experiments.Config(
        experiment_days=10,
        position_duration=60 * 60 * 24 * 7,  # 1 week
        initial_liquidity=FixedPoint(1_000_000),
        daily_volume_percentage_of_liquidity=FixedPoint("0.1"),
        agent_budget=FixedPoint(1_000_000),
        initial_variable_rate=FixedPoint("0.01"),
        initial_fixed_rate=FixedPoint("0.05"),
        curve_fee=FixedPoint("0.001"),
        flat_fee=FixedPoint("0.001"),
        experiment=experiment,
        num_agents=2,
    )

    if use_wandb:
        # Login to wandb
        wandb.login()
    else:
        os.environ["WANDB_SILENT"] = "true"
        exp_config.wandb_init_mode = "disabled"

    # Run the experiment
    # TODO: Remove the proposed_repo path once we adopt this fully
    experiment_module = importlib.import_module(name="proposed_repo.experiments." + experiment)
    getattr(experiment_module, experiment)(asdict(exp_config))

    if use_wandb:
        wandb.finish()
