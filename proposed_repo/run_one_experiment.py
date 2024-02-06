"""Run a single experiment once."""
# %%
from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import experiments
from fixedpointmath import FixedPoint

import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs a loop to check Hyperdrive invariants at each block.")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="If set, will use wandb.",
    )
    use_wandb = parser.parse_args().wandb

    # Experiment details
    # exp_config = experiments.RandomConfig(
    exp_config = experiments.LpPnlConfig(
        # pool config
        fixed_rate=FixedPoint("0.05"),
        variable_rate=FixedPoint("0.01"),
        curve_fee=FixedPoint("0.001"),
        flat_fee=FixedPoint("0.001"),
        position_duration=60 * 60 * 24 * 7,  # 1 week
        experiment_days=10,
        num_agents=2,
        # trade config
        initial_liquidity=FixedPoint(1_000_000),
        daily_volume_percentage_of_liquidity=FixedPoint("0.1"),
        agent_budget=FixedPoint(1_000_000),
    )

    if use_wandb:
        # Login to wandb
        wandb.login()
    else:
        os.environ["WANDB_SILENT"] = "true"
        exp_config.wandb_init_mode = "disabled"

    # Run the experiment
    # experiments.random_experiment(asdict(exp_config))
    experiments.lp_pnl_experiment(asdict(exp_config))

    if use_wandb:
        wandb.finish()
