"""Run a single experiment once."""
# %%
from __future__ import annotations

import argparse
import os
from dataclasses import asdict

from experiment.configs import LpPnlConfig
from experiment.lp_pnl import lp_pnl_experiment as my_experiment
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
    # 200M initial liquidity; 10% = 20M trade volume per day
    # 2 agents, 30 days of trading = 10M trades per agent
    exp_config = LpPnlConfig(
        # pool config
        fixed_rate=FixedPoint("0.05"),
        variable_rate=FixedPoint("0.05"),
        curve_fee=FixedPoint("0.01"),
        flat_fee=FixedPoint("0.01"),
        position_duration=60 * 60 * 24 * 7,  # 1 week
        experiment_days=30,  # 1 month
        num_agents=2,
        # trade config
        initial_liquidity=FixedPoint(100_000_000),
        daily_volume_percentage_of_liquidity=FixedPoint("0.10"),
        opens_per_day=FixedPoint(5),
        minimum_trade_hold_days=FixedPoint(3),
        agent_budget=FixedPoint(100_000_000),
    )

    if use_wandb:
        # Login to wandb
        wandb.login()
    else:
        os.environ["WANDB_SILENT"] = "true"
        exp_config.wandb_init_mode = "disabled"

    # Run the experiment
    my_experiment(asdict(exp_config))

    if use_wandb:
        wandb.finish()
