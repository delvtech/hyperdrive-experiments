"""Simulate an LP return.

We simulate an LP position in a pool with random trades that are profitable half the time.
The average daily trading volume is 10% of the total pool liquidity.
The variable rate is chosen to have an average of 4.5% return, about equal to 2023 staking returns.

Goal: Demonstrate that LPs backing trades can result in negative returns,
which is mitigated by profits from the yield source and fees.
"""
# %%
from __future__ import annotations

import wandb

from .run_experiment import run_lp_pnl_experiment

# %%
# Login
wandb.login()

# %%
# Experiment details
experiment_params = {"fixed_rate": 0.05, "curve_fee": 0.01}

## Initialize sweep config
sweep_config = {"method": "random"}

## Set goals
metric = {"name": "pnl", "goal": "maximize"}
sweep_config["metric"] = metric

## What to sweep over
grid_parameters = {"fixed_rate": {"values": [0, 0.01, 0.1]}}

random_parameters = {
    "variable_rate": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.05,
    },
    "curve_fee": {
        "distribution": "normal",
        "mu": 0.01,
        "sigma": 0.001,
    },
}

constant_parameters = {"position_duration": {"value": 60 * 60 * 24 * 365}}

parameters_dict = {}
parameters_dict.update(grid_parameters)
parameters_dict.update(random_parameters)
parameters_dict.update(constant_parameters)

sweep_config["parameters"] = parameters_dict

# %%
# Setup & run sweep
sweep_id = wandb.sweep(sweep_config, project=experiment_name)

wandb.agent(sweep_id, run_lp_pnl_experiment, count=500)

wandb.finish()
