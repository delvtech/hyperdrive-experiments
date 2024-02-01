"""Sweep over a given experiment."""
# %%
from __future__ import annotations

import wandb

from .lp_pnl_experiment import lp_pnl_experiment

# %%
# Login
wandb.login()

# %%
## Initialize sweep config
sweep_config = {"method": "random"}

## What to sweep over
grid_parameters = {"fixed_rate": {"values": [0, 0.01, 0.1]}}

random_parameters = {
    "flat_fee": {
        "distribution": "normal",
        "mu": 0.01,
        "sigma": 0.001,
    },
    "curve_fee": {
        "distribution": "normal",
        "mu": 0.01,
        "sigma": 0.001,
    },
    "governance_fee": {
        "distribution": "normal",
        "mu": 0.01,
        "sigma": 0.001,
    },
}

constant_parameters = {
    "position_duration": {"value": 60 * 60 * 24 * 182},
    "variable_rate": {"value": 0.045},
}

parameters_dict = {}
parameters_dict.update(grid_parameters)
parameters_dict.update(random_parameters)
parameters_dict.update(constant_parameters)

sweep_config["parameters"] = parameters_dict

## Set goals
metric = {"name": "pnl", "goal": "maximize"}
sweep_config["metric"] = metric

# %%
# Setup & run sweep
sweep_id = wandb.sweep(sweep_config, project="lp pnl sweep")

wandb.agent(sweep_id, lp_pnl_experiment, count=500)

wandb.finish()
