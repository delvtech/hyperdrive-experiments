"""Run a single experiment once."""
# %%
from __future__ import annotations

from dataclasses import asdict

from experiment.configs import LpPnlConfig
from experiment.lp_pnl import lp_pnl_experiment as my_experiment

import wandb

# %%
# Login
wandb.login()

# %%
# Experiment details
exp_config = LpPnlConfig(
    fixed_rate=0.05,
    variable_rate=0.05,
    curve_fee=0.01,
    flat_fee=0.01,
    position_duration=60 * 60 * 24 * 182,
)
# %%
# Run the experiment
my_experiment(asdict(exp_config))

wandb.finish()
