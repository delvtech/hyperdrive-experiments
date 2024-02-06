"""Run a single experiment once."""
# %%
from __future__ import annotations

from experiment.lp_pnl import lp_pnl_experiment as my_experiment

import wandb

# TODO: Correct this syntax

# %%
# Login
wandb.login()

# %%
# Repro experiment
api = wandb.api

entity = "delv"
project = "lp-analysis"

runs = api.runs(path=f"{entity}/{project}")

config_to_repro = runs[0].config

my_experiment(config_to_repro)

wandb.finish()
