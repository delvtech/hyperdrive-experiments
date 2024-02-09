"""Load config from a previous wandb experiment."""

from __future__ import annotations

from typing import Any

import yaml

import wandb


def load_config(entity: str, project: str, run_id: str) -> dict[str, Any]:
    """Load the config dict from a wandb run."""
    # login
    wandb.login()
    api = wandb.Api()
    # You can also get the run path from the run's overview page
    run_path = f"{entity}/{project}/{run_id}"
    run = api.run(path=run_path)
    config_stream = run.file("config.yaml").download(replace=True)
    config = yaml.load(stream=config_stream, Loader=yaml.FullLoader)
    config.pop("_wandb")
    for key, value in config.items():
        if isinstance(value, dict) and "value" in value.keys():
            config[key] = value["value"]
    return config
