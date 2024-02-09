"""Function for converting a wandb run config spec to an experiment Config object."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..config import Config


def convert_run_config(run_config: dict[str, Any], skip_keys: list[str] = ["rng"]) -> Config:
    """Convert a wandb run config dictionary into an experiment Config object."""
    skip_keys = ["rng"]
    exp_config = Config()
    for key, value in run_config.items():
        if key not in skip_keys and hasattr(exp_config, key):
            exp_type = type(asdict(exp_config)[key]) if value is not None else lambda x: None
            if getattr(exp_config, key) != exp_type(value):
                setattr(exp_config, key, exp_type(value))
    return exp_config
