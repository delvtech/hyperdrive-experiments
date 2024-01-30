"""Utilities to check the environment we're running in."""
import os
import sys

def running_interactive():
    try:
        from IPython.core.getipython import get_ipython  # pylint: disable=import-outside-toplevel

        return bool("ipykernel" in sys.modules and get_ipython())
    except ImportError:
        return False

def running_wandb():
    # Check for a specific wandb environment variable
    # For example, 'WANDB_RUN_ID' is set by wandb during a run
    return "WANDB_RUN_ID" in os.environ
