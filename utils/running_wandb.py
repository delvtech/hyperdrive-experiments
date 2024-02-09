import os


def running_wandb():
    # Check for a specific wandb environment variable
    # For example, 'WANDB_RUN_ID' is set by wandb during a run
    return "WANDB_RUN_ID" in os.environ
