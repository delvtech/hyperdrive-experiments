"""Create a sweep configuration for Weights and Biases (wandb)."""
import pprint
import sys

import wandb

sweep_config = {
    "method": "random",
    "parameters": {
        "term_days": {"values": [365]},
        "amount_of_liquidity": {"values": [10_000_000]},
        "fixed_rate": {"values": [0.035]},
        # "daily_volume_percentage_of_liquidity": {
        #     "distribution": "uniform",
        #     "min": 0.01,
        #     "max": 0.1,
        # },
        "daily_volume_percentage_of_liquidity": {"values": [0.01, 0.05, 0.1]},
        # "curve_fee": {
        #     "distribution": "uniform",
        #     "min": 0.001,
        #     "max": 0.1,
        # },
        "curve_fee": {"values": [0.001, 0.005, 0.01]},
        "flat_fee": {"values": [0.0001]},
        "governance_fee": {"values": [0.1]},
        "randseed": {"distribution": "int_uniform", "min": 0, "max": sys.maxsize},
    },
    "program": "run_experiment.py",
    "project": "econ-experiment-one",
    "entity": "delvtech",
}

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project="econ-experiment-one", entity="delvtech")
print(f"{sweep_id=}")
