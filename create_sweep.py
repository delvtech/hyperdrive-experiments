"""Experiment one.

Experiment 1:

Use a random agent (rob) and a smart agent (sally) to trade X% of the TVL per day and plot how profitable it is for LP (larry).
Trial 1:
3.5% initial market rate
variable rate between 3 to 4%
1 year term
Fees: Flat 1bps, Curve try with 1%
1% of TVL traded per day
Trial 2:
3.5% initial market rate
variable rate between 3 to 4%
1 year term
Fees: Flat 1bps, Curve try with 1%
10% of TVL traded per day
Trial 3:
3.5% initial market rate
variable rate between 3 to 4%
1 year term
Fees: Flat 1bps, Curve try with .1%
1% of TVL traded per day
Trial 4:
3.5% initial market rate
variable rate between 3 to 4%
1 year term
Fees: Flat 1bps, Curve try with .1%
10% of TVL traded per day
"""
import sys

import wandb
import pprint

sweep_config = {
    "method": "random",
    "parameters": {
        "term_days": {"values": [20]},
        "amount_of_liquidity": {"values": [10_000_000]},
        "fixed_rate": {"values": [0.035]},
        "daily_volume_percentage_of_liquidity": {
            "distribution": "uniform",
            "min": 0.01,
            "max": 0.1,
        },
        "curve_fee": {
            "distribution": "uniform",
            "min": 0.001,
            "max": 0.1,
        },
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
