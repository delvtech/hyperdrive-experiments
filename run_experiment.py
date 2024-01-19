# %%
import itertools
import os
import subprocess
import sys

import numpy as np
import pandas as pd

# pylint: disable=redefined-outer-name,invalid_name

def running_interactive():
    try:
        from IPython.core.getipython import get_ipython  # pylint: disable=import-outside-toplevel

        return bool("ipykernel" in sys.modules and get_ipython())
    except ImportError:
        return False

# %%
EXPERIMENTS_DIR = "./runs"
EPOCHS = 15
RUNS_TABLE_FILE = "runs_table.csv"

# "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY": {"name": "linspace", "min": 0.01, "max": 0.1},
# if runs_Table.csv doesn't exist
if not os.path.exists(RUNS_TABLE_FILE):
    # Define your parameters
    parameters = {
        "FIXED_RATE": [0.035],
        "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY": list(np.arange(0.01, 0.11, 0.01)),
        "CURVE_FEE": [0.001, 0.005, 0.01],  # 0.1% to 1
        "FLAT_FEE": [0.0001, 0.0005, 0.001]  # 1 to 10 bps
    }

    # Generate combinations for fixed parameters
    nonrandom_params = {k: v for k, v in parameters.items() if isinstance(v, list)}
    nonrandom_combinations = np.array(list(itertools.product(*[nonrandom_params[key] for key in nonrandom_params])))

    # Tile nonrandom combinations for each epoch
    nonrandom_combinations_tiled = np.tile(nonrandom_combinations, (EPOCHS, 1))

    runs_table = pd.DataFrame(
        data=nonrandom_combinations_tiled,
        columns=nonrandom_params.keys(),
        index=pd.Index(
            data=range(len(nonrandom_combinations_tiled)),
            name="experiment_id"
            )
        )
    # Create the EPOCH column
    runs_table["EPOCH"] = np.repeat(np.arange(EPOCHS), len(nonrandom_combinations))

    random_params = {k: v for k, v in parameters.items() if not isinstance(v, list)}
    # Create a vector of length equal to runs_table sampling from each distribution
    for param, distribution in random_params.items():
        # One random value per combination per epoch
        if distribution["name"] == "uniform":
            rand_num = np.random.uniform(low=distribution["min"], high=distribution["max"], size=EPOCHS)
        elif distribution["name"] == "int_uniform":
            rand_num = np.random.randint(low=distribution["min"], high=distribution["max"] + 1, size=EPOCHS)
        elif distribution["name"] == "linspace":
            rand_num = np.linspace(distribution["min"], distribution["max"], EPOCHS)

        runs_table[param] = np.repeat(rand_num, len(nonrandom_combinations))

    # write runs_table.csv
    runs_table.to_csv(RUNS_TABLE_FILE, index=False)
else:
    runs_table = pd.read_csv(RUNS_TABLE_FILE)

if running_interactive():
    display(runs_table)

# %%
# Check if an argument is passed to the script
if len(sys.argv) > 1:
    try:
        NEXT_EXPERIMENT_ID = int(sys.argv[1])
    except ValueError as exc:
        raise ValueError("Provided experiment ID is not a valid integer.") from exc
else:
    if not os.path.exists(EXPERIMENTS_DIR):
        os.mkdir(EXPERIMENTS_DIR)
    # Count the number of directories in EXPERIMENTS_DIR
    NEXT_EXPERIMENT_ID = sum(os.path.isdir(os.path.join(EXPERIMENTS_DIR, d)) for d in os.listdir(EXPERIMENTS_DIR))
print(f"Next Experiment ID: {NEXT_EXPERIMENT_ID}")
if NEXT_EXPERIMENT_ID > runs_table.index[-1]:
    # exit with code 0 if all experiments have been run
    print("All experiments have been run.")
    sys.exit(0)
EXPERIMENT_DIR = os.path.join(EXPERIMENTS_DIR, str(NEXT_EXPERIMENT_ID))
if not os.path.exists(EXPERIMENT_DIR):
    os.mkdir(EXPERIMENT_DIR)
print(f"Experiment Directory: {EXPERIMENT_DIR}")
PARAM_FILE = os.path.join(EXPERIMENT_DIR, "parameters.env")
if not os.path.exists(PARAM_FILE):
    with open(PARAM_FILE, "w") as env_file:
        for param in runs_table.columns:
            param_name = "RANDSEED" if param == "EPOCH" else param
            env_file.write(f"{param_name} = {runs_table[param][NEXT_EXPERIMENT_ID]}\n")
        env_file.write(f"EXPERIMENT_ID = {NEXT_EXPERIMENT_ID}\n")
        env_file.write(f"TERM_DAYS = {365}\n")
        env_file.write(f"AMOUNT_OF_LIQUIDITY = {10000000}\n")
        env_file.write(f"GOVERNANCE_FEE = {0.1}\n")

# %%
# Run the experiment script within the experiment directory
PYTHON_BIN = "python" if not os.path.exists(".venv") else "../../.venv/bin/python"
os.chdir(EXPERIMENT_DIR)
subprocess.run([PYTHON_BIN, "../../interactive_econ.py"], check=True)
