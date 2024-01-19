#!/bin/bash
# Base directory for experiments
EXPERIMENTS_DIR="./experiments"

# Create experiments directory if it doesn't exist
mkdir -p "$EXPERIMENTS_DIR"

# Sweep arrays
# FIXED_RATES=(0.02 0.025 0.03 0.035 0.04 0.045 0.05)
# VOLUME_VALUES=(0.01 0.05 0.1)
# CURVE_FEES=(0.001 0.005 0.01)
FIXED_RATES=(0.035)
VOLUME_VALUES=(0.01 0.05 0.1)
CURVE_FEES=(0.001 0.005 0.01)
FLAT_FEES=(0.001 0.005 0.01)
RUN_FOREVER=true

total_permutations=$(( ${#FIXED_RATES[@]} * ${#VOLUME_VALUES[@]} * ${#CURVE_FEES[@]} ))
echo "total_permutations: $total_permutations"

# If there's an input param, run that experiment
if [ -n "$1" ]; then
    NEXT_EXPERIMENT_ID=$1
else
    # Determine the next experiment ID
    NEXT_EXPERIMENT_ID=$(find "$EXPERIMENTS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
fi

# Check if the index is within the range, otherwise end
if [ $NEXT_EXPERIMENT_ID -ge $total_permutations ]; then
    if [ "$RUN_FOREVER" = true ]; then
        echo "Starting next epoch."
    else
        echo "Run finished."
        exit 1
    fi
fi

# Create a new directory for this experiment
EXPERIMENT_DIR="$EXPERIMENTS_DIR/exp_$NEXT_EXPERIMENT_ID"
mkdir -p "$EXPERIMENT_DIR"

# File to store environment variables
ENV_FILE="$EXPERIMENT_DIR/parameters.env"

# Write fixed environment variables to the file
echo "TERM_DAYS=365" >> "$ENV_FILE"
echo "AMOUNT_OF_LIQUIDITY=10000000" >> "$ENV_FILE"
echo "FLAT_FEE=0.0001" >> "$ENV_FILE"
echo "GOVERNANCE_FEE=0.1" >> "$ENV_FILE"
# Calculate which epoch we're on
EPOCH=$(( NEXT_EXPERIMENT_ID / total_permutations ))
# set random seed equal to EPOCH
echo "RANDSEED=$((EPOCH))" >> "$ENV_FILE"
# echo "RANDSEED=0" >> "$ENV_FILE"
# add experiment_id to base port values
echo "DB_PORT=$((11000+$NEXT_EXPERIMENT_ID))" >> "$ENV_FILE"
echo "CHAIN_PORT=$((10000+$NEXT_EXPERIMENT_ID))" >> "$ENV_FILE"

# Calculate indices for each array using modulo for cycling through permutations
cycle_index=$(( NEXT_EXPERIMENT_ID % total_permutations ))
# Calculate indices for each array
rate_index=$(( cycle_index / (${#VOLUME_VALUES[@]} * ${#CURVE_FEES[@]}) ))
volume_index=$(( (cycle_index / ${#CURVE_FEES[@]}) % ${#VOLUME_VALUES[@]} ))
curve_fee_index=$(( cycle_index % ${#CURVE_FEES[@]} ))
# Get values from arrays
FIXED_RATE=${FIXED_RATES[$rate_index]}
DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY=${VOLUME_VALUES[$volume_index]}
CURVE_FEE=${CURVE_FEES[$curve_fee_index]}
# Write to the environment file
echo "FIXED_RATE=$FIXED_RATE" >> "$ENV_FILE"
echo "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY=$DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY" >> "$ENV_FILE"
echo "CURVE_FEE=$CURVE_FEE" >> "$ENV_FILE"

# Print the next experiment ID
echo "Next experiment ID=$NEXT_EXPERIMENT_ID on EPOCH=$EPOCH"
# Print the values for the next experiment
echo "FIXED_RATE=$FIXED_RATE"
echo "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY=$DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY"
echo "CURVE_FEE=$CURVE_FEE"

echo "ID $NEXT_EXPERIMENT_ID: Rate = $FIXED_RATE, Volume = $DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY, Fee = $CURVE_FEE"
# Run the experiment script within the experiment directory
# Set the path to the Python interpreter
PYTHON_BIN="python"
if [ -d ".venv" ]; then
    PYTHON_BIN="../../.venv/bin/python"
fi

# Run the experiment script within the experiment directory
(cd "$EXPERIMENT_DIR" && source parameters.env && echo "Experiment ID: $NEXT_EXPERIMENT_ID" && $PYTHON_BIN ../../interactive_econ.py)
