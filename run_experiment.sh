#!/bin/bash

# Base directory for experiments
EXPERIMENTS_DIR="./experiments"

# Create experiments directory if it doesn't exist
mkdir -p "$EXPERIMENTS_DIR"

# Determine the next experiment ID
NEXT_EXPERIMENT_ID=$(find "$EXPERIMENTS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)

# Create a new directory for this experiment
EXPERIMENT_DIR="$EXPERIMENTS_DIR/exp_$NEXT_EXPERIMENT_ID"
mkdir -p "$EXPERIMENT_DIR"

# File to store environment variables
ENV_FILE="$EXPERIMENT_DIR/parameters.env"

# Write fixed environment variables to the file
echo "TERM_DAYS=365" >> "$ENV_FILE"
echo "AMOUNT_OF_LIQUIDITY=10000000" >> "$ENV_FILE"
echo "FIXED_RATE=0.035" >> "$ENV_FILE"
echo "FLAT_FEE=0.0001" >> "$ENV_FILE"
echo "GOVERNANCE_FEE=0.1" >> "$ENV_FILE"
echo "RANDSEED=$NEXT_EXPERIMENT_ID" >> "$ENV_FILE"
# add experiment_id to base port values
echo "DB_PORT=$((11000+$NEXT_EXPERIMENT_ID))" >> "$ENV_FILE"
echo "CHAIN_PORT=$((10000+$NEXT_EXPERIMENT_ID))" >> "$ENV_FILE"

# Generate random values within given ranges
# echo "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY=$(awk -v min=0.01 -v max=0.1 'BEGIN{srand(); print min+rand()*(max-min)}')" >> "$ENV_FILE"
# echo "CURVE_FEE=$(awk -v min=0.001 -v max=0.01 'BEGIN{srand(); print min+rand()*(max-min)}')" >> "$ENV_FILE"

# Generate random values from provided values
VOLUME_VALUES=(0.01 0.05 0.1)
echo "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY=${VOLUME_VALUES[$RANDOM % ${#VOLUME_VALUES[@]}]}" >> "$ENV_FILE"
CURVE_FEES=(0.001 0.005 0.01)
echo "CURVE_FEE=${CURVE_FEES[$RANDOM % ${#CURVE_FEES[@]}]}" >> "$ENV_FILE"


# Run the experiment script within the experiment directory
(cd "$EXPERIMENT_DIR" && source parameters.env && echo "Experiment ID: $NEXT_EXPERIMENT_ID" && python ../../interactive_econ.py)