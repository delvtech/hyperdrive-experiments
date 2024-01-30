#!/bin/bash
docker volume prune -f
rm -rf ~/.foundry/anvil/tmp
export NUM_PYTHONS=$(pgrep -f "python ../../interactive_econ.py" | wc -l)
export ANVILS=$(pgrep anvil | wc -l)
echo "pythons=$NUM_PYTHONS anvils=$ANVILS"
python clean_anvil.py $NUM_PYTHONS
python monitor_experiment.py
