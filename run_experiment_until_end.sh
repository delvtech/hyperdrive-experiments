#!/bin/bash

while true; do
  python run_experiment.py

  # Check the exit status of run_experiment.py
  if [ $? -eq 0 ]; then
    echo "Exiting the loop as run_experiment.py finished its execution."
    break
  fi

  sleep 1
done
