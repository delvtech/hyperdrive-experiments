#!/bin/bash

while true; do
  ./run_experiment.sh

  # Check the exit status of run_experiment.sh
  if [ $? -eq 1 ]; then
    echo "Exiting the loop as run_experiment.sh finished its execution."
    break
  fi

  sleep 1
done
