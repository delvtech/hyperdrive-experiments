# Sweeps experiment harness

## Endpoints

### run_one_experiment

You can test run experiments with `run_one_experiment.py`, which has `wandb` disabled by default. You can enable `wandb` using a flag: `run_one_experiment.py --experiment=random_trades --wandb`.

### run_sweeps

Run sweeps with `run_sweeps.py`. This can receive a `--sweep_id` argument to coordinate sweeps across different people and machines.

### repro_experiment

You can reproduce any wandb run exactly with `repro_experiment.py`, which requires an `entity`, `project`, and `run-id`.

## Experiment guidelines

Write new experiments in the `experiments` folder. Experiments should be encapsulated and minimally branching, so that they are easy to parse for someone that lacks any additional context.