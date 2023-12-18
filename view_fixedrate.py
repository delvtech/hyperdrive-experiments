"""View experiment results."""
# %%
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker

EXPERIMENT_ID = 44
FLOAT_FMT = ",.0f"

experiment_path = Path("/nvme/experiments/experiments") / f"exp_{EXPERIMENT_ID}"

results1 = pd.read_parquet(experiment_path / "results1.parquet")
results2 = pd.read_parquet(experiment_path / "results2.parquet")

pool_info = pd.read_parquet(experiment_path / "pool_info.parquet")

pool_info.plot(
    x="timestamp",
    y=["fixed_rate","variable_rate"],
)
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
# get major ticks every 0.5%
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.005))
# plot horizontal line at 3.5%
# plt.axhline(0.035, color="red")
plt.show()

# %%