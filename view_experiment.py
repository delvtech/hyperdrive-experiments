"""View experiment results."""

# %%
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker

EXPERIMENT_ID = 1
FLOAT_FMT = ",.0f"

experiment_path = Path("/code/experiments/experiments") / f"exp_{EXPERIMENT_ID}"

results1 = pd.read_parquet(experiment_path / "results1.parquet")
results2 = pd.read_parquet(experiment_path / "results2.parquet")

# %%
if results1.shape[0] > 0:
    print("material non-WETH positions:")
    display(results1.style.hide(axis="index"))
else:
    print("no material non-WETH positions")
print("WETH positions:")
display(
    results2.style.format(
        subset=[col for col in results2.columns if results2.dtypes[col] == "float64" and col not in ["hpr", "apr"]],
        formatter="{:" + FLOAT_FMT + "}",
    )
    .hide(axis="index")
    .format(
        subset=["hpr", "apr"],
        formatter="{:.2%}",
    )
)

# %%
pool_info = pd.read_parquet(experiment_path / "pool_info.parquet")

# %%
pool_info.plot(
    x="timestamp",
    y="fixed_rate",
)
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
# get major ticks every 0.5%
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.005))
# plot horizontal line at 3.5%
plt.axhline(0.035, color="red")
plt.show()

# %%
