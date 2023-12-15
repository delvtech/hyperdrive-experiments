"""View experiment results."""
# %%
from pathlib import Path

import pandas as pd

EXPERIMENT_ID = 0
FLOAT_FMT = ",.0f"

experiment_path = Path("experiments") / f"exp_{EXPERIMENT_ID}"
print(f"{experiment_path=}")

results1 = pd.read_csv(experiment_path / "results1.csv")
results2 = pd.read_csv(experiment_path / "results2.csv")

# %%
if results1.shape[0] > 0:
    print("material non-WETH positions:")
    display(results1.style.hide(axis="index"))
else:
    print("no material non-WETH positions")
print("WETH positions:")
display(
    results2.style.format(
        subset=[col for col in results1.columns if results1.dtypes[col] == "float64" and col not in ["hpr", "apr"]],
        formatter="{:" + FLOAT_FMT + "}",
    )
    .hide(axis="index")
    .format(
        subset=["hpr", "apr"],
        formatter="{:.2%}",
    )
    .hide(axis="columns", subset=["base_token_type", "maturity_time"])
)

# %%
