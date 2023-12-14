"""Parse the experiments folder."""
# %%
from copy import copy
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from dotenv import dotenv_values
from matplotlib import pyplot as plt
from matplotlib import ticker
from scipy.interpolate import griddata

FLOAT_FMT = ",.0f"

# %%
# do shit
EXPERIMENT_FOLDER = Path("experiments")

df1 = pd.read_parquet("agg_results1.parquet") if Path("agg_results1.parquet").exists() else pd.DataFrame()
df2 = pd.read_parquet("agg_results2.parquet") if Path("agg_results2.parquet").exists() else pd.DataFrame()
# for each folder in the experiments folder
for folder in EXPERIMENT_FOLDER.iterdir():
    if folder.is_dir():
        # extract the experiment id: experiments/exp_0
        experiment_id = int(folder.name.split("_")[1])
        print(f"Experiment ID {experiment_id}")
        # check if parameters.env exists and it has a non-zero size
        parameters = folder / "parameters.env"
        if parameters.exists() and parameters.stat().st_size > 0:
            # load it
            params = dotenv_values(parameters)
        if df1.shape[0] > 0 and experiment_id in df1["experiment"].values:
            print(f"Experiment ID {experiment_id} already in agg_results1.parquet")
        else:
            # check if agg_results1.parquet exists and it has a non-zero size
            file1 = folder / "results1.parquet"
            if file1.exists() and file1.stat().st_size > 0:
                # load it
                print(f"loading results from {file1} for experiment {experiment_id}")
                df_new = pd.read_parquet(file1)
                # add the experiment id and parameters
                df_new["experiment"] = experiment_id
                for param, value in params.items():
                    df_new[param] = value
                # append it
                df1 = pd.concat([df1, df_new], ignore_index=True)
        if df2.shape[0] > 0 and experiment_id in df2["experiment"].values:
            print(f"Experiment ID {experiment_id} already in agg_results2.parquet")
        else:
            # check if agg_results2.parquet exists and it has a non-zero size
            file2 = folder / "results2.parquet"
            if file2.exists() and file2.stat().st_size > 0:
                # load it
                print(f"loading results from {file2} for experiment {experiment_id}")
                df_new = pd.read_parquet(file2)
                # add the experiment id and parameters
                df_new["experiment"] = experiment_id
                for param, value in params.items():
                    df_new[param] = value
                # append it
                df2 = pd.concat([df2, df_new], ignore_index=True)
if "AGENT0_INSTALL_FOLDER" in df2.columns:
    df2 = df2.drop(columns=["AGENT0_INSTALL_FOLDER"])
df1.to_parquet("agg_results1.parquet")
df2.to_parquet("agg_results2.parquet")

# %%
# ensure min and max are equal to average
grpd = df2.loc[:, ["experiment", "block_number"]].groupby("experiment").count()
display(grpd)
assert grpd.min().values[0] == grpd.max().values[0] == grpd.mean().values[0]

# %%
cols = df2.columns
# keep only columns after "experiment"
cols = cols[cols.get_loc("experiment") + 1 :]
# convert to float
df2[cols] = df2[cols].astype(float)
# summarize
df2.loc[:, cols].describe()
idx = df2.username == "larry"
last_share_price = (df2.username == "share price") & (df2.experiment == df2.experiment.max())

# %%
# linearly regress pnl on CURVE_FEE and DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY
model = smf.ols("apr ~ CURVE_FEE + DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY", data=df2.loc[idx, :]).fit()
display(model.summary())

# Print the formula with coefficients
intercept = model.params.Intercept
coef_curve_fee = model.params.CURVE_FEE
coef_daily_volume = model.params.DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY

formula = f"apr = {intercept:,.0f} {'+' if coef_curve_fee > 0 else '-'} {abs(coef_curve_fee):,.3f} * CURVE_FEE\n{'+' if coef_daily_volume > 0 else '-'} {abs(coef_daily_volume):,.5f} * DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY"
print(formula)

# %%
# fancier model
model = smf.ols(
    "apr ~ CURVE_FEE + DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY + I(CURVE_FEE*DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY)",
    data=df2.loc[idx, :],
).fit()
model.summary()

# %%
# surf plot

# Extracting data from the dataframe
x = df2.loc[idx, "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY"].values
y = df2.loc[idx, "CURVE_FEE"].values
z = df2.loc[idx, "apr"].values

# Creating a meshgrid
xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolating z values
zi = griddata((x, y), z, (xi, yi), method="linear")

# Check if zi is a tuple
if isinstance(zi, tuple):
    raise ValueError("zi should not be a tuple.")

# Creating the plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(221, projection="3d")
surface = ax.plot_surface(xi, yi, zi, cmap="viridis")
# Rotate 90 degrees and re-plot
for i in range(1, 4):
    ax = fig.add_subplot(2, 2, i + 1, projection="3d")
    surface = ax.plot_surface(xi, yi, zi, cmap="viridis")
    ax.set_title(f"Rotation {i} (90 degrees)")
    ax.view_init(30, 30 + i * 90)  # Rotate by 90 degrees each time

plt.tight_layout()
plt.show()

# %%
# Best angle
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
surface = ax.plot_surface(xi, yi, zi, cmap="viridis", label="apr")
ax.view_init(30, 60 + 180)

# Adding labels and title
ax.set_xlabel("DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY")
ax.set_ylabel("CURVE_FEE")
ax.set_zlabel("apr")

ax.set_title(f"n={df2.loc[idx,:].shape[0]} rsq={model.rsquared:,.2f}\n{formula}")

breakeven_value = df2.loc[df2.username == "share price", "apr"].values.mean()
# Ensuring breakeven_value is a single value, not an array
if isinstance(breakeven_value, np.ndarray):
    breakeven_value = breakeven_value.item()
# draw a transparent horizontal plane at the breakeven value
ax.plot_surface(
    xi,
    yi,
    np.full_like(zi, breakeven_value),
    alpha=0.5,
    label="Breakeven",
    color="green",
)

# Custom legend
apr_patch = mpatches.Patch(color="blue", label="apr")
breakeven_patch = mpatches.Patch(color="green", label="Breakeven")
ax.legend(handles=[apr_patch, breakeven_patch])

# Display the plot
# plt.tight_layout()
plt.show()

# %%
# big table
display(
    df2.loc[idx | last_share_price, :]
    .rename(columns={"DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY": "volume_per_day"})
    .sort_values("apr", ascending=False)
    .style.format(
        subset=["apr", "position", "total_volume"],
        formatter="{:" + FLOAT_FMT + "}",
    )
    .hide(axis="index")
    .format(
        subset=["hpr", "apr"],
        formatter="{:.2%}",
    )
    .hide(
        axis="columns",
        subset=[
            "block_number",
            "TERM_DAYS",
            "AMOUNT_OF_LIQUIDITY",
            "FIXED_RATE",
            "GOVERNANCE_FEE",
            "RANDSEED",
            "DB_PORT",
            "CHAIN_PORT",
            "FLAT_FEE",
        ],
    )
)
df2.loc[idx | last_share_price, :].to_csv("bigtable.csv", index=False)

# %%
# how many of each combination do we have?
matrix = df2.loc[idx, ["CURVE_FEE", "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY", "apr"]].pivot_table(
    index="CURVE_FEE", columns="DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY", values=["apr"], aggfunc="count"
)
matrix_formatted = matrix.copy()
matrix_formatted.columns = matrix.columns.map(lambda x: f"volume={x[1]:,.0%}")
matrix_formatted.index = matrix.index.map(lambda x: f"{x:,.1%}")
# matrix_formatted = matrix_formatted.applymap(lambda x: f"{x:,.2%}")
matrix_formatted = matrix_formatted.reset_index(drop=False)
display(matrix_formatted.style.hide(axis="index"))

# %%
print("range in total_volume as a fraction of the mean for each combination:")
matrix = df2.loc[idx, ["CURVE_FEE", "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY", "total_volume"]].pivot_table(
    index="CURVE_FEE",
    columns="DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY",
    values=["total_volume"],
    aggfunc=["min", "max", "mean"],
)
# normalize min and max in terms of mean
matrix[("min", "total_volume")] = matrix[("min", "total_volume")] / matrix[("mean", "total_volume")]
matrix[("max", "total_volume")] = matrix[("max", "total_volume")] / matrix[("mean", "total_volume")]
# replace values with a string representing normalized_min-normalized_max
normalized_min = matrix[("min", "total_volume")]
normalized_max = matrix[("max", "total_volume")]
matrix_formatted = pd.DataFrame(index=matrix.index, columns=normalized_min.columns)
# Iterate through each column
for col in matrix[("mean", "total_volume")].columns:
    matrix_formatted[col] = (
        matrix[("min", "total_volume")][col].apply(lambda x: f"{x:.3f}")
        + "-"
        + matrix[("max", "total_volume")][col].apply(lambda x: f"{x:.3f}")
    )
# Now matrix_formatted contains the string representation of normalized_min-normalized_max
display(matrix_formatted)

# %%
# little matrix
matrix = df2.loc[idx, ["CURVE_FEE", "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY", "apr"]].pivot_table(
    index="CURVE_FEE", columns="DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY", values=["apr"], aggfunc="mean"
)
print("lil matrix, apr is the value in the middle")
matrix_formatted = matrix.copy()
matrix_formatted.columns = matrix.columns.map(lambda x: f"volume={x[1]:,.0%}")
matrix_formatted.index = matrix.index.map(lambda x: f"{x:,.1%}")
matrix_formatted = matrix_formatted.applymap(lambda x: f"{x:,.2%}")
matrix_formatted = matrix_formatted.reset_index(drop=False)
display(matrix_formatted.style.hide(axis="index"))

# %%
# df2.loc[idx, :].plot(
#     x="CURVE_FEE",
#     y="apr",
#     kind="scatter",
# )
for vol in np.sort(df2.loc[idx, "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY"].unique()):
    df2temp = copy(df2.loc[idx, :])
    subidx = df2temp.DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY == vol
    p = plt.scatter(
        df2temp.loc[subidx, "CURVE_FEE"],
        df2temp.loc[subidx, "apr"],
        label=f"volume={vol:,.0%}",
        alpha=0.5,
    )
    # set edgecolors to facecolor then facecolor to none
    p.set_edgecolor(p.get_facecolor())
    p.set_facecolor("none")
    df2temp["DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY"] = vol
    # line = model.predict(df2temp)
    # plt.plot(
    #     df2temp["CURVE_FEE"],
    #     line,
    #     label=f"DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY={vol:,.0%}",
    # )
# set y-axis format
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
plt.legend()
plt.show()

# df2.loc[idx, :].plot(
#     x="DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY",
#     y="apr",
#     kind="scatter",
# )
for fee in np.sort(df2.loc[idx, "CURVE_FEE"].unique()):
    df2temp = copy(df2.loc[idx, :])
    subidx = df2temp.CURVE_FEE == fee
    p = plt.scatter(
        df2temp.loc[subidx, "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY"],
        df2temp.loc[subidx, "apr"],
        label=f"fee={fee:.1%}",
        alpha=0.5,
    )
    # set edgecolors to facecolor then facecolor to none
    p.set_edgecolor(p.get_facecolor())
    p.set_facecolor("none")
    df2temp["CURVE_FEE"] = fee
    # line = model.predict(df2temp)
    # plt.plot(df2temp["DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY"], line, label=f"fee={fee:.1%}")
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
plt.legend()
plt.show()

# %%
