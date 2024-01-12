"""Parse the experiments folder."""
# %%
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from dotenv import dotenv_values
from matplotlib import pyplot as plt
from matplotlib import ticker
from statsmodels.tools.tools import add_constant

import darkmode_orange

FLOAT_FMT = ",.0f"
short_variable_names = {
    "CURVE_FEE": "curve fee",
    "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY": "volume",
    "FIXED_RATE": "rate"
}

# %%
# do shit
EXPERIMENT_FOLDER = Path("results/exp_two")
DELETE_UNFINISHED_EXPERIMENTS = False
PARQUET_FILES = ["agg_results1.parquet", "agg_results2.parquet", "rate_paths.parquet"]
DELETE_PREVIOUS_PARQUET_FILES = True

if DELETE_PREVIOUS_PARQUET_FILES:
    for file in PARQUET_FILES:
        if Path(file).exists():
            Path(file).unlink()
df1 = pd.read_parquet("agg_results1.parquet") if Path("agg_results1.parquet").exists() else pd.DataFrame()
df2 = pd.read_parquet("agg_results2.parquet") if Path("agg_results2.parquet").exists() else pd.DataFrame()
rate_paths = pd.read_parquet("rate_paths.parquet") if Path("rate_paths.parquet").exists() else pd.DataFrame()
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
            elif DELETE_UNFINISHED_EXPERIMENTS:
                # delete the folder
                print(f"deleting folder {folder}")
                parameters.unlink()
                folder.rmdir()
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
        if rate_paths.shape[0] > 0 and experiment_id in rate_paths["experiment"].values:
            print(f"Experiment ID {experiment_id} already in rate_paths.parquet")
        else:
            # check if rate_paths.parquet exists and it has a non-zero size
            file3 = folder / "pool_info.parquet"
            if file3.exists() and file3.stat().st_size > 0:
                # load it
                print(f"loading results from {file3} for experiment {experiment_id}")
                df_new = pd.read_parquet(file3)
                # add the experiment id
                df_new["experiment"] = experiment_id
                # append it
                rate_paths = pd.concat([rate_paths, df_new], ignore_index=True)
if "AGENT0_INSTALL_FOLDER" in df2.columns:
    df2 = df2.drop(columns=["AGENT0_INSTALL_FOLDER"])
df1.to_parquet("agg_results1.parquet")
df2.to_parquet("agg_results2.parquet")

# %%
# ensure data looks correct
# min and max are equal to average
grpd = df2.loc[:, ["experiment", "block_number"]].groupby("experiment").count()
assert grpd.min().values[0] == grpd.max().values[0]

# IDs are continuous
missing_ids = []
for experiment_id in range(df2.experiment.max() + 1):
    if experiment_id not in df2.experiment:
        missing_ids.append(experiment_id)
if len(missing_ids) > 0:
    print(f"missing experiment IDS: {', '.join(map(str, missing_ids))}")

# %%
# update values
for experiment in df2.experiment.unique():
    idx1 = df1.experiment == experiment
    idx2 = df2.experiment == experiment
    larry_lp = df1.loc[idx1 & (df1.username == "larry"), "position"]
    if larry_lp.shape[0] > 0:
        lp_share_price = df2.loc[idx2 & (df2.username == "share price"), "position"].values[0]/10_000_000
        # print(f"{lp_share_price=}")
        if df2.loc[idx2 & (df2.username == "larry"), "position"].values[0] == 0:
            df2.loc[idx2 & (df2.username == "larry"), "position"] = larry_lp * lp_share_price
        df2.loc[idx2 & (df2.username == "larry"), "pnl"] = df2.loc[idx2 & (df2.username == "larry"), "position"] - 10_000_000
        df2.loc[idx2 & (df2.username == "larry"), "hpr"] = df2.loc[idx2 & (df2.username == "larry"), "pnl"] / df2.loc[idx2 & (df2.username == "larry"), "position"]
        df2.loc[idx2 & (df2.username == "larry"), "apr"] = df2.loc[idx2 & (df2.username == "larry"), "pnl"] / df2.loc[idx2 & (df2.username == "larry"), "position"]
        # display(df1.loc[idx1 & (df1.username == "larry"), :])

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
df2["CURVE_FEE_X_DAILY_VOLUME"] = df2.CURVE_FEE * df2.DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY * 1
df2["FIXED_V_VARIABLE"] = df2.FIXED_RATE - 0.035
df2["ABS_FIXED_V_VARIABLE"] = abs(df2.FIXED_RATE - 0.035)

# %%
# regression
vars = []
vars.append("CURVE_FEE_X_DAILY_VOLUME")
vars.append("ABS_FIXED_V_VARIABLE")
# vars.append("FIXED_V_VARIABLE")
# vars.append("FIXED_RATE")
model = smf.ols(f"apr ~ {' + '.join(vars)}", data=df2.loc[idx, :]).fit()
display(model.summary())

# Print the formula with coefficients
intercept = model.params.Intercept
coefs = model.params

formula = f"apr = {intercept:,.3f}"
for var, coef in zip(vars, coefs[1:]):
    formula += f" {'+' if coef > 0 else '-'} {coef:,.3f} * {var}"
print(formula)

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
# check every combination
# var_list = ["CURVE_FEE", "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY", "FIXED_RATE"]
var_list = ["CURVE_FEE", "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY"]
x_var = var_list[0]

print("count")
matrix_count = df2.loc[idx, var_list+["apr"]].pivot_table(
    index="CURVE_FEE", columns=[v for v in var_list if v != x_var], values=["apr"], aggfunc="count"
)
matrix_count.columns.names = [short_variable_names[v] if v else v for v in matrix_count.columns.names]
display(matrix_count)

# %%
print("return")
matrix = df2.loc[idx, var_list+["apr"]].pivot_table(
    index="CURVE_FEE", columns=[v for v in var_list if v != x_var], values=["apr"], aggfunc="mean"
)
matrix.columns = matrix.columns.map(lambda x: f"{short_variable_names[var_list[1]]}={x[1]:,.1%}")
# matrix.index = matrix.index.map(lambda x: f"{x:,.1%}")
matrix.index = matrix.index.map(lambda x: f"{short_variable_names['CURVE_FEE']}={x:,.1%}")
matrix.index.name = ''
matrix = matrix.applymap(lambda x: f"{x:,.2%}")
matrix.columns.names = [short_variable_names[v] if v else v for v in matrix.columns.names]
display(matrix)
# display(matrix.style
#     .format(
#         subset=["apr"],
#         formatter="{:.2%}",
#     )
# )

# %%
# range in trade volume
if matrix_count.max().max() > 1:
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
# parse variables
variables_to_check_for_variability = ["CURVE_FEE", "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY"]
most_variable = ''
least_variable = ''
for variable in variables_to_check_for_variability:
    if most_variable == '':
        most_variable = variable
    if least_variable == '':
        least_variable = variable
    if df2[variable].nunique() > df2[most_variable].nunique():
        most_variable = variable
    if df2[variable].nunique() < df2[least_variable].nunique():
        least_variable = variable

# %%
# little 1-d table
matrix = df2.loc[idx, [most_variable, "apr"]].pivot_table(
    columns=most_variable, values=["apr"], aggfunc="mean"
)
matrix_formatted = matrix.copy()
column_index = matrix.columns
column_index.name = ''
column_index = matrix.columns.map(lambda x: f"{short_variable_names[most_variable]}={x:,.1%}")
matrix_formatted.columns = column_index
matrix_formatted = matrix_formatted.applymap(lambda x: f"{x:,.2%}")
matrix_formatted.set_index(pd.Index(name="", data=["LP profitability"]), inplace=True)
display(matrix_formatted)

# %%
# little matrix
matrix = df2.loc[idx, [least_variable, most_variable, "apr"]].pivot_table(
    index=least_variable, columns=most_variable, values=["apr"], aggfunc="mean"
)
print("lil matrix, apr is the value in the middle")
matrix_formatted = matrix.copy()
matrix_formatted.columns = matrix.columns.map(lambda x: f"{short_variable_names[most_variable]}={x[1]:,.1%}")
matrix_formatted.index = matrix.index.map(lambda x: f"{x:,.1%}")
matrix_formatted = matrix_formatted.applymap(lambda x: f"{x:,.2%}")
matrix_formatted = matrix_formatted.reset_index(drop=False)
cols = list(matrix_formatted.columns)
cols[0] = short_variable_names[least_variable]
matrix_formatted.columns = cols
display(matrix_formatted.style.hide(axis="index"))

# %%
# plot APR against most_variable
plot_data = copy(df2.loc[idx, :]).sort_values(by=most_variable)
plt.scatter(plot_data.loc[:, most_variable],plot_data.loc[:, "apr"], label='LP return')
plt.xlabel(short_variable_names[most_variable])
plt.ylabel("Return (APR)")
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
ylim = plt.gca().get_ylim()
ylim = np.floor(ylim[0] * 1000) / 1000, np.ceil(ylim[1] * 1000) / 1000
plt.gca().set_ylim(ylim)
yticks = np.arange(min(ylim[0],0.034), ylim[1]+0.001, 0.001)  # 0.1% increment
plt.gca().set_yticks(yticks)
x_vars = plot_data.loc[:, most_variable].values
m, b = np.polyfit(x_vars, plot_data.loc[:, "apr"], 1)
x_vars_with_zero = np.append(x_vars,0)
y_fit = m * x_vars_with_zero + b
plt.plot(x_vars_with_zero, y_fit, color='orange')
# plot horizontal line
plt.axhline(0.035, color='red', label="Vault variable return", alpha=1, linestyle='--')
plt.title(f"LP Profitability vs. {short_variable_names[most_variable]}")
plt.legend()
plt.show()

# %%
for abs_fixed_v_variable in np.sort(df2.loc[idx, "ABS_FIXED_V_VARIABLE"].apply(lambda x: round(x, 3)).unique()):
    df2temp = copy(df2.loc[idx, :])
    subidx = df2temp.ABS_FIXED_V_VARIABLE.apply(lambda x: round(x, 3)) == abs_fixed_v_variable
    print(f"num={subidx.sum()}")
    p = plt.scatter(
        df2temp.loc[subidx, "CURVE_FEE_X_DAILY_VOLUME"],
        df2temp.loc[subidx, "apr"],
        label=f"abs_fixed_v_variable={abs_fixed_v_variable:.1%}",
        alpha=0.5,
    )
    color = p.get_facecolor()
    p.set_edgecolor(color)
    p.set_facecolor("none")
    predicted_values = model.predict(df2temp.loc[subidx, vars])
    # plt.plot(
    #     df2temp.loc[subidx, "CURVE_FEE_X_DAILY_VOLUME"],
    #     predicted_values,
    #     color=color,
    #     # label=f"regression fit ({abs_fixed_v_variable:.1%})",
    # )
    X_with_const = add_constant(df2temp.loc[subidx, "CURVE_FEE_X_DAILY_VOLUME"])
    temp_model = sm.OLS(df2temp.loc[subidx, "apr"], X_with_const).fit()
    predictions = temp_model.predict(X_with_const)
    plt.plot(df2temp.loc[subidx, "CURVE_FEE_X_DAILY_VOLUME"],predictions,color=color)
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
plt.xlabel("CURVE_FEE_X_DAILY_VOLUME")
plt.ylabel("apr")
plt.legend()
plt.show()

# %%
min_timestamp_by_experiment = rate_paths.groupby('experiment')['timestamp'].min()
rate_paths.loc[:,"adjusted_timestamp"] = rate_paths["timestamp"] - min_timestamp_by_experiment[rate_paths["experiment"]].values
rate_paths['adjusted_timestamp_seconds'] = rate_paths['adjusted_timestamp'].dt.total_seconds()
rate_paths['adjusted_timestamp_days'] = rate_paths['adjusted_timestamp_seconds'] / (60 * 60 * 24)

# %%
# plot rate paths
fig = plt.figure()
ax = plt.gca()
unique_experiments = rate_paths['experiment'].unique()
rate_volatility_records = []
for experiment in unique_experiments[:20]:
    idx = rate_paths['experiment'] == experiment
    rate_paths.loc[idx].plot(x='adjusted_timestamp_days', y='fixed_rate', label=experiment, alpha=0.1, ax=ax)
    rate_volatility = rate_paths.loc[idx, "fixed_rate"].std()
    volume = df1.loc[df1.experiment==experiment,"DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY"].iloc[0]
    fee = df1.loc[df1.experiment==experiment,"CURVE_FEE"].iloc[0]
    print(f"{rate_volatility=}")
    rate_volatility_records.append([experiment, rate_volatility, volume, fee])
# disable legend
ax.legend().set_visible(False)
# print x axis format
# print(ax.xaxis.get_major_formatter())
# set x axis format
# ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f} days"))
# Set the custom formatter for the x-axis
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
plt.xlabel('Time (days)')
plt.xticks(rotation=45)
plt.xticks(np.append(np.arange(0, 360, 30),365))
plt.ylabel('Fixed Rate')
plt.ylim(0, 0.06)
plt.show()

# %%
rate_volatility_df = pd.DataFrame(rate_volatility_records, columns=["experiment", "rate_volatility", "volume", "fee"])
display(rate_volatility_df.style.hide(axis="index"))
rate_volatility_df.volume = rate_volatility_df.volume.astype(float)
corr = rate_volatility_df.corr()["rate_volatility"]["volume"]
rate_volatility_df.plot(x="volume", y="rate_volatility", kind="scatter", label=f"correlation={corr:.2%}")
display(corr)
plt.legend(loc="upper left");

# %%
# rate historgrams

bins = np.arange(0.0175, 0.055, 0.0025)
# histogram of starting rate
fig = plt.figure()
first_row_in_each_experiment = rate_paths.groupby('experiment').first()
h1 = first_row_in_each_experiment['fixed_rate'].hist(label="starting fixed rate", bins=bins)

# histogram of ending rate
# fig = plt.figure()
last_row_in_each_experiment = rate_paths.groupby('experiment').last()
h2 = last_row_in_each_experiment['fixed_rate'].hist(label="ending fixed rate", bins=bins, alpha=0.5)
plt.legend()
plt.show()

# %%
# check bad experiment outcomes
delta = 0.00005
MIN_RATE = 0.035 - delta
MAX_RATE = 0.035 + delta
rate_paths["day"] = rate_paths["timestamp"].dt.day
agg_data = rate_paths.groupby('day')['fixed_rate'].agg(['mean', 'std']).reset_index()
rates_by_day_and_experiment = rate_paths.groupby(['day', 'experiment'])['fixed_rate'].mean().reset_index()
good_ending_rates = (last_row_in_each_experiment.fixed_rate > MIN_RATE) & (last_row_in_each_experiment.fixed_rate < MAX_RATE)
filtered_rows = last_row_in_each_experiment.loc[~good_ending_rates,:]
bad_experiments = list(filtered_rows.index)
print(f"bad experiments defined as ending rates outside of {MIN_RATE:.3%} and {MAX_RATE:.3%}:")
print(f"  we have {len(bad_experiments)} bad experiments: {bad_experiments}")

if len(bad_experiments) > 0:
    # plot rate paths for bad experiments
    for experiment in bad_experiments:
        idx = rates_by_day_and_experiment['experiment'] == experiment
        plt.plot(rates_by_day_and_experiment.loc[idx, 'day'], rates_by_day_and_experiment.loc[idx, 'fixed_rate'], label=experiment)
    plt.xlabel('Day')
    plt.ylabel('Ending Fixed Rate')
    plt.show()

# %%
# inspect df2
experiment = 0
df2temp = df2.loc[df2["experiment"] == experiment]
# display(df2temp)
display(
    df2temp.style.format(
        subset=[
            col
            for col in df2temp.columns
            if df2temp.dtypes[col] == "float64" and col not in ["hpr", "apr"]
        ],
        formatter="{:" + FLOAT_FMT + "}",
    )
    .hide(axis="index")
    .format(
        subset=["hpr", "apr"],
        formatter="{:.2%}",
    )
)

# %%