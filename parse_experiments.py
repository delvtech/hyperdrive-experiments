"""Parse the experiments folder."""
# %%
from copy import copy
from decimal import Decimal
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

# import darkmode_orange

FLOAT_FMT = ",.0f"
short_variable_names = {
    "CURVE_FEE": "curve fee",
    "FLAT_FEE": "flat fee",
    "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY": "volume",
    "FIXED_RATE": "rate",
    "MINIMUM_TRADE_DAYS": "trade length"
}

# %%
# do shit
EXPERIMENT_FOLDER = Path("runs")
PARQUET_FILES = ["agg_results1.parquet", "agg_results2.parquet", "rate_paths.parquet"]
DELETE_PREVIOUS_PARQUET_FILES = True

if DELETE_PREVIOUS_PARQUET_FILES:
    for file in PARQUET_FILES:
        if Path(file).exists():
            Path(file).unlink()
df1 = pd.read_parquet("agg_results1.parquet") if Path("agg_results1.parquet").exists() else pd.DataFrame()
df2 = pd.read_parquet("agg_results2.parquet") if Path("agg_results2.parquet").exists() else pd.DataFrame()
rate_paths = pd.read_parquet("rate_paths.parquet") if Path("rate_paths.parquet").exists() else pd.DataFrame()
incomplete_runs = []
experiment_stats = pd.DataFrame()
# for each folder in the experiments folder
for folder in EXPERIMENT_FOLDER.iterdir():
    if folder.is_dir():
        if "exp" in folder.name:
            experiment_id = int(folder.name.split("_")[1])
        else:
            experiment_id = int(folder.name)
        print(f"Experiment ID {experiment_id}")
        # check if parameters.env exists and it has a non-zero size
        parameters = folder / "parameters.env"
        if parameters.exists() and parameters.stat().st_size > 0:
            # load it
            params = dotenv_values(parameters)
        if df1.shape[0] > 0 and experiment_id in df1["experiment"].values:
            print(f"Experiment ID {experiment_id} already in agg_results1.parquet")
        else:
            # check if results1.parquet exists and it has a non-zero size
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
            else:
                incomplete_runs.append(experiment_id)
                print(f"Experiment ID {experiment_id} has no results1.parquet")
        if df2.shape[0] > 0 and experiment_id in df2["experiment"].values:
            print(f"Experiment ID {experiment_id} already in agg_results2.parquet")
        else:
            # check if results2.parquet exists and it has a non-zero size
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
            else:
                incomplete_runs.append(experiment_id)
                print(f"Experiment ID {experiment_id} has no results2.parquet")
        if rate_paths.shape[0] > 0 and experiment_id in rate_paths["experiment"].values:
            print(f"Experiment ID {experiment_id} already in rate_paths.parquet")
        else:
            # check if pool_info.parquet exists and it has a non-zero size
            file3 = folder / "pool_info.parquet"
            if file3.exists() and file3.stat().st_size > 0:
                # load it
                print(f"loading results from {file3} for experiment {experiment_id}")
                df_new = pd.read_parquet(file3)
                # add the experiment id
                df_new["experiment"] = experiment_id
                # append it
                print("rate paths before concat: ", rate_paths.shape)
                print("df_new: ", df_new.shape)
                rate_paths = pd.concat([rate_paths, df_new], ignore_index=True, axis=0)
                print("rate paths  after concat: ", rate_paths.shape)
            else:
                incomplete_runs.append(experiment_id)
                print(f"Experiment ID {experiment_id} has no pool_info.parquet")
        file4 = folder / "experiment_stats.json"
        if file4.exists() and file4.stat().st_size > 0:
            # load it
            print(f"loading results from {file4} for experiment {experiment_id}")
            new_df = pd.read_json(file4, orient="records", lines=True)
            new_df.index = pd.Index(data=[experiment_id],name="experiment")
            experiment_stats = pd.concat([experiment_stats, new_df], ignore_index=False)
        else:
            incomplete_runs.append(experiment_id)
            print(f"Experiment ID {experiment_id} has no experiment_stats.json")
if "AGENT0_INSTALL_FOLDER" in df2.columns:
    df2 = df2.drop(columns=["AGENT0_INSTALL_FOLDER"])
# df1.to_parquet("agg_results1.parquet")
# df2.to_parquet("agg_results2.parquet")
incomplete_runs = list(set(incomplete_runs))
print(f"{len(incomplete_runs)} incomplete runs: {','.join(map(str, incomplete_runs))}")

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
# manipulate columns
cols = df2.columns
# keep only columns after "experiment"
cols = cols[cols.get_loc("experiment") + 1 :]
# convert to float
df2[cols] = df2[cols].astype(float)
df2.apr = df2.apr.astype(float)
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
# display(
#     df2.loc[idx | last_share_price, :]
#     .rename(columns={"DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY": "volume_per_day"})
#     .sort_values("apr", ascending=False)
#     .style.format(
#         subset=["apr", "position", "total_volume"],
#         formatter="{:" + FLOAT_FMT + "}",
#     )
#     .hide(axis="index")
#     .format(
#         subset=["hpr", "apr"],
#         formatter="{:.2%}",
#     )
#     .hide(
#         axis="columns",
#         subset=[
#             "block_number",
#             "TERM_DAYS",
#             "AMOUNT_OF_LIQUIDITY",
#             "GOVERNANCE_FEE",
#             "RANDSEED",
#             "FLAT_FEE",
#         ],
#     )
# )
# df2.loc[idx | last_share_price, :].to_csv("bigtable.csv", index=False)

# %%
# check every combination
# var_list = ["CURVE_FEE", "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY", "FIXED_RATE"]
var_list = ["CURVE_FEE", "FLAT_FEE", "MINIMUM_TRADE_DAYS", "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY"]
# keep only vars that exist in our columns
var_list = [v for v in var_list if v in df2.columns]
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
variables_to_check_for_variability = ["CURVE_FEE", "FLAT_FEE", "MINIMUM_TRADE_DAYS", "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY"]
variables_to_check_for_variability = [v for v in variables_to_check_for_variability if v in df2.columns]
most_variable = ''
least_variable = ''
for variable in variables_to_check_for_variability:
    num_unique = df2[variable].nunique()
    print(f"{variable} has {num_unique} unique values")
    if most_variable == '':
        most_variable = variable
    if least_variable == '':
        least_variable = variable
    if num_unique > df2[most_variable].nunique():
        most_variable = variable
    if num_unique < df2[least_variable].nunique():
        least_variable = variable
print(f"most variable is {most_variable}")
print(f"least variable is {least_variable}")

# %%
# little 1-d table
matrix = df2.loc[idx, [most_variable, "apr"]].pivot_table(
    columns=most_variable, values=["apr"], aggfunc="mean"
)
matrix_formatted = matrix.copy()
column_index = matrix.columns
column_index.name = ''
short_name = short_variable_names[most_variable]
fmt = ",.1%" if short_name != "MINIMUM_TRADE_DAYS" else ",.0f"
column_index = matrix.columns.map(lambda x: f"{short_name}={x:,.0f}")
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
# plot experiment_stats[experiment_id, "total_volume"] vs. "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY"
plot_data = copy(df2.loc[idx, :])
y_data = [experiment_stats.loc[x,"total_volume"] for x in plot_data.loc[:, "EXPERIMENT_ID"]]
plot_data["total_volume"] = y_data
x_data = plot_data.loc[:, "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY"]
p = plt.scatter(x_data,y_data, label='total_volume')
color = p.get_facecolor()
p.set_edgecolor(color)
p.set_facecolor("none")
plt.xlabel("target_volume")
plt.ylabel("total_volume")
# plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
plt.title(f"total_volume vs. target_volume")
plt.show()

# groupby DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY, show min and max of total_volume
minmaxvol = plot_data.groupby("DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY").agg(
    min_total_volume=("total_volume", "min"),
    max_total_volume=("total_volume", "max"),
).reset_index()
minmaxvol.rename(columns={"DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY": "target_volume"}, inplace=True)
minmaxvol["max_vs_min"] = minmaxvol.max_total_volume - minmaxvol.min_total_volume
minmaxvol["max_vs_min_pct"] = minmaxvol.max_vs_min / minmaxvol.min_total_volume
display(minmaxvol.style.hide(axis="index")
    .format(subset=["min_total_volume", "max_total_volume", "max_vs_min", "max_vs_min_pct"],formatter="{:" + FLOAT_FMT + "}")
    .format(subset=["target_volume","max_vs_min_pct"],formatter="{:,.2%}")
)

# %%
# plot APR vs. volume for given curve and flat fees
plot_data = copy(df2.loc[idx, :])
records = []
for curve_fee in np.sort(df2.loc[idx, "CURVE_FEE"].unique()):
    for flat_fee in np.sort(df2.loc[idx, "FLAT_FEE"].unique()):
        subidx = (plot_data.CURVE_FEE == curve_fee) & (plot_data.FLAT_FEE == flat_fee)
        print(f"num={subidx.sum()}")
        x_data = plot_data.loc[subidx, "DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY"]
        # x_data = [experiment_stats.loc[x,"total_volume"] for x in plot_data.loc[subidx, "EXPERIMENT_ID"]]
        y_data = plot_data.loc[subidx, "apr"]
        p = plt.scatter(
            x_data,
            y_data,
            label=f"curve_fee={curve_fee:.2%}, flat_fee={flat_fee:.2%}",
            alpha=1,
        )
        plt.xlabel(short_variable_names["DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY"])
        plt.ylabel("Return (APR)")
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ylim = plt.gca().get_ylim()
        ylim = np.floor(ylim[0] * 1000) / 1000, np.ceil(ylim[1] * 1000) / 1000
        plt.gca().set_ylim(ylim)
        yticks = np.arange(min(ylim[0],0.034), ylim[1]+0.001, 0.001)  # 0.1% increment
        plt.gca().set_yticks(yticks)
        # use sm to get linear regression line of best fit
        model = sm.OLS(y_data, sm.add_constant(x_data)).fit()
        # plot linear regression line
        # plt.plot(x_vars_with_zero, y_fit, color='orange')
        plt.plot(x_data, model.predict(sm.add_constant(x_data)), color='orange')
        plt.axhline(0.035, color='red', label="Vault variable return", alpha=1, linestyle='--')
        plt.title(f"LP Profitability vs. {short_variable_names['DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY']}")
        plt.legend()
        plt.show()
        slope = model.params[1]
        records.append([curve_fee,flat_fee,slope])

# %%
df3 = pd.DataFrame(records, columns=["CURVE_FEE","FLAT_FEE","SLOPE"])
# df3 = df3.set_index(["CURVE_FEE","FLAT_FEE"])
display(df3)

display(df3.pivot(index="FLAT_FEE", columns="CURVE_FEE", values="SLOPE"))

for curve_fee in np.sort(df2.loc[idx, "CURVE_FEE"].unique()):
    subidx = df3.CURVE_FEE == curve_fee
    x_data = df3.loc[subidx, "FLAT_FEE"]
    y_data = df3.loc[subidx, "SLOPE"]
    p = plt.scatter(x_data,y_data,label=f"curve_fee={curve_fee:.2%}")
plt.xlabel(short_variable_names["FLAT_FEE"])
plt.ylabel("Slope")
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
plt.legend()
plt.title(f"Slope vs. {short_variable_names['FLAT_FEE']}")
plt.show()

for flat_fee in np.sort(df2.loc[idx, "FLAT_FEE"].unique()):
    subidx = df3.FLAT_FEE == flat_fee
    x_data = df3.loc[subidx, "CURVE_FEE"]
    y_data = df3.loc[subidx, "SLOPE"]
    p = plt.scatter(x_data,y_data,label=f"flat_fee={flat_fee:.2%}")
plt.xlabel(short_variable_names["CURVE_FEE"])
plt.ylabel("Slope")
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
plt.legend()
plt.title(f"Slope vs. {short_variable_names['CURVE_FEE']}")
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
# prepare rate paths
min_timestamp_by_experiment = rate_paths.groupby('experiment')['timestamp'].min()
rate_paths.loc[:,"adjusted_timestamp"] = rate_paths["timestamp"] - min_timestamp_by_experiment[rate_paths["experiment"]].values
rate_paths['adjusted_timestamp_seconds'] = rate_paths['adjusted_timestamp'].dt.total_seconds()
rate_paths['adjusted_timestamp_days'] = rate_paths['adjusted_timestamp_seconds'] / (60 * 60 * 24)
rate_paths["fixed_rate"] = rate_paths["fixed_rate"].astype(float)

# %%
# plot rate paths
fig = plt.figure(figsize=(16/2, 9/2))
ax = plt.gca()
unique_experiments = rate_paths['experiment'].unique()
rate_volatility_records = []
for experiment in unique_experiments[:20]:
    idx = rate_paths['experiment'] == experiment
    rate_paths.loc[idx].plot(x='adjusted_timestamp_days', y='fixed_rate', label=experiment, alpha=0.2, ax=ax)
    rate_volatility = rate_paths.loc[idx, "fixed_rate"].std()
    volume = df1.loc[df1.experiment==experiment,"DAILY_VOLUME_PERCENTAGE_OF_LIQUIDITY"].iloc[0]
    fee = df1.loc[df1.experiment==experiment,"CURVE_FEE"].iloc[0]
    # print(f"{rate_volatility=}")
    rate_volatility_records.append([experiment, rate_volatility, volume, fee])
# disable legend
ax.legend().set_visible(False)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
ylim = plt.gca().get_ylim()
ylim = 0, ylim[1]
plt.gca().set_ylim(ylim)
inc = 0.005  # 0.5% increment
yticks = np.arange(ylim[0], ylim[1]+inc, inc)
plt.gca().set_yticks(yticks)
plt.xlabel('Time (days)')
plt.xticks(rotation=45)
plt.xticks(np.append(np.arange(0, 360, 30),365))
plt.ylabel('Fixed Rate')
plt.ylim(0, 0.06)
plt.show()

# %%
# plot rate volatility vs. volume
rate_volatility_df = pd.DataFrame(rate_volatility_records, columns=["experiment", "rate_volatility", "volume", "fee"])
display(rate_volatility_df.style.hide(axis="index"))
rate_volatility_df.volume = rate_volatility_df.volume.astype(float)
corr = rate_volatility_df.corr()["rate_volatility"]["volume"]
rate_volatility_df.plot(x="volume", y="rate_volatility", kind="scatter", label=f"correlation={corr:.2%}")
display(corr)
plt.legend(loc="upper left")
plt.title("Rate Volatility vs. Volume")
plt.show()

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
# inspect results of a specific experiment
experiment = 0
df1temp = df1.loc[df1["experiment"] == experiment]
display(df1temp)
df2temp = df2.loc[df2["experiment"] == experiment]
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
# %%
# inspect pool_info of a specific experiment
pool_info = pd.read_parquet("results/exp_five/4/pool_info.parquet")
pool_info

# %%
# inspect current_wallet of a specific experiment
current_wallet = pd.read_parquet("results/exp_five/4/current_wallet.parquet")
current_wallet

# %%