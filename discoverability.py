# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fixedpointmath import FixedPoint

from agent0.core.hyperdrive.interactive import LocalChain, LocalHyperdrive
from agent0.core.hyperdrive.interactive.local_hyperdrive_agent import LocalHyperdriveAgent
from agent0.ethpy.hyperdrive.interface.read_write_interface import HyperdriveReadWriteInterface
from darkmode_orange import *

RUNNING_INTERACTIVE = False
try:
    from IPython.core.getipython import get_ipython  # pylint: disable=import-outside-toplevel

    RUNNING_INTERACTIVE = bool("ipykernel" in sys.modules and get_ipython())
except ImportError:
    pass

if RUNNING_INTERACTIVE:
    from IPython.display import display  # pylint: disable=import-outside-toplevel

    print("Running in interactive mode.")
else:  # being run from the terminal or something similar
    display = print  # pylint: disable=redefined-builtin,unused-import
    plt.switch_backend('Agg')  # switch to non-interactive backend
    print("Running in non-interactive mode.")

# pylint: disable=missing-function-docstring, missing-return-type-doc, missing-return-doc, redefined-outer-name, line-too-long, wildcard-import, bare-except
# ruff: noqa: D103

# %%
def calc_price_and_rate(interface:HyperdriveReadWriteInterface):
    price = interface.calc_spot_price()
    rate = interface.calc_spot_rate()
    return price, rate

def trade(interface:HyperdriveReadWriteInterface, agent:LocalHyperdriveAgent, trade_portion, max_long, max_short):
    relevant_max = max_long if trade_portion > 0 else max_short
    trade_size = float(relevant_max) * trade_portion
    trade_result = trade_long(interface, agent, trade_size) if trade_size > 0 else trade_short(interface, agent, abs(trade_size))
    return *trade_result, trade_size

def trade_long(interface:HyperdriveReadWriteInterface, agent:LocalHyperdriveAgent, trade_size):
    try:
        trade_result = agent.open_long(base=FixedPoint(trade_size))
        base_traded = trade_result.amount
        bonds_traded = trade_result.bond_amount
        return *calc_price_and_rate(interface), base_traded, bonds_traded
    except:
        pass
    return None, None, None, None

def trade_short(interface:HyperdriveReadWriteInterface, agent:LocalHyperdriveAgent, trade_size):
    try:
        trade_result = agent.open_short(bonds=FixedPoint(trade_size))
        base_traded = -trade_result.amount
        bonds_traded = -trade_result.bond_amount
        return *calc_price_and_rate(interface), base_traded, bonds_traded
    except:
        pass
    return None, None, None, None

def trade_liq(interface:HyperdriveReadWriteInterface, agent:LocalHyperdriveAgent, trade_size):
    agent.add_liquidity(base=trade_size)
    return calc_price_and_rate(interface)

# %%

# sourcery skip: merge-list-append, move-assign-in-block
YEAR_IN_SECONDS = 31_536_000
TIME_STRETCH_LIST = [0.05, 0.1, 0.2]
TRADE_PORTION_ONE = 0

chain = LocalChain(LocalChain.Config(chain_port=10_000, db_port=10_001))

liquidity = FixedPoint(100)

trade_portion_list = [*np.arange(0.1, 1.1, 0.1)]
trade_portion_list += [-x for x in trade_portion_list]  # add negative portions

all_results = pd.DataFrame()
for trial,TIME_STRETCH_APR in enumerate(TIME_STRETCH_LIST):
    records = []
    print(f"Time stretch APR: {TIME_STRETCH_APR}")
    interactive_config = LocalHyperdrive.Config(
        position_duration=YEAR_IN_SECONDS,  # 1 year term
        governance_lp_fee=FixedPoint(0.1),
        curve_fee=FixedPoint(0.01),
        flat_fee=FixedPoint(0),
        initial_liquidity=liquidity,
        initial_fixed_apr=FixedPoint(TIME_STRETCH_APR),
        initial_time_stretch_apr=FixedPoint(TIME_STRETCH_APR),
        factory_min_fixed_apr=FixedPoint(0.001),
        factory_max_fixed_apr=FixedPoint(1000),
        factory_min_time_stretch_apr=FixedPoint(0.001),
        factory_max_time_stretch_apr=FixedPoint(1000),
        calc_pnl=False,
        minimum_share_reserves=liquidity*FixedPoint(1/10_000),
    )
    hyperdrive:LocalHyperdrive = LocalHyperdrive(chain, interactive_config)
    agent:LocalHyperdriveAgent = hyperdrive.init_agent(base=FixedPoint(1e18), eth=FixedPoint(1e18))
    interface = hyperdrive.interface
    time_stretch = interface.current_pool_state.pool_config.time_stretch
    print("Time stretch: %s", time_stretch)

    # do a specific trade
    max_long = interface.calc_max_long(budget=FixedPoint(1e18))
    max_short = interface.calc_max_short(budget=FixedPoint(1e18))
    price, rate, base_traded, bonds_traded, trade_size = trade(interface, agent, TRADE_PORTION_ONE, max_long, max_short)
    records.append((trial, "first", interface.calc_effective_share_reserves(), trade_size, base_traded, bonds_traded, TRADE_PORTION_ONE, price, rate, TIME_STRETCH_APR, interface.current_pool_state.pool_info.bond_reserves, interface.current_pool_state.pool_info.share_reserves))
    price, rate = trade_liq(interface, agent, liquidity)
    records.append((trial, "addliq", interface.calc_effective_share_reserves(), trade_size, None, None, TRADE_PORTION_ONE, price, rate, TIME_STRETCH_APR, interface.current_pool_state.pool_info.bond_reserves, interface.current_pool_state.pool_info.share_reserves))
    del price, rate, trade_size

    # save the snapshot
    chain.save_snapshot()

    # do a range of trades
    max_short_two = interface.calc_max_short(budget=FixedPoint(1e18))
    max_long_two = interface.calc_max_long(budget=FixedPoint(1e18))
    for trade_portion_two in trade_portion_list:
        chain.load_snapshot()
        price, rate, base_traded, bonds_traded, trade_size = trade(interface, agent, trade_portion_two, max_long_two, max_short_two)
        records.append((trial, "second", interface.calc_effective_share_reserves(), trade_size, base_traded, bonds_traded, trade_portion_two, price, rate, TIME_STRETCH_APR, interface.current_pool_state.pool_info.bond_reserves, interface.current_pool_state.pool_info.share_reserves))
        print("trade_portion=%s, rate=%s", trade_portion_two, rate)
    columns = ["trial", "type", "liquidity", "trade_size", "base_traded", "bonds_traded", "portion", "price", "rate", "time_stretch_apr", "bond_reserves", "share_reserves"]
    new_result = pd.DataFrame.from_records(records, columns=columns)
    display(new_result)
    all_results = pd.concat([all_results, new_result], ignore_index=True, axis=0)

all_results.to_csv("discoverability.csv", index=False)

# %%
# rate vs. dollars
all_results = pd.read_csv("discoverability.csv")
X_VAL = "trade_size"  # one of trade_size, base_traded, bonds_traded
X_TITLE = X_VAL.replace("_", " ")
palette = sns.color_palette("colorblind")
unique_time_stretches = all_results.time_stretch_apr.unique()
num_data = len(unique_time_stretches)
num_rows = min(4, num_data)
num_cols = num_data//num_rows + 0
fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 8), sharex=True)
fig.text(0.5, 0.92, f"Fixed Rate vs. {X_TITLE.capitalize()}", ha="center", fontsize=14)
for idx,time_stretch in enumerate(unique_time_stretches):
    row = idx // num_rows
    col = idx % num_rows
    print(f"{row=} {col=}")
    ax = axs[row, col] if num_cols > 1 else axs[idx] if isinstance(axs, (list,np.ndarray)) else axs
    assert isinstance(ax, plt.Axes)
    print(f"{time_stretch=} len={sum(all_results.time_stretch_apr == time_stretch)}")
    all_results_filtered = all_results.loc[(all_results.type == "second") & (all_results.time_stretch_apr == time_stretch),:]
    sns.lineplot(x=X_VAL, y="rate", data=all_results_filtered, hue="trial", palette=palette, marker="o", ax=ax)
    ax.set_xlabel(f"{X_TITLE} in base (+ = long, - = short)")
    ax.set_ylabel("fixed rate after trade")
    ax.set_title(f"Time Stretch {time_stretch:.0%}, Liquidity {float(liquidity):,.0f}")
    # plot_handles = ax.get_legend_handles_labels()[0]
    # ax.legend(title="Trial", handles=plot_handles, labels=[f"{'short' if all_results.loc[all_results.trial == trial, 'portion'].values[0] < 0 else 'long'} {abs(all_results.loc[all_results.trial == trial, 'portion'].values[0]*100):.0f}%" for trial in all_results_filtered.trial.unique()])
    fig.savefig(f"time_stretch_{time_stretch*100:.0f}.png")

# %%
# multiple time stretches
fig, axs = plt.subplots(1,1, figsize=(9, 6), sharex=True)
plt.title(f"Fixed Rate vs. {X_TITLE.capitalize()}", ha="center", fontsize=14, weight="normal")
plot_data = all_results.loc[all_results.type == "second",:]
elbow_data = plot_data.loc[plot_data.portion==-0.8,:]
sns.lineplot(x=X_VAL, y="rate", data=plot_data, hue="time_stretch_apr", palette=palette, marker="o", ax=axs)
sns.lineplot(x=X_VAL, y="rate", label="elbow method", data=elbow_data, palette=palette, marker="o", ax=axs)
plot_handles = axs.get_legend_handles_labels()[0]
plot_labels = [f"time_stretch {time_stretch:>3.0%}" for time_stretch in plot_data.time_stretch_apr.unique()]
plot_labels += ["elbow method"]
axs.legend(handles=plot_handles, labels=plot_labels)
fig.savefig("multiple_time_stretches.png")

# %%
# elbow method
fig, axs = plt.subplots(1,1, figsize=(9, 6), sharex=True)
sns.lineplot(x="time_stretch_apr", y="rate", data=elbow_data, palette=palette, marker="o", color="red")
plt.title("Elbow method")
regression = np.polyfit(elbow_data.time_stretch_apr, elbow_data.rate, 1)
p=plt.plot(elbow_data.time_stretch_apr, regression[0] * elbow_data.time_stretch_apr + regression[1], color="blue", linestyle="dashed", linewidth=2)
p[0].set_label(f"y = {regression[0]:.2f}x + {regression[1]:.2f}")
plt.legend()
fig.savefig("elbow.png")

# %%
