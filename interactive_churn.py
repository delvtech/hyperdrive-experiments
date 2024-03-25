# %%
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
from fixedpointmath import FixedPoint
from matplotlib import pyplot as plt  # pylint: disable=import-error,no-name-in-module

from agent0.hyperdrive.interactive import InteractiveHyperdrive, LocalChain

# pylint: disable=bare-except
# ruff: noqa: A001 (allow shadowing a python builtin)
# using the variable "max"
# pylint: disable=redefined-builtin
# don't make me use upper case variable names
# pylint: disable=invalid-name
# let me use long lines
# pylint: disable=line-too-long
# let me use magic numbers
# ruff: noqa: PLR2004
# too many branches
# ruff: noqa: PLR0912
# pylint: disable=redefined-outer-name

cols = ["block_number", "username", "position", "pnl"]
DISPLAY_COLS = cols + ["base_token_type", "maturity_time"]
DISPLAY_COLS_WITH_HPR = cols + ["hpr", "apr"]
FLOAT_FMT = ",.0f"
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
    print("Running in non-interactive mode.")

chain = LocalChain()

# %%
# Parameters for pool initialization. If empty, defaults to default values, allows for custom values if needed
initial_liquidity = FixedPoint(10_000_000)
config = InteractiveHyperdrive.Config(
    curve_fee=FixedPoint(0.01),
    flat_fee=FixedPoint(0),
    position_duration=60 * 60 * 24 * 365,  # 1 year
    checkpoint_duration=60 * 60 * 24,  # 1 day
    initial_liquidity=initial_liquidity,  # 10 million
    governance_lp_fee=FixedPoint(0),  # no governance fee
    calc_pnl=False,  # don't calculate pnl, we'll do all returns in WETH after closing positions
)
MINIMUM_TRANSACTION_AMOUNT = config.minimum_transaction_amount
interactive_hyperdrive = InteractiveHyperdrive(chain, config)

# %%
# set up accounts
account_1 = interactive_hyperdrive.init_agent(base=FixedPoint(1e23-1), name="account 1")

print("=== STARTING WETH BALANCES ===")
starting_base = {}
for agent in interactive_hyperdrive._pool_agents:  # pylint: disable=protected-access
    starting_base[agent.name] = agent.wallet.balance.amount
for k,v in starting_base.items():
    print(f"{k:6}: {float(v):13,.0f}")

# %%
# do trades until it crashes
try:
    for _ in range(100):
        event = account_1.open_long(base=FixedPoint(1_000_000))
        account_1.open_short(bonds=event.bond_amount)
except Exception as exc:
    print(exc)

# %%
# close positions
# events = account_1.liquidate()

# %%
# view wallets
current_wallet = deepcopy(interactive_hyperdrive.get_current_wallet())
if RUNNING_INTERACTIVE:
    display(
        current_wallet.loc[current_wallet.position != 0,DISPLAY_COLS].style.format(
            subset=[
                col
                for col in current_wallet.columns
                if current_wallet.dtypes[col] == "float64" and col not in ["hpr", "apr"]
            ],
            formatter="{:" + FLOAT_FMT + "}",
        )
        .hide(axis="index")
        .hide(axis="columns", subset=["pnl", "maturity_time"])
    )
else:
    print(current_wallet)

# %%
# conclude
pool_info = interactive_hyperdrive.get_pool_state()
initial_fixed_rate = float(pool_info.fixed_rate.iloc[0])
ending_fixed_rate = float(pool_info.fixed_rate.iloc[-1])
print(f"starting fixed rate is {initial_fixed_rate:7.2%}")
print(f"  ending fixed rate is {ending_fixed_rate:7.2%}")
governance_fees = float(interactive_hyperdrive.hyperdrive_interface.get_gov_fees_accrued(block_number=None))
current_wallet = deepcopy(interactive_hyperdrive.get_current_wallet())

wallet_positions = deepcopy(interactive_hyperdrive.get_wallet_positions())
# time passed
time_passed_days = (pool_info.timestamp.iloc[-1] - pool_info.timestamp.iloc[0]).total_seconds() / 60 / 60 / 24
print(f"time passed = {time_passed_days:.2f} days")
apr_factor = 365 / time_passed_days
print(f"  to scale APR from HPR we multiply by {apr_factor:,.0f} (365/{time_passed_days:.2f})")
print(f"  share price went from {pool_info.share_price.iloc[0]:.4f} to {pool_info.share_price.iloc[-1]:.4f}")

# do return calculations
non_weth_index = (current_wallet.token_type != "WETH") & (current_wallet.position > float(MINIMUM_TRANSACTION_AMOUNT))
weth_index = current_wallet.token_type == "WETH"
ws_index = current_wallet.token_type == "WITHDRAWAL_SHARE"
for user in current_wallet.username.unique():
    user_idx = current_wallet.username == user
    # check if user has withdrawal shares
    if (user_idx & ws_index).sum() > 0:
        # add withdrawal shares at 1:1 with WETH
        current_wallet.loc[user_idx & weth_index, ["position"]] += current_wallet.loc[user_idx & ws_index, ["position"]].values
    if user not in ["governance", "total", "share price"]:
        # simple PNL based on starting WETH balance
        current_wallet.loc[user_idx & weth_index, ["pnl"]] = current_wallet.loc[user_idx & weth_index, ["position"]].values - float(starting_base[user])
# add HPR
mask = current_wallet['pnl'].notna() & current_wallet['position'].notna()
current_wallet.loc[mask, 'hpr'] = current_wallet.loc[mask, 'pnl'] / (current_wallet.loc[mask, 'position'] - current_wallet.loc[mask, 'pnl'])

def new_row(user, position, pnl, hpr = None):  # pylint: disable=redefined-outer-name
    _new_row = current_wallet.iloc[len(current_wallet) - 1].copy()
    _new_row["username"], _new_row["position"], _new_row["pnl"], _new_row["token_type"] = user, position, pnl, "WETH"
    _new_row["hpr"] = hpr if hpr is not None else _new_row["pnl"] / (_new_row["position"] - _new_row["pnl"])
    return _new_row.to_frame().T
governance_row = new_row("governance", governance_fees, governance_fees, np.inf)
total_row = new_row("total", float(current_wallet["position"].values.sum()), current_wallet.loc[current_wallet.token_type.values == "WETH", ["pnl"]].values.sum())
share_price_row = new_row("share price", pool_info.share_price.iloc[-1] * 1e7, pool_info.share_price.iloc[-1] * 1e7 - pool_info.share_price.iloc[0] * 1e7)
current_wallet = pd.concat([current_wallet, governance_row, total_row, share_price_row], ignore_index=True)

# re-index
non_weth_index = (current_wallet.token_type != "WETH") & (current_wallet.position > float(MINIMUM_TRANSACTION_AMOUNT))
weth_index = current_wallet.token_type == "WETH"
# convert to float
current_wallet.position = current_wallet.position.astype(float)
current_wallet.pnl = current_wallet.pnl.astype(float)
# add APR
current_wallet.loc[:, ["apr"]] = current_wallet.loc[:, ["hpr"]].values * apr_factor

results1 = current_wallet.loc[non_weth_index, DISPLAY_COLS]
results2 = current_wallet.loc[weth_index, DISPLAY_COLS_WITH_HPR]
# display final results
if non_weth_index.sum() > 0:
    print("\nmaterial non-WETH positions:")
    if RUNNING_INTERACTIVE:
        display(results1.style.hide(axis="index"))
    else:
        print(results1)
else:
    print("no material non-WETH positions")
print("WETH positions:")
if RUNNING_INTERACTIVE:
    display(results2.style.format(
        subset=[col for col in results2.columns if results2.dtypes[col] == "float64" and col not in ["hpr", "apr"]],
        formatter="{:" + FLOAT_FMT + "}",
    ).hide(axis="index").format(subset=["hpr", "apr"],formatter="{:.2%}",))
else:
    print(results2)

# %%
pool_info = interactive_hyperdrive.get_pool_state()
# plot rates
if RUNNING_INTERACTIVE:
    from matplotlib import ticker
    fig, axes = plt.subplots(2,1, figsize=(10,8))
    pool_info.plot(
        x="block_number",
        y=["fixed_rate","variable_rate"],
        ax=axes[0],
    )
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    # plt.show()
    pool_info.plot(
        x="block_number",
        y="bond_reserves",
        ax=axes[1],
    )
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    plt.show()

# %%
# clear resources
if not RUNNING_INTERACTIVE:
    chain.cleanup()

# %%