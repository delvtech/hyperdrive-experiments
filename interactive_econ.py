# %%
"""Run experiments of economic activity.

We want to better understand return profiles of participants in Hyperdrive.
To do so, we run various scenarios of plausible economic activity.
We target a certain amount of daily activity, as a percentage of the liquidity provided.
That trading activity is executed by a random agent named Rob.
The liquidity is provided by an agent named Larry.
At the end, we close out all positions, and evaluate results based off the WETH in their wallets.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field, fields
from decimal import Decimal

import numpy as np
import pandas as pd
import wandb
from chainsync.db.hyperdrive import get_wallet_deltas
from dotenv import load_dotenv
from fixedpointmath import FixedPoint
from matplotlib import pyplot as plt  # pylint: disable=import-error,no-name-in-module

from agent0.hyperdrive.interactive import InteractiveHyperdrive, LocalChain
from agent0.hyperdrive.interactive.interactive_hyperdrive_agent import InteractiveHyperdriveAgent
from agent0.hyperdrive.policies import Zoo
from utils import get_max, running_interactive, running_wandb, safe_cast  # pylint: disable=no-name-in-module

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


# %%
# check what environment we're running in
if RUNNING_INTERACTIVE := running_interactive():
    from IPython.display import display  # pylint: disable=import-outside-toplevel

    print("Running in interactive mode.")
else:  # being run from the terminal or something similar
    display = print  # pylint: disable=redefined-builtin
    logging.basicConfig(level=logging.ERROR)
    print("Running in non-interactive mode.")

if RUNNING_WANDB := running_wandb():
    print("Running inside a wandb environment.")
else:
    print("Not running inside a wandb environment.")

# %%
# config
cols = ["block_number", "username", "position", "pnl"]


@dataclass
class ExperimentConfig:  # pylint: disable=too-many-instance-attributes,missing-class-docstring
    experiment_id: int = 0
    db_port: int = 5_433
    chain_port: int = 10_000
    daily_volume_percentage_of_liquidity: float = 0.05
    term_days: int = 365
    minimum_trade_days: float = 0  # minimum number of days to keep a trade open
    float_fmt: str = ",.0f"
    display_cols: list[str] = field(default_factory=lambda: cols + ["base_token_type", "maturity_time"])
    display_cols_with_hpr: list[str] = field(default_factory=lambda: cols + ["hpr", "apr"])
    amount_of_liquidity: int = 10_000_000
    max_trades_per_day: int = 10
    fixed_rate: FixedPoint = FixedPoint(0.045)
    curve_fee: FixedPoint = FixedPoint("0.005")
    flat_fee: FixedPoint = FixedPoint("0.0005")
    governance_fee: FixedPoint = FixedPoint("0")
    randseed: int = 0
    term_seconds: int = 0
    variable_rate: FixedPoint = FixedPoint(0.045)
    calc_pnl: bool = False
    use_average_spend: bool = False

    def calculate_values(self):
        self.term_seconds: int = 60 * 60 * 24 * self.term_days


# %%
# config
exp = ExperimentConfig()
field_names = [f.name for f in fields(exp)]
print("=== CONFIG ===")
if RUNNING_INTERACTIVE:
    load_dotenv("./runs/98/parameters.env")  # inspect a specific experiment
    pass  # do nothing
else:
    # when calling from the commandline, we always load parameters.env
    load_dotenv("parameters.env")
for key, value in os.environ.items():
    lkey = key.lower()
    if lkey in field_names:
        attribute_type = exp.__annotations__[lkey]  # pylint: disable=no-member
        setattr(exp, lkey, safe_cast(attribute_type, value, debug=False))
        # check that it worked
        print(f"  {lkey} = {getattr(exp, lkey)} ({type(getattr(exp, lkey)).__name__})")
        assert getattr(exp, lkey) == safe_cast(attribute_type, value)
        # check type
        assert (
            type(getattr(exp, lkey)).__name__ == attribute_type
        ), f"{type(getattr(exp, lkey)).__name__} != {attribute_type}"
# if chain_port not provided, set it to 10000 + experiment_id
if "chain_port" not in os.environ:
    exp.chain_port = 10_000 + int(exp.experiment_id)
    print(f"  chain_port = {exp.chain_port}")
if "db_port" not in os.environ:
    exp.db_port = 5_433 + int(exp.experiment_id)
    print(f"  db_port = {exp.db_port}")

# update calculated values
exp.calculate_values()
rng = np.random.default_rng(seed=int(exp.randseed))

# %%
# set up chain
chain = LocalChain(LocalChain.Config(db_port=exp.db_port, chain_port=exp.chain_port))

# %%
# set up pool
config = InteractiveHyperdrive.Config(
    position_duration=exp.term_seconds,
    checkpoint_duration=60 * 60 * 24,  # 1 day
    initial_liquidity=FixedPoint(20),
    initial_fixed_apr=exp.fixed_rate,
    initial_variable_rate=exp.variable_rate,
    curve_fee=exp.curve_fee,
    flat_fee=exp.flat_fee,
    governance_lp_fee=exp.governance_fee,
    calc_pnl=exp.calc_pnl,
)
MINIMUM_TRANSACTION_AMOUNT = config.minimum_transaction_amount
MINIMUM_TRANSACTION_AMOUNT *= 5
interactive_hyperdrive = InteractiveHyperdrive(chain, config)

# %%
# set up agents
larry = interactive_hyperdrive.init_agent(base=FixedPoint(exp.amount_of_liquidity), name="larry")
rob = interactive_hyperdrive.init_agent(base=FixedPoint(exp.amount_of_liquidity), name="rob")
andy_base = FixedPoint(exp.amount_of_liquidity * 100)
andy_config = Zoo.lp_and_arb.Config(
    lp_portion=FixedPoint(0),
    high_fixed_rate_thresh=FixedPoint(0),
    low_fixed_rate_thresh=FixedPoint(0),
    minimum_trade_amount=MINIMUM_TRANSACTION_AMOUNT,
)
andy = interactive_hyperdrive.init_agent(base=andy_base, name="andy", policy=Zoo.lp_and_arb, policy_config=andy_config)
print("=== STARTING WETH BALANCES ===")
starting_base = {}
for agent in interactive_hyperdrive._pool_agents:  # pylint: disable=protected-access
    starting_base[agent.name] = agent.wallet.balance.amount
for k, v in starting_base.items():
    print(f"{k:6}: {float(v):13,.0f}")
larry.add_liquidity(base=FixedPoint(exp.amount_of_liquidity))  # 10 million

# %%
# At START Arbitrage Andy does one trade 📈
event_list = andy.execute_policy_action()
for event in event_list:
    print(event)


# %%
# Random Rob does a buncha trades 🤪
def trade_in_direction(
    go_long: bool,
    agent: InteractiveHyperdriveAgent,
    _interactive_hyperdrive: InteractiveHyperdrive,
    _share_price: FixedPoint,
    _spot_price: FixedPoint,
    _current_block_time: int,
    _amount_to_trade_base: FixedPoint,
):
    max = None
    event = None
    # execute the trade in the chosen direction
    if go_long:
        if len(agent.wallet.shorts) > 0:  # check if we have shorts, and close them if we do
            for maturity_time, short in agent.wallet.shorts.copy().items():
                days_passed = (short.maturity_time - _current_block_time) // 60 // 60 // 24
                if days_passed >= exp.minimum_trade_days:
                    max = get_max(_interactive_hyperdrive, _share_price, agent.wallet.balance.amount)
                    amount_to_trade_bonds = _interactive_hyperdrive.interface.calc_bonds_out_given_shares_in_down(
                        _amount_to_trade_base / _share_price, pool_state
                    )
                    trade_size_bonds = min(amount_to_trade_bonds, short.balance, max.long.bonds)
                    if trade_size_bonds > MINIMUM_TRANSACTION_AMOUNT:
                        event = agent.close_short(maturity_time, trade_size_bonds)
                        _amount_to_trade_base -= event.bond_amount * _spot_price
                    if _amount_to_trade_base <= 0:
                        break  # stop looping across shorts if we've traded enough
                else:
                    print(f"not closing position of {days_passed=} because it's earlier than {exp.minimum_trade_days=}")
        if _amount_to_trade_base > 0:
            max = get_max(_interactive_hyperdrive, _share_price, agent.wallet.balance.amount)
            trade_size_base = min(_amount_to_trade_base, max.long.base)
            if trade_size_base > MINIMUM_TRANSACTION_AMOUNT:
                event = agent.open_long(trade_size_base)
                _amount_to_trade_base -= event.base_amount
    else:
        if len(agent.wallet.longs) > 0:  # check if we have longs, and close them if we do
            for maturity_time, long in agent.wallet.longs.copy().items():
                days_passed = (long.maturity_time - _current_block_time) // 60 // 60 // 24
                if days_passed >= exp.minimum_trade_days:
                    max = get_max(_interactive_hyperdrive, _share_price, agent.wallet.balance.amount)
                    amount_to_trade_bonds = _interactive_hyperdrive.interface.calc_bonds_out_given_shares_in_down(
                        _amount_to_trade_base / _share_price, pool_state
                    )
                    trade_size_bonds = min(amount_to_trade_bonds, long.balance, max.short.bonds)
                    if trade_size_bonds > MINIMUM_TRANSACTION_AMOUNT:
                        event = agent.close_long(maturity_time, trade_size_bonds)
                        _amount_to_trade_base -= event.base_amount
                    if _amount_to_trade_base <= 0:
                        break  # stop looping across longs if we've traded enough
                else:
                    print(f"not closing position of {days_passed=} because it's earlier than {exp.minimum_trade_days=}")
        if _amount_to_trade_base > 0:
            max = get_max(_interactive_hyperdrive, _share_price, agent.wallet.balance.amount)
            amount_to_trade_bonds = _interactive_hyperdrive.interface.calc_bonds_out_given_shares_in_down(
                _amount_to_trade_base / share_price, pool_state
            )
            trade_size_bonds = min(amount_to_trade_bonds, max.short.bonds)
            if trade_size_bonds > MINIMUM_TRANSACTION_AMOUNT:
                event = agent.open_short(trade_size_bonds)
                _amount_to_trade_base -= event.bond_amount * _spot_price
    return _amount_to_trade_base


# sourcery skip: avoid-builtin-shadow, do-not-use-bare-except, invert-any-all,
# remove-unnecessary-else, swap-if-else-branches
fixed_rates = []
start_time = time.time()
for day in range(exp.term_days):
    amount_to_trade_base = FixedPoint(exp.amount_of_liquidity * exp.daily_volume_percentage_of_liquidity)
    trades_today = 0
    while amount_to_trade_base > MINIMUM_TRANSACTION_AMOUNT:
        pool_state = interactive_hyperdrive.interface.current_pool_state
        share_price = pool_state.pool_info.lp_share_price
        spot_price = interactive_hyperdrive.interface.calc_spot_price(pool_state)
        current_block_time: int = pool_state.block_time

        # decide direction to trade
        # go_long = rng.random() < 0.5  # go long 50% of the time
        go_long = interactive_hyperdrive.interface.calc_fixed_rate(pool_state) > pool_state.variable_rate
        # X% of the time let the arbitrageur act
        arbitrageur_chance = 0
        if rng.random() < arbitrageur_chance:
            # go_long = interactive_hyperdrive.interface.calc_fixed_rate() > interactive_hyperdrive.interface.get_variable_rate()
            for event in andy.execute_policy_action():
                amount_to_trade_base -= event.bond_amount * spot_price
        else:
            amount_to_trade_base = trade_in_direction(
                go_long, rob, interactive_hyperdrive, share_price, spot_price, current_block_time, amount_to_trade_base
            )
        fixed_rates.append(interactive_hyperdrive.interface.calc_fixed_rate())
        print(f"day {day} secs/day={(time.time() - start_time)/(day+1):,.1f}", end="\r", flush=True)
        trades_today += 1
        if RUNNING_WANDB:
            wandb.log({"day": day})
        if amount_to_trade_base < MINIMUM_TRANSACTION_AMOUNT or trades_today >= exp.max_trades_per_day:
            break  # end the day if we've traded enough
    chain.advance_time(datetime.timedelta(days=1), create_checkpoints=False)
# make sure a year has passed
# if day < 364:  # days are 0-indexed
#     chain.advance_time(datetime.timedelta(days=364 - day), create_checkpoints=True)
print(f"experiment finished in {(time.time() - start_time):,.2f} seconds")

# %%
# inspect pool state
pool_state = interactive_hyperdrive.get_pool_state()
pool_state.to_parquet("pool_state.parquet")
effective_shares = pool_state.share_reserves.iloc[-1] + pool_state.share_adjustment.iloc[-1]
print(
    f"pool reserves are: bonds={pool_state.bond_reserves.iloc[-1]:,.0f} effective_shares={effective_shares:,.0f} rate={pool_state.fixed_rate.iloc[-1]:7.2%}"
)

# %%
# view wallets before closing
current_wallet = deepcopy(interactive_hyperdrive.get_current_wallet())
if RUNNING_INTERACTIVE:
    display(
        current_wallet.loc[current_wallet.position != 0, exp.display_cols]
        .style.format(
            subset=[
                col
                for col in current_wallet.columns
                if current_wallet.dtypes[col] == "float64" and col not in ["hpr", "apr"]
            ],
            formatter="{:" + exp.float_fmt + "}",
        )
        .hide(axis="index")
        .hide(axis="columns", subset=["pnl", "maturity_time"])
    )
else:
    print(current_wallet)

# %%
# Liquidate Rob's trades, at wherever the rate is
events = rob.liquidate()
for event in events:
    print(event)
# Before liquidation Arbitrage Andy does one trade 📈
event_list = andy.execute_policy_action()
for event in event_list:
    print(event)
# move forward a year to let the arbitrage expire profitably
chain.advance_time(datetime.timedelta(days=365), create_checkpoints=False)
# let Andy close their position
events = andy.liquidate()
for event in events:
    print(event)
# Now rate is where we want it to be
if larry.wallet.lp_tokens > FixedPoint(0):
    larry.remove_liquidity(larry.wallet.lp_tokens)

# %%
# conclude
pool_info = interactive_hyperdrive.get_pool_state()
initial_fixed_rate = float(pool_info.fixed_rate.iloc[0])
ending_fixed_rate = float(pool_info.fixed_rate.iloc[-1])
print(f"starting fixed rate is {initial_fixed_rate:7.2%}")
print(f"  ending fixed rate is {ending_fixed_rate:7.2%}")
governance_fees = float(interactive_hyperdrive.interface.get_gov_fees_accrued(block_number=None))
current_wallet = deepcopy(interactive_hyperdrive.get_current_wallet())

wallet_positions = deepcopy(interactive_hyperdrive.get_wallet_positions())
weth_changes = wallet_positions.loc[wallet_positions.token_type == "WETH", :].copy()
weth_changes.loc[:, "absDelta"] = abs(weth_changes["delta"])
weth_changes.loc[:, "day"] = (weth_changes.timestamp - weth_changes.timestamp.min()).dt.days + 1
weth_changes_agg = weth_changes[["day", "absDelta"]].groupby("day").sum().reset_index()
total_volume = weth_changes_agg.absDelta.sum()
print(f"  total volume is {total_volume:,.0f}")
print(f"  curve_fee={float(exp.curve_fee):,.2%}")
print(f"  volume/day={exp.daily_volume_percentage_of_liquidity:,.2%} of TVL")
# time passed
time_passed_days = (pool_info.timestamp.iloc[-1] - pool_info.timestamp.iloc[0]).total_seconds() / 60 / 60 / 24
print(f"time passed = {time_passed_days:.2f} days")
apr_factor = 365 / time_passed_days
print(f"  to scale APR from HPR we multiply by {apr_factor:,.0f} (365/{time_passed_days:.2f})")
print(f"  share price went from {pool_info.lp_share_price.iloc[0]:.4f} to {pool_info.lp_share_price.iloc[-1]:.7f}")

# do return calculations
share_price = pool_info.lp_share_price.iloc[-1]
non_weth_index = (current_wallet.token_type != "WETH") & (current_wallet.position > float(MINIMUM_TRANSACTION_AMOUNT))
weth_index = current_wallet.token_type == "WETH"
ws_index = current_wallet.token_type == "WITHDRAWAL_SHARE"
is_larry = current_wallet.username == "larry"
for user in current_wallet.username.unique():
    user_idx = current_wallet.username == user
    # check if user has withdrawal shares
    if (user_idx & ws_index).sum() > 0:
        # add withdrawal shares valued at the latest share price
        withdrawal_shares = current_wallet.loc[user_idx & ws_index, ["position"]].sum().values[0]
        current_wallet.loc[user_idx & weth_index, ["position"]] += withdrawal_shares * share_price
        print(
            f"adding pnl for {user} holding {withdrawal_shares:,.0f} withdrawal shares valued at share_price={share_price:,.7f} each"
        )
    if user not in ["governance", "total", "share price"]:
        # simple PNL based on starting WETH balance
        current_wallet.loc[user_idx & weth_index, ["pnl"]] = current_wallet.loc[
            user_idx & weth_index, ["position"]
        ].values - Decimal(str(starting_base[user]))
# add HPR
mask = current_wallet["pnl"].notna() & current_wallet["position"].notna()
current_wallet.loc[mask, "hpr"] = current_wallet.loc[mask, "pnl"] / (
    current_wallet.loc[mask, "position"] - current_wallet.loc[mask, "pnl"]
)

# calculate average spend if it's turned on
if exp.use_average_spend is True:
    wallet_positions_by_time = (
        wallet_positions.loc[wallet_positions.token_type == "WETH", :]
        .pivot(index="timestamp", columns="username", values="position")
        .reset_index()
    )
    wallet_positions_by_time.loc[:, ["rob"]] = (
        wallet_positions_by_time["rob"].max() - wallet_positions_by_time["rob"]
    ).fillna(0)
    wallet_positions_by_time["timestamp_delta"] = (
        wallet_positions_by_time["timestamp"].diff().dt.total_seconds().fillna(0)
    )
    average_by_time = np.average(wallet_positions_by_time["rob"], weights=wallet_positions_by_time["timestamp_delta"])
    # adjust the random trader's position to be their average spend
    idx = weth_index & (current_wallet.username == "rob")
    current_wallet.loc[idx, ["position"]] = average_by_time  # type: ignore
    current_wallet.loc[idx, ["hpr"]] = current_wallet.loc[idx, ["pnl"]].astype("float").iloc[0].values / current_wallet.loc[idx, ["position"]].astype("float").iloc[0].values  # type: ignore


def new_row(user, position, pnl, hpr=None):  # pylint: disable=redefined-outer-name
    _new_row = current_wallet.iloc[len(current_wallet) - 1].copy()
    _new_row["username"], _new_row["position"], _new_row["pnl"], _new_row["token_type"] = user, position, pnl, "WETH"
    _new_row["hpr"] = hpr if hpr is not None else _new_row["pnl"] / (Decimal(_new_row["position"]) - _new_row["pnl"])
    return _new_row.to_frame().T


governance_row = new_row("governance", governance_fees, governance_fees, 0)
total_row = new_row(
    "total",
    float(current_wallet["position"].values.sum()),
    current_wallet.loc[current_wallet.token_type.values == "WETH", ["pnl"]].values.sum(),
)
share_price_row = new_row(
    "share price",
    pool_info.lp_share_price.iloc[-1] * Decimal(1e7),
    pool_info.lp_share_price.iloc[-1] * Decimal(1e7) - pool_info.lp_share_price.iloc[0] * Decimal(1e7),
)
current_wallet = pd.concat([current_wallet, governance_row, total_row, share_price_row], ignore_index=True)

# re-index
non_weth_index = (current_wallet.token_type != "WETH") & (current_wallet.position > float(MINIMUM_TRANSACTION_AMOUNT))
weth_index = current_wallet.token_type == "WETH"
not_inf = current_wallet.hpr != np.inf
# convert to float
current_wallet.position = current_wallet.position.astype(float)
current_wallet.pnl = current_wallet.pnl.astype(float)
# add APR
current_wallet["apr"] = Decimal(np.nan)
current_wallet.loc[weth_index & not_inf, ["apr"]] = current_wallet.loc[weth_index & not_inf, ["hpr"]].values * Decimal(
    apr_factor
)

results1 = current_wallet.loc[non_weth_index, exp.display_cols]
results2 = current_wallet.loc[weth_index, exp.display_cols_with_hpr]
experiment_stats = {"total_volume": float(total_volume)}

# log results
if RUNNING_WANDB:
    wandb.log({"results1": wandb.Table(dataframe=results1)})
    wandb.log({"results2": wandb.Table(dataframe=results2)})
    wandb.log({"wallet_positions": wandb.Table(dataframe=wallet_positions)})
    wandb.log({"current_wallet": wandb.Table(dataframe=current_wallet)})
    wandb.log({"pool_info": wandb.Table(dataframe=pool_info)})
    wandb.log({"lp_value": results2.loc[results2.username == "larry", "pnl"].values[0]})
else:
    results1.to_parquet("results1.parquet", index=False)
    results2.to_parquet("results2.parquet", index=False)
    wallet_positions.to_parquet("wallet_positions.parquet", index=False)
    current_wallet.to_parquet("current_wallet.parquet", index=False)
    pool_info.to_parquet("pool_info.parquet", index=False)
    with open("experiment_stats.json", "w") as f:
        json.dump(experiment_stats, f)
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
    display(
        results2.style.format(
            subset=[col for col in results2.columns if results2.dtypes[col] == "float64" and col not in ["hpr", "apr"]],
            formatter="{:" + exp.float_fmt + "}",
        )
        .hide(axis="index")
        .format(
            subset=["hpr", "apr"],
            formatter="{:.2%}",
        )
    )
else:
    print(results2)

# %%
pool_info = interactive_hyperdrive.get_pool_state()
pool_info.fixed_rate = pool_info.fixed_rate.astype(float)
pool_info.variable_rate = pool_info.variable_rate.astype(float)
# plot rates
if RUNNING_INTERACTIVE:
    from matplotlib import ticker

    pool_info.plot(
        x="block_number",
        y=["fixed_rate", "variable_rate"],
    )
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    plt.show()

# %%
# get wallet deltas
wallet_deltas = get_wallet_deltas(session=interactive_hyperdrive.db_session)

# %%
# import time
# start_time = time.time()
# from chainsync.db.hyperdrive.interface import get_wallet_pnl
# session = interactive_hyperdrive.db_session

# wallet_pnl = get_wallet_pnl(session=session)
# print(f"done in {time.time() - start_time:.2f} seconds")

# %%
# calc wallet pnl
# start_time = time.time()
# from chainsync.analysis.calc_pnl import calc_closeout_pnl
# current_wallet = deepcopy(interactive_hyperdrive.get_current_wallet())
# pnl = calc_closeout_pnl(
#     current_wallet=current_wallet,
#     hyperdrive_contract=interactive_hyperdrive.interface.hyperdrive_contract,
#     hyperdrive_interface=interactive_hyperdrive.interface,
# )

# %%
# clear resources
if not RUNNING_INTERACTIVE:
    chain.cleanup()

# %%
