# %%
# tests making money from:
# - short up to a high rate
# - add liquidity
# - long back down
# this is really a long-range attack

# %%
import datetime
import itertools
import os
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

def datefmt(seconds):
    unix_epoch = datetime.datetime(1970, 1, 1)  # Unix epoch start date
    date = unix_epoch + datetime.timedelta(seconds=seconds)
    return date.strftime('%Y-%m-%d %H:%M:%S')

def display_pnl(agent0,agent1,verbose=False):
    pos0 = agent0.get_positions()
    pos1 = agent1.get_positions()
    hist0 = agent0.get_trade_events()
    hist1 = agent1.get_trade_events()
    if verbose:
        cols = ["token_type", "token_balance", "unrealized_value", "realized_value", "pnl"]
        print("agent0 positions")
        display(pos0[cols])
        print("agent1 positions")
        display(pos1[cols])
    # calculate spend by summing up only negative values
    spend0 = -hist0.loc[hist0.base_delta < 0, "base_delta"].sum()
    spend1 = -hist1.loc[hist1.base_delta < 0, "base_delta"].sum()
    # calculate return by summing up only positive values
    return0 = hist0.loc[hist0.base_delta > 0, "base_delta"].sum()
    return1 = hist1.loc[hist1.base_delta > 0, "base_delta"].sum()
    unrealized0 = pos0.unrealized_value.sum()
    unrealized1 = pos1.unrealized_value.sum()
    pnl0 = (return0 + unrealized0) - spend0
    pnl1 = (return1 + unrealized1) - spend1
    roi0 = pnl0 / spend0
    roi1 = pnl1 / spend1
    print(f"agent0 PNL: {pnl0} on spend of {spend0} for an ROI of {roi0:.6%}")
    print(f"agent1 PNL: {pnl1} on spend of {spend1} for an ROI of {roi1:.6%}")

    pnl_table = pd.DataFrame({"pnl": [pnl0, pnl1], "spend": [spend0, spend1], "roi": [roi0, roi1]}, index=["lp", "attacker"])
    if RUNNING_INTERACTIVE:
        display(pnl_table.style.format(subset=["pnl", "spend"], formatter="{:,.3f}").format(subset=["roi"], formatter="{:,.5%}"))
    return pos0, pos1, pnl0, pnl1, spend0, spend1, roi0, roi1

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

def calc_shorts(chain:LocalChain, interface:HyperdriveReadWriteInterface):
    # pseudo calc_targeted_short
    chain.save_snapshot()
    price, rate, base_traded, bonds_traded_short, trade_size = trade(interface, agent1, 1, interface.calc_max_long(budget=FixedPoint(1e18)), interface.calc_max_short(budget=FixedPoint(1e18)))
    print(f"max long  rate = {float(rate):,.5%}")
    chain.load_snapshot()
    price, rate, base_traded, bonds_traded_short, trade_size = trade(interface, agent1, -1, interface.calc_max_long(budget=FixedPoint(1e18)), interface.calc_max_short(budget=FixedPoint(1e18)))
    print(f"max short rate = {float(rate):,.5%}")
    max_delta = np.floor((float(rate)-TIME_STRETCH_APR)*100)/100
    short_target = min(STARTING_RATE + DELTA, max_delta)
    trade_size = interface.calc_targeted_long(budget=FixedPoint(1e18), target_rate=FixedPoint(short_target))
    bonds_traded_long = 0
    if trade_size > minimum_transaction_amount:
        price, rate, base_traded, bonds_traded_long = trade_long(interface, agent1, trade_size)
    chain.load_snapshot()
    return bonds_traded_short, bonds_traded_long, short_target

# %%

# sourcery skip: merge-list-append, move-assign-in-block
YEAR_IN_SECONDS = 31_536_000
TIME_STRETCH_LIST = [0.1]
DELTA_LIST = [1e18]
STARTING_RATE_LIST = list(TIME_STRETCH_LIST)
TRADE_PORTION = -1  # 100% short
STEPS = 12

# %%
# liquidity = FixedPoint(10_000)
# minimum_share_reserves = max(liquidity*FixedPoint(1/10_000),FixedPoint(1))
start_liq = FixedPoint(100_000)
liquidity = FixedPoint(os.environ["LIQUIDITY"]) if "LIQUIDITY" in os.environ else FixedPoint(140)*start_liq/FixedPoint(100)

# %%
results = []
for TIME_STRETCH_APR, STARTING_RATE, DELTA in itertools.product(TIME_STRETCH_LIST, STARTING_RATE_LIST, DELTA_LIST):
    print(f"=== {TIME_STRETCH_APR=} {DELTA=} ===")
    chain = LocalChain(LocalChain.Config(chain_port=10_000, db_port=10_001))
    interactive_config = LocalHyperdrive.Config(
        position_duration=YEAR_IN_SECONDS,  # 1 year term
        governance_lp_fee=FixedPoint(0),
        curve_fee=FixedPoint(0),
        flat_fee=FixedPoint(0),
        initial_liquidity=start_liq,
        initial_fixed_apr=FixedPoint(TIME_STRETCH_APR),
        initial_time_stretch_apr=FixedPoint(TIME_STRETCH_APR),
        factory_min_fixed_apr=FixedPoint(0.001),
        factory_max_fixed_apr=FixedPoint(1000),
        factory_min_time_stretch_apr=FixedPoint(0.001),
        factory_max_time_stretch_apr=FixedPoint(1000),
        minimum_share_reserves=FixedPoint(0.0001),
        factory_max_circuit_breaker_delta=FixedPoint(1000),
        circuit_breaker_delta=FixedPoint(1e3),
        initial_variable_rate=FixedPoint(STARTING_RATE),
    )
    hyperdrive:LocalHyperdrive = LocalHyperdrive(chain, interactive_config)
    hyperdrive.interface.calc_spot_rate()

    # %%
    # set up agents
    agent0 = chain.init_agent(base=FixedPoint(1e18), eth=FixedPoint(1e18), private_key=chain.get_deployer_account_private_key(), pool=hyperdrive)
    agent1 = chain.init_agent(base=FixedPoint(1e18), eth=FixedPoint(1e18), pool=hyperdrive)
    interface = hyperdrive.interface
    minimum_transaction_amount = interface.pool_config.minimum_transaction_amount
    all_results = pd.DataFrame()
    pos0 = pos1 = pnl0 = pnl1 = spend0 = spend1 = roi0 = roi1 = None
    records = []

    # %%
    # agent0 longs to STARTING_RATE
    if STARTING_RATE > TIME_STRETCH_APR:
        trade_size = interface.calc_targeted_long(budget=FixedPoint(1e18), target_rate=FixedPoint(STARTING_RATE))
        price, rate, base_traded, bonds_traded_long = trade_long(interface, agent0, trade_size)

    # %%
    bonds_traded_short, bonds_traded_long, short_target = calc_shorts(chain=chain, interface=interface)

    # %%
    # short to TIME_STRETCH_APR + DELTA
    print(f"before trading, the share adjustment is {float(interface.current_pool_state.pool_info.share_adjustment):,.0f}")
    trade_size = - ((bonds_traded_short or 0) + (bonds_traded_long or 0))  # type: ignore
    print(f"{trade_size=}")
    price, rate, base_traded, bonds_traded = trade_short(interface, agent1, trade_size)
    records.append((1, "short", interface.calc_effective_share_reserves(), liquidity, None, None, TRADE_PORTION, price, rate, TIME_STRETCH_APR, interface.current_pool_state.pool_info.bond_reserves, interface.current_pool_state.pool_info.share_reserves))
    print(f"after short, the rate is {float(rate):.3%}, bonds traded is {float(bonds_traded):.4f}, share adjustment is {float(interface.current_pool_state.pool_info.share_adjustment):,.0f}")

    # %%
    # add liquidity
    price, rate = trade_liq(interface, agent1, liquidity)
    print(f"added {liquidity} base of liquidity")
    records.append((2, "addliq", interface.calc_effective_share_reserves(), liquidity, None, None, TRADE_PORTION, price, rate, TIME_STRETCH_APR, interface.current_pool_state.pool_info.bond_reserves, interface.current_pool_state.pool_info.share_reserves))
    print(f"after adding liquidity, share adjustment is {float(interface.current_pool_state.pool_info.share_adjustment):,.0f}")

    # %%
    # long to TIME_STRETCH_APR
    trade_size = interface.calc_targeted_long(budget=FixedPoint(1e18), target_rate=FixedPoint(TIME_STRETCH_APR))
    price, rate, base_traded, bonds_traded = trade_long(interface, agent1, trade_size)
    print(f"after long, the rate is {float(rate):.3%}, bonds traded is {float(bonds_traded):.4f}, share adjustment is {float(interface.current_pool_state.pool_info.share_adjustment):,.0f}")

    # %%
    # advance time
    print(f"{chain.block_time()=}")
    print(f"block = {chain.block_number()} timestamp = {datefmt(chain.block_time())} vault_share_price = {interface.current_pool_state.pool_info.vault_share_price}")
    print("advancing time by 1 year..", end="")
    chain.advance_time(datetime.timedelta(seconds=60*60*24*365), create_checkpoints=False)
    print(". ")
    print(f"block = {chain.block_number()} timestamp = {datefmt(chain.block_time())} vault_share_price = {interface.current_pool_state.pool_info.vault_share_price}")

    # %%
    # display pnl before closing positions
    print("before closing positions")
    pos0, pos1, pnl0, pnl1, spend0, spend1, roi0, roi1 = display_pnl(agent0, agent1)

    # %%
    # close positions
    print("after closing positions")
    agent1.close_long(maturity_time=int(pos1.loc[pos1.token_type=="LONG","maturity_time"].values[0]), bonds=FixedPoint(pos1.loc[pos1.token_type=="LONG","token_balance"].values[0]))
    agent1.close_short(maturity_time=int(pos1.loc[pos1.token_type=="SHORT","maturity_time"].values[0]), bonds=FixedPoint(pos1.loc[pos1.token_type=="SHORT","token_balance"].values[0]))
    # agent1.remove_liquidity(shares=FixedPoint(pos1.loc[pos1.token_type=="LP","token_balance"].values[0]))
    # agent0.remove_liquidity(shares=FixedPoint(pos0.loc[pos0.token_type=="LP","token_balance"].values[0]) - 2*interactive_config.minimum_share_reserves)
    pos0, pos1, pnl0, pnl1, spend0, spend1, roi0, roi1 = display_pnl(agent0, agent1)

    # %%
    # stop the steal!
    print(f"attack steals {(roi1-roi0)/2:.5%} from agent0")

    # %%
    # short + add liquidity again
    print("before second short + add liquidity")
    bonds_traded_short, bonds_traded_long, short_target = calc_shorts(chain=chain, interface=interface)

    # short to TIME_STRETCH_APR + DELTA
    print(f"before trading, the share adjustment is {float(interface.current_pool_state.pool_info.share_adjustment):,.0f}")
    trade_size = - ((bonds_traded_short or 0) + (bonds_traded_long or 0))  # type: ignore
    print(f"{trade_size=}")
    price, rate, base_traded, bonds_traded = trade_short(interface, agent1, trade_size)
    records.append((1, "short", interface.calc_effective_share_reserves(), liquidity, None, None, TRADE_PORTION, price, rate, TIME_STRETCH_APR, interface.current_pool_state.pool_info.bond_reserves, interface.current_pool_state.pool_info.share_reserves))
    print(f"after short, the rate is {float(rate):.3%}, bonds traded is {float(bonds_traded):.4f}, share adjustment is {float(interface.current_pool_state.pool_info.share_adjustment):,.0f}")

    # add liquidity
    price, rate = trade_liq(interface, agent1, liquidity)
    print(f"added {liquidity} base of liquidity")
    records.append((2, "addliq", interface.calc_effective_share_reserves(), liquidity, None, None, TRADE_PORTION, price, rate, TIME_STRETCH_APR, interface.current_pool_state.pool_info.bond_reserves, interface.current_pool_state.pool_info.share_reserves))
    print(f"after adding liquidity, share adjustment is {float(interface.current_pool_state.pool_info.share_adjustment):,.0f}")

    print("after second short + add liquidity")
    bonds_traded_short, bonds_traded_long, short_target = calc_shorts(chain=chain, interface=interface)

    # %%
    chain.cleanup()
