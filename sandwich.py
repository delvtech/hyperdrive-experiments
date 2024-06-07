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
TIME_STRETCH_APR = 0.05
INITIAL_RATE = 0.01
UPPER_RATE = 0.085
TRADE_PORTION = -1  # 100% short
STEPS = 12

chain = LocalChain(LocalChain.Config(chain_port=10_000, db_port=10_001))

# %%
# liquidity = FixedPoint(10_000)
# minimum_share_reserves = max(liquidity*FixedPoint(1/10_000),FixedPoint(1))
liquidity = FixedPoint(100)
starting_amount = FixedPoint(13)
minimum_share_reserves = max(liquidity*FixedPoint(1/10_000),FixedPoint(0.001))

# %%
interactive_config = LocalHyperdrive.Config(
    position_duration=YEAR_IN_SECONDS,  # 1 year term
    governance_lp_fee=FixedPoint(0),
    curve_fee=FixedPoint(0),
    flat_fee=FixedPoint(0),
    initial_liquidity=FixedPoint(10),
    initial_fixed_apr=FixedPoint(TIME_STRETCH_APR),
    initial_time_stretch_apr=FixedPoint(TIME_STRETCH_APR),
    factory_min_fixed_apr=FixedPoint(0.001),
    factory_max_fixed_apr=FixedPoint(1000),
    factory_min_time_stretch_apr=FixedPoint(0.001),
    factory_max_time_stretch_apr=FixedPoint(1000),
    minimum_share_reserves=FixedPoint(0.0001),
    factory_max_circuit_breaker_delta=FixedPoint(1000),
    circuit_breaker_delta=FixedPoint(10),
    initial_variable_rate=FixedPoint(INITIAL_RATE),
)
hyperdrive:LocalHyperdrive = LocalHyperdrive(chain, interactive_config)

# %%
# set up agents
deployer_privkey = chain.get_deployer_account_private_key()
agent0 = chain.init_agent(base=FixedPoint(1e18), eth=FixedPoint(1e18), private_key=deployer_privkey, pool=hyperdrive)
agent1 = chain.init_agent(base=FixedPoint(1e18), eth=FixedPoint(1e18), pool=hyperdrive)
interface = hyperdrive.interface
all_results = pd.DataFrame()
records = []

# %%
# first long to INITIAL_RATE
trade_size = interface.calc_targeted_long(budget=FixedPoint(1e18), target_rate=FixedPoint(INITIAL_RATE))
price, rate, base_traded, bonds_traded = trade_long(interface, agent0, trade_size)
records.append((1, "long", interface.calc_effective_share_reserves(), liquidity, None, None, TRADE_PORTION, price, rate, TIME_STRETCH_APR, interface.current_pool_state.pool_info.bond_reserves, interface.current_pool_state.pool_info.share_reserves))
print(f"after first long, the rate is {rate}, bonds traded is {bonds_traded}")

# %%
# max short
# max_short_two = interface.calc_max_short(budget=FixedPoint(1e18))
# max_long_two = interface.calc_max_long(budget=FixedPoint(1e18))
# price, rate, base_traded, bonds_traded, trade_size = trade(interface, agent1, -1, max_long_two, max_short_two)
# records.append((2, "short", interface.calc_effective_share_reserves(), trade_size, base_traded, bonds_traded, TRADE_PORTION, price, rate, TIME_STRETCH_APR, interface.current_pool_state.pool_info.bond_reserves, interface.current_pool_state.pool_info.share_reserves))
# print(f"after max short, the rate is {rate}, bonds traded is {bonds_traded}")

# # %%
# # second long to UPPER_RATE
# trade_size = interface.calc_targeted_long(budget=FixedPoint(1e18), target_rate=FixedPoint(UPPER_RATE))
# price, rate, base_traded, bonds_traded = trade_long(interface, agent1, trade_size)
# records.append((3, "long", interface.calc_effective_share_reserves(), liquidity, None, None, TRADE_PORTION, price, rate, TIME_STRETCH_APR, interface.current_pool_state.pool_info.bond_reserves, interface.current_pool_state.pool_info.share_reserves))
# print(f"after second long, the rate is {rate}, bonds traded is {bonds_traded}")

# %%
# short to UPPER_RATE
trade_size = 4.916643575785862144 - 1.630735704071033979
price, rate, base_traded, bonds_traded = trade_short(interface, agent1, trade_size)
records.append((3, "short", interface.calc_effective_share_reserves(), liquidity, None, None, TRADE_PORTION, price, rate, TIME_STRETCH_APR, interface.current_pool_state.pool_info.bond_reserves, interface.current_pool_state.pool_info.share_reserves))
print(f"after short, the rate is {rate}, bonds traded is {bonds_traded}")

# %%
# add liquidity
price, rate = trade_liq(interface, agent1, liquidity)
records.append((4, "addliq", interface.calc_effective_share_reserves(), liquidity, None, None, TRADE_PORTION, price, rate, TIME_STRETCH_APR, interface.current_pool_state.pool_info.bond_reserves, interface.current_pool_state.pool_info.share_reserves))

# %%
# third long to INITIAL_RATE
trade_size = interface.calc_targeted_long(budget=FixedPoint(1e18), target_rate=FixedPoint(INITIAL_RATE))
price, rate, base_traded, bonds_traded = trade_long(interface, agent1, trade_size)
print(f"after third long, the rate is {rate}, bonds traded is {bonds_traded}")

# %%
# check agent PNLs
# agent0_wallet = agent0.get_wallet()
# agent1_wallet = agent1.get_wallet()
# print(f"agent0 PNL: {agent0_wallet}")
# print(f"agent1 PNL: {agent1_wallet}")
pos0 = agent0.get_positions()
pos1 = agent1.get_positions()
cols = ["token_type", "token_balance", "unrealized_value", "realized_value", "pnl"]
print("agent0 positions")
display(pos0[cols])
print("agent1 positions")
display(pos1[cols])
pnl0 = pos0.pnl.sum()
pnl1 = pos1.pnl.sum()
print(f"agent0 PNL: {pnl0}")
print(f"agent1 PNL: {pnl1}")

# %%
