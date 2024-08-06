# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fixedpointmath import FixedPoint

from agent0.core.hyperdrive.interactive import LocalChain, LocalHyperdrive
from agent0.core.hyperdrive.interactive.local_hyperdrive_agent import LocalHyperdriveAgent
from agent0.ethpy.hyperdrive import HyperdriveReadInterface
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
# setup
# sourcery skip: merge-list-append, move-assign-in-block
YEAR_IN_SECONDS = 31_536_000
TIME_STRETCH_APR = 0.05
LIQUIDITY = FixedPoint(100)

chain = LocalChain(LocalChain.Config(chain_port=10_000, db_port=10_001))
interactive_config = LocalHyperdrive.Config(
    position_duration=YEAR_IN_SECONDS,  # 1 year term
    governance_lp_fee=FixedPoint(0.1),
    curve_fee=FixedPoint(0.01),
    flat_fee=FixedPoint(0),
    initial_liquidity=LIQUIDITY,
    initial_fixed_apr=FixedPoint(TIME_STRETCH_APR),
    initial_time_stretch_apr=FixedPoint(TIME_STRETCH_APR),
    factory_min_fixed_apr=FixedPoint(0.001),
    factory_max_fixed_apr=FixedPoint(1000),
    factory_min_time_stretch_apr=FixedPoint(0.001),
    factory_max_time_stretch_apr=FixedPoint(1000),
    minimum_share_reserves=max(LIQUIDITY*FixedPoint(1/10_000),FixedPoint(0.001)),
)
hyperdrive = LocalHyperdrive(chain, interactive_config)
agent0:LocalHyperdriveAgent = chain.init_agent(base=FixedPoint(1e18), eth=FixedPoint(1e18), pool=hyperdrive)
agent1:LocalHyperdriveAgent = chain.init_agent(base=FixedPoint(1e18), eth=FixedPoint(1e18), pool=hyperdrive)
lp:LocalHyperdriveAgent = chain.init_agent(base=FixedPoint(1e18), eth=FixedPoint(1e18), pool=hyperdrive, private_key=chain.get_deployer_account_private_key())
interface:HyperdriveReadInterface = hyperdrive.interface

# %%
# define functions
def report_stats(interface:HyperdriveReadInterface):
    # FORMULA ONE: lp_allocation = share_reserves 
    # FORMULA TWO: lp_allocation = long_exposure (lp_short_positions) + lp_idle_capital
    pool_state = interface.current_pool_state
    pool_info = pool_state.pool_info
    # 🏎️
    formula_one = pool_info.share_reserves - pool_state.pool_config.minimum_share_reserve
    # 🚗
    long_exposure = pool_info.long_exposure
    idle_capital = interface.get_idle_shares(pool_state=pool_state)
    formula_two = long_exposure + idle_capital
    # 🏁
    assert formula_one == formula_two, f"share_reserves != long_exposure + idle_capital: {formula_one} != {formula_two}"

# %%
# Let's say we start with a market of 1 LP of 100 base tokens.
# ```
# The total LP supply is 100.
# The exposure is 0.
# The idle capital is 100.
# The LP is given all of the yield source rewards/points.
# pool_share_reserves = 100
# idle_capital_calc = 100 - 0 = 100
# ```

pool_state = interface.current_pool_state
idle_shares = interface.get_idle_shares(pool_state=pool_state)
print(f"beginning {idle_shares=}")
idle_base = idle_shares * interface.calc_spot_price()

#  %%
# we open a long
event_list = agent0.open_long(base=FixedPoint(10))
event = event_list[0] if isinstance(event_list, list) else event_list
print(" open long:")
for d in dir(event):
    if not d.startswith('_'):
        print(f"  {d}: {getattr(event, d)}")
pool_state = interface.current_pool_state
pool_config = pool_state.pool_config
pool_info = pool_state.pool_info
share_price = pool_info.vault_share_price
share_reserves = pool_info.share_reserves
print(f"{share_reserves=}")
base_reserves = share_reserves * share_price
print(f"{base_reserves=}")
long_exposure = pool_info.long_exposure
# long exposure tracks the amount of longs open
# you can think of this as being in units of bonds
# it also equals the number of base that Hyperdrive sets aside to back these bonds
print(f"{long_exposure=}")
idle_shares = interface.get_idle_shares(pool_state=pool_state)
print(f"{idle_shares=}")
idle_base = idle_shares * share_price
print(f"{idle_base=}")
assert base_reserves - pool_config.minimum_share_reserves*share_price == long_exposure + idle_base, f"base_reserves - pool_config.minimum_share_reserves*share_price != long_exposure + idle_base: {base_reserves - pool_config.minimum_share_reserves*share_price} != {long_exposure + idle_base}"
assert share_reserves - pool_config.minimum_share_reserves == long_exposure/share_price + idle_shares, f"share_reserves - pool_config.minimum_share_reserves != long_exposure/share_price + idle_shares: {share_reserves - pool_config.minimum_share_reserves} != {long_exposure/share_price + idle_shares}"
base_earning_points_for_lps = long_exposure + idle_base
short_portion_of_share_reserves = long_exposure/share_price/(long_exposure/share_price + idle_shares)
idle_portion_of_share_reserves = idle_shares/(long_exposure/share_price + idle_shares)
print(f" short portion of share reserves: {float(short_portion_of_share_reserves):>5.1%}")
print(f"  idle portion of share reserves: {float(idle_portion_of_share_reserves):>5.1%}")
lp_present_value = interface.calc_present_value()
print(f"{lp_present_value=}")
lp_points_multiplier = base_earning_points_for_lps / lp_present_value
print(f"lp points multiplier = {lp_points_multiplier} (base_earning_points_for_lps/lp_present_value = {base_earning_points_for_lps}/{lp_present_value})")

# %%
# If a user opens a short position, the LP would open a long position to back the trade.
# ```
# A user shorts 10 base tokens for 1 base token.
# The LP longs 10 base tokens.
# The long exposure is 10.
# The LP's idle capital is 90.
# pool_share_reserves = 100 - 10 = 90
# idle_capital_calc = 90 - 0 = 90
# ```

# do the trade
# share_price = interface.current_pool_state.pool_info.vault_share_price
# bonds_for_10_base = interface.calc_bonds_out_given_shares_in_down(amount_in=FixedPoint(10)/share_price)*(FixedPoint(1) + interface.pool_config.fees.curve)
# agent0.open_short(bonds=bonds_for_10_base)

# after trade
# pool_state = interface.current_pool_state
# long_exposure = pool_state.pool_info.long_exposure
# print(f"long exposure: {long_exposure}")
# idle_capital = interface.get_idle_shares(pool_state=pool_state)
# print(f"idle capital: {idle_capital}")

# %%
# Now what happens if the short position is netted out by another user that opens a long position?

# ```
# A user longs for 10 base tokens.
# Now the market is netted.
# The long position earns fixed yield on 10 base tokens.
# The LP would earn points on their entire position.
# The short position is still earning 10x multiplier on points.
# pool_share_reserves = 100 (because short and long net out)
# idle_capital_calc = 100 - 0 = 100
# ```

# do the trade
# agent1.open_long(base=FixedPoint(10))

# after trade
# pool_state = interface.current_pool_state
# long_exposure = pool_state.pool_info.long_exposure
# print(f"long exposure: {long_exposure}")
# idle_capital = interface.get_idle_shares(pool_state=pool_state)
# print(f"idle capital: {idle_capital}")

# %%