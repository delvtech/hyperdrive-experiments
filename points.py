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
    # FORMULA TWO: lp_allocation = share_reserves 
    # FORMULA THREE: lp_short_positions + lp_idle_capital
    pool_state = interface.current_pool_state
    idle_shares = interface.get_idle_shares(pool_state=pool_state)
    print(f"{idle_shares=}")
    idle_base = idle_shares * interface.calc_spot_price()
    print(f"{idle_base=}")

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
# spot_price = interface.calc_spot_price()
share_price = interface.current_pool_state.pool_info.vault_share_price
bonds_for_10_base = interface.calc_bonds_out_given_shares_in_down(amount_in=FixedPoint(10)/share_price)*(FixedPoint(1) + interface.pool_config.fees.curve)
agent0.open_short(bonds=bonds_for_10_base)

# after trade
pool_state = interface.current_pool_state
long_exposure = pool_state.pool_info.long_exposure
print(f"long exposure: {long_exposure}")
idle_capital = interface.get_idle_shares(pool_state=pool_state)
print(f"idle capital: {idle_capital}")

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

agent1.open_long(base=FixedPoint(10))

# after trade
pool_state = interface.current_pool_state
long_exposure = pool_state.pool_info.long_exposure
print(f"long exposure: {long_exposure}")
idle_capital = interface.get_idle_shares(pool_state=pool_state)
print(f"idle capital: {idle_capital}")

# %%