# %%
# This script showcases the burrito attack on hyperdrive.
# Key parameters:
#   fixed rate = variable rate = time stretch APR = 0.5 (50%)
#   circuit breaker delta = 1e3
#   position duration = 4 weeks (32 days)
#   target agent (Alice) contribution = 100k
#   bad agent (Celine) contribution = 1M
#
# Steps:
#   1. Alice LPs
#   2. Celine opens a max short (for as much as budget allows)
#   3. Celine LPs (for contribution amount)
#   4. Celine opens a max long (for as much as budget allows)
#   5. Time elapses (x-axis amount in plot)
#   6. Celine closes all positions
#   7. Alice closes all positions

# %
from decimal import Decimal

from fixedpointmath import FixedPoint

from agent0 import LocalChain, LocalHyperdrive
from agent0.core.hyperdrive.interactive.local_hyperdrive_agent import LocalHyperdriveAgent

SECONDS_IN_A_YEAR = 365 * 24 * 60 * 60
SECONDS_IN_A_DAY = 24 * 60 * 60

FIXED_APR = FixedPoint(0.1)
TIME_STRETCH = FixedPoint(0.05)
INITIAL_VARIABLE_RATE = FixedPoint(0.1)
NEW_VARIABLE_RATE = FixedPoint(0.05)
CIRCUIT_BREAKER_DELTA = FixedPoint(1e3)

ALICE_CONTRIBUTION = FixedPoint(50_000)
LONG_BASE = FixedPoint(5_000)
POSITION_DURATION = SECONDS_IN_A_YEAR  # 1 year

data = []

chain = LocalChain()
# Initialize pool
pool = LocalHyperdrive(
    chain=chain,
    config=LocalHyperdrive.Config(
        factory_max_circuit_breaker_delta=FixedPoint(2e3),
        factory_max_fixed_apr=FixedPoint(10),
        circuit_breaker_delta=CIRCUIT_BREAKER_DELTA,
        initial_fixed_apr=FIXED_APR,
        initial_time_stretch_apr=TIME_STRETCH,
        initial_liquidity=FixedPoint(100),
        # curve_fee=FixedPoint(0.01),
        # flat_fee=FixedPoint(0.0005),
        curve_fee=FixedPoint(0),
        flat_fee=FixedPoint(0),
        governance_lp_fee=FixedPoint(0),
        governance_zombie_fee=FixedPoint(0),
        initial_variable_rate=INITIAL_VARIABLE_RATE,
        position_duration=POSITION_DURATION,
    ),
)

alice = chain.init_agent(
    eth=FixedPoint(10),
    base=ALICE_CONTRIBUTION,
    pool=pool,
    name="alice",
)

celine = chain.init_agent(
    eth=FixedPoint(10),
    base=ALICE_CONTRIBUTION,
    pool=pool,
    name="celine",
)

print(f"starting lpTotalSupply: {pool.interface.current_pool_state.pool_info.lp_total_supply}")
# Alice adds liquidity
alice_contribution = FixedPoint(50_000)
print(f"Alice adds {alice_contribution} liquidity")
alice.add_liquidity(alice_contribution)
print(f"lpTotalSupply after add liquidity: {pool.interface.current_pool_state.pool_info.lp_total_supply}")
print(f"shareReserves after add liquidity: {pool.interface.current_pool_state.pool_info.share_reserves}")

# Celine opens a long with base
long_base = FixedPoint(5_000)
print(f"celine opens a long with {long_base} at 10%")
open_long_event = celine.open_long(long_base)
print(f"lpSharePrice after open long: {pool.interface.current_pool_state.pool_info.lp_share_price}")
print(f"lpTotalSupply after open long: {pool.interface.current_pool_state.pool_info.lp_total_supply}")
print(f"shareReserves after open long: {pool.interface.current_pool_state.pool_info.share_reserves}")
print(f"celine gets {open_long_event.bond_amount} bonds")
net_apr = open_long_event.bond_amount / long_base - FixedPoint(1)
print(f"celine's net apr is {net_apr}")

# Advance the time 60 days and accrue interest.
# for seconds in range(0, SECONDS_IN_A_DAY * 60, SECONDS_IN_A_DAY):
#     # DB updates with pnl information after each advance time
#     chain.advance_time(SECONDS_IN_A_DAY, create_checkpoints=True)
#     print(f"advancing time, day {seconds / SECONDS_IN_A_DAY} with lpSharePrice {pool.interface.current_pool_state.pool_info.lp_share_price}")
# print(f"60 days elapse at {INITIAL_VARIABLE_RATE*100}% APY")
# print(f"lpSharePrice after 60 days: {pool.interface.current_pool_state.pool_info.lp_share_price}")
# print(f"lpTotalSupply after 60 days: {pool.interface.current_pool_state.pool_info.lp_total_supply}")
# print(f"shareReserves after 60 days: {pool.interface.current_pool_state.pool_info.share_reserves}")

remove_liquidity_event = alice.remove_liquidity(alice.get_wallet().lp_tokens)
print(f"lpShares withdrawn: {remove_liquidity_event.lp_amount}")
print(f"base withdrawn: {remove_liquidity_event.amount}")
lp_shares_redeemed = remove_liquidity_event.lp_amount - remove_liquidity_event.withdrawal_share_amount
print(f"lpShares redeemed: {lp_shares_redeemed}")
price_per_share = remove_liquidity_event.amount / lp_shares_redeemed
print(f"price per share redeemed: {price_per_share}")
price_per_share_total = (remove_liquidity_event.amount + remove_liquidity_event.withdrawal_share_amount * pool.interface.current_pool_state.pool_info.lp_share_price) / remove_liquidity_event.lp_amount
print(f"price per share full value: {price_per_share_total}")
print(f"withdrawal shares: {remove_liquidity_event.withdrawal_share_amount}")
lp_share_price = pool.interface.current_pool_state.pool_info.lp_share_price
print(f"lpSharePrice after remove liquidity: {lp_share_price}")
print(f"lpTotalSupply after remove liquidity: {pool.interface.current_pool_state.pool_info.lp_total_supply}")
print(f"shareReserves after remove liquidity: {pool.interface.current_pool_state.pool_info.share_reserves}")

# Advance the time and accrue interest at a lower rate.
new_variable_rate = FixedPoint(0.05)
# pool.set_variable_rate(new_variable_rate)
# for seconds in range(0, SECONDS_IN_A_DAY * 7, SECONDS_IN_A_DAY):
#     # DB updates with pnl information after each advance time
#     chain.advance_time(SECONDS_IN_A_DAY, create_checkpoints=True)
#     print(f"advancing time, day {seconds / SECONDS_IN_A_DAY} with lpSharePrice {pool.interface.current_pool_state.pool_info.lp_share_price}")
# lp_share_price = pool.interface.current_pool_state.pool_info.lp_share_price
# print(f"lpSharePrice after 7 days elapse at {new_variable_rate*100}% APY: {lp_share_price}")

# Alice adds liquidity to match Celine's long.
extra_liquidity = FixedPoint(23500)
print(f"alice adds {extra_liquidity} extra liquidity")
alice.add_liquidity(extra_liquidity)
print(f"lpSharePrice after add liquidity: {pool.interface.current_pool_state.pool_info.lp_share_price}")
print(f"lpTotalSupply after add liquidity: {pool.interface.current_pool_state.pool_info.lp_total_supply}")
print(f"shareReserves after add liquidity: {pool.interface.current_pool_state.pool_info.share_reserves}")

# Advance the time and accrue interest at a lower rate.
pool.set_variable_rate(FixedPoint(0.05))
# for seconds in range(0, POSITION_DURATION, SECONDS_IN_A_DAY):
for seconds in range(0, SECONDS_IN_A_DAY * 15, SECONDS_IN_A_DAY):
    # DB updates with pnl information after each advance time
    chain.advance_time(SECONDS_IN_A_DAY, create_checkpoints=True)
    print(f"advancing time, day {seconds / SECONDS_IN_A_DAY} with lpSharePrice {pool.interface.current_pool_state.pool_info.lp_share_price}")
lp_share_price = pool.interface.current_pool_state.pool_info.lp_share_price
print(f"lpSharePrice after 15 more days elapse at {new_variable_rate*100}% APY: {lp_share_price}")