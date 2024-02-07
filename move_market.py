# %%
from fixedpointmath import FixedPoint

from agent0.hyperdrive.interactive import InteractiveHyperdrive, LocalChain

LIQUDITY = 10_000_000  # 10 million
config = InteractiveHyperdrive.Config(
    position_duration=60 * 60 * 24 * 365,  # 365 days,
    checkpoint_duration=60 * 60 * 24,  # 1 day
    initial_liquidity=FixedPoint(LIQUDITY),
    initial_fixed_apr=FixedPoint(0.035),
    initial_variable_rate=FixedPoint(0.035),
    curve_fee=FixedPoint(0.001),
    flat_fee=FixedPoint(0),
    governance_lp_fee=FixedPoint(0),
    calc_pnl=False,
)
chain = LocalChain(LocalChain.Config(db_port=5_433, chain_port=10_000))
interactive_hyperdrive = InteractiveHyperdrive(chain, config)
agent = interactive_hyperdrive.init_agent(base=FixedPoint(LIQUDITY))

# %%
# measure trade impact
fixed_rate_before = interactive_hyperdrive.interface.calc_fixed_rate()
print(f"fixed rate before trade: {fixed_rate_before}")
agent.open_long(FixedPoint(10_000))
fixed_rate_after = interactive_hyperdrive.interface.calc_fixed_rate()
print(f"fixed rate after  trade: {fixed_rate_after}")
trade_impact = float(fixed_rate_before - fixed_rate_after)
print(f"trade impact is {float():.4%}")

# %%