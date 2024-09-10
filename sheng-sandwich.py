from fixedpointmath import FixedPoint

from agent0 import LocalChain, LocalHyperdrive

TIME_STRETCH = FixedPoint(0.5)
SECONDS_IN_A_YEAR = 365 * 24 * 60 * 60
TARGET_AGENT_CONTRIBUTION = FixedPoint(400_000)
BAD_AGENT_CONTRIBUTION = FixedPoint(100_000)

with LocalChain() as chain:
    pool = LocalHyperdrive(
        chain=chain,
        config=LocalHyperdrive.Config(
            factory_max_circuit_breaker_delta=FixedPoint(2e3),
            factory_max_fixed_apr=FixedPoint(10),
            circuit_breaker_delta=FixedPoint(1e3),
            initial_fixed_apr=TIME_STRETCH,
            initial_time_stretch_apr=TIME_STRETCH,
            initial_liquidity=FixedPoint(1_000),
            curve_fee=FixedPoint(0.01),
            flat_fee=FixedPoint(0.0005),
            initial_variable_rate=TIME_STRETCH,
        ),
    )

    target_agent = chain.init_agent(
        eth=FixedPoint(10),
        base=TARGET_AGENT_CONTRIBUTION,
        pool=pool,
        name="target",
    )
    bad_agent = chain.init_agent(
        eth=FixedPoint(10),
        base=FixedPoint(100_000_000),
        pool=pool,
        name="bad",
    )

    # Target adds lp
    target_lp_event = target_agent.add_liquidity(target_agent.get_wallet().balance.amount)

    # Bad agent opens a max short, adds lp, and opens a max long
    max_short = pool.interface.calc_max_short(budget=bad_agent.get_wallet().balance.amount)
    bad_agent.open_short(max_short)
    bad_agent.add_liquidity(BAD_AGENT_CONTRIBUTION)
    max_long = pool.interface.calc_max_long(budget=bad_agent.get_wallet().balance.amount)
    bad_agent.open_long(max_long)

    # Advance position duration
    chain.advance_time(pool.config.position_duration, create_checkpoints=True)

    # target removes liquidity
    target_agent.remove_liquidity(target_agent.get_wallet().lp_tokens)

    # close all bad agent's positions
    long = bad_agent.get_longs()[0]
    bad_agent.close_long(maturity_time=long.maturity_time, bonds=long.balance)
    short = bad_agent.get_shorts()[0]
    bad_agent.close_short(maturity_time=short.maturity_time, bonds=short.balance)
    bad_agent.remove_liquidity(bad_agent.get_wallet().lp_tokens)

    # Calculate total apr
    bad_agent_trade_events = bad_agent.get_trade_events()
    bad_agent_total_spent = bad_agent_trade_events[bad_agent_trade_events["base_delta"] < 0]["base_delta"].sum()
    bad_agent_total_pnl = bad_agent.get_positions(show_closed_positions=True)["pnl"].sum()
    bad_agent_apr = bad_agent_total_pnl / (-bad_agent_total_spent) / pool.config.position_duration * SECONDS_IN_A_YEAR

    target_agent_trade_events = target_agent.get_trade_events()
    target_agent_total_spent = target_agent_trade_events[target_agent_trade_events["base_delta"] < 0][
        "base_delta"
    ].sum()
    target_agent_total_pnl = target_agent.get_positions(show_closed_positions=True)["pnl"].sum()
    target_agent_apr = (
        target_agent_total_pnl / (-target_agent_total_spent) / pool.config.position_duration * SECONDS_IN_A_YEAR
    )

    print(f"Target agent APR: {target_agent_apr}")
    print(f"Bad agent APR: {bad_agent_apr}")

    # Runs the dashboard in blocking mode
    chain.run_dashboard(blocking=True)
