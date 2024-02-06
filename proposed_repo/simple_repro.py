"""Run one trade."""
from __future__ import annotations

from dataclasses import dataclass
from pprint import pprint

import numpy as np
from fixedpointmath import FixedPoint

from agent0.hyperdrive.interactive import InteractiveHyperdrive, LocalChain


@dataclass
class RandomConfig:
    experiment_days: int = 30
    position_duration: int = 60 * 60 * 24 * 7  # 1 week
    checkpoint_duration: int = 60 * 60 * 24  # 1 day
    initial_liquidity: FixedPoint = FixedPoint(200_000_000)
    # trading amounts
    agent_budget: FixedPoint = FixedPoint(10_000_000)
    # rates
    fixed_rate: FixedPoint = FixedPoint("0.05")
    variable_rate: FixedPoint = FixedPoint("0.01")
    # fees
    curve_fee: FixedPoint = FixedPoint("0.00")
    flat_fee: FixedPoint = FixedPoint("0.00")
    governance_fee: FixedPoint = FixedPoint("0.0")
    # misc extra
    randseed: int = 1234


exp_config = RandomConfig()

## Interactive Hyperdrive config has a subset of experiment config
hyperdrive_config = InteractiveHyperdrive.Config(
    position_duration=exp_config.position_duration,
    checkpoint_duration=exp_config.checkpoint_duration,
    initial_liquidity=exp_config.initial_liquidity,
    initial_fixed_apr=exp_config.fixed_rate,
    initial_variable_rate=exp_config.variable_rate,
    curve_fee=exp_config.curve_fee,
    flat_fee=exp_config.flat_fee,
    governance_lp_fee=exp_config.governance_fee,
    calc_pnl=False,
)

## Initialize primary objects
rng = np.random.default_rng(seed=exp_config.randseed)
chain = LocalChain(LocalChain.Config(db_port=5_433, chain_port=10_000))
interactive_hyperdrive = InteractiveHyperdrive(chain, hyperdrive_config)

## Do the trade
agent = interactive_hyperdrive.init_agent(base=exp_config.agent_budget, name=f"agent_0")
trade_amount = FixedPoint(interactive_hyperdrive.interface.pool_config.minimum_transaction_amount + 1)
agent.open_long(FixedPoint(interactive_hyperdrive.interface.pool_config.minimum_transaction_amount + 1))

chain.cleanup()
