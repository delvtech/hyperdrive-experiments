"""Simulate an LP return.


We simulate an LP position in a pool with random trades that are profitable half the time.
The average daily trading volume is 10% of the total pool liquidity.
The variable rate is chosen to have an average of 4.5% return, about equal to 2023 staking returns.
LPs backing trades can result in negative returns, which is mitigated by profits from the yield source and fees.
"""
from __future__ import annotations

from dataclasses import dataclass


from agent0.hyperdrive.interactive import InteractiveHyperdrive, LocalChain

import numpy as np
from fixedpointmath import FixedPoint


@dataclass
class ExperimentConfig:  # pylint: disable=too-many-instance-attributes,missing-class-docstring
    experiment_id: int = 0
    experiment_run: int = 60 * 60 * 24 * 365 # 1 year
    position_duration: int = 60 * 60 * 24 * 73 # 5 times per year
    checkpoint: int = 60 * 60 * 24 * 7 # 1 week
    daily_volume_percentage_of_liquidity: float = 0.05  # 1%
    minimum_trade_days: float = 0  # minimum number of days to keep a trade open
    amount_of_liquidity: int = 10_000_000
    max_trades_per_day: int = 10
    variable_rate: FixedPoint = FixedPoint("0.045")
    fixed_rate: FixedPoint = FixedPoint("0.045")
    curve_fee: FixedPoint = FixedPoint("0")
    flat_fee: FixedPoint = FixedPoint("0")
    governance_fee: FixedPoint = FixedPoint("0")
    randseed: int = 0
    term_seconds: int = 0

exp = ExperimentConfig()
rng = np.random.default_rng(seed=exp.randseed)

chain = LocalChain(LocalChain.Config(db_port=5_433, chain_port=10_000))

interactive_config = InteractiveHyperdrive.Config(
    position_duration=exp.experiment_run_seconds,
    checkpoint_duration