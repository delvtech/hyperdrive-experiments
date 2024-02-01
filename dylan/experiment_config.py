"""Specification for the experiment config."""

from dataclasses import dataclass

from fixedpointmath import FixedPoint


## Experiment settings
@dataclass
class ExperimentConfig:
    # experiment times
    experiment_days: int = 60 * 60 * 24 * 365  # 1 year
    position_duration: int = 60 * 60 * 24 * 30  # 1 month
    checkpoint_duration: int = 60 * 60 * 24 * 7  # 1 week
    initial_liquidity: FixedPoint = FixedPoint(2 * 10**8)
    # trading amounts
    daily_volume_percentage_of_liquidity: float = 0.10
    opens_per_day: int = 6  # how often to open a trade
    minimum_trade_hold_days: int = 1
    agent_budget: FixedPoint = FixedPoint(10_000_000)
    # rates
    variable_rate: FixedPoint = FixedPoint("0.045")
    fixed_rate: FixedPoint = FixedPoint("0.045")
    # fees
    curve_fee: FixedPoint = FixedPoint("0.0")
    flat_fee: FixedPoint = FixedPoint("0.0")
    governance_fee: FixedPoint = FixedPoint("0.0")
    # misc extra
    experiment_id: int = 0
    randseed: int = 1234
