"""Specification for the experiment config."""

from dataclasses import dataclass

from fixedpointmath import FixedPoint


@dataclass
class LpPnlConfig:
    # experiment times
    experiment_days: int = 365  # 1 year
    position_duration: int = 60 * 60 * 24 * 30  # 1 month
    checkpoint_duration: int = 60 * 60 * 24  # 1 day
    initial_liquidity: FixedPoint = FixedPoint(200_000_000)
    # trading amounts
    daily_volume_percentage_of_liquidity: FixedPoint = FixedPoint("0.10")
    opens_per_day: FixedPoint = FixedPoint(1)  # how often to open a trade
    minimum_trade_hold_days: FixedPoint = FixedPoint(1)
    agent_budget: FixedPoint = FixedPoint(10_000_000_000)
    # rates
    variable_rate: FixedPoint = FixedPoint("0.045")
    fixed_rate: FixedPoint = FixedPoint("0.045")
    # fees
    curve_fee: FixedPoint = FixedPoint("0.0")
    flat_fee: FixedPoint = FixedPoint("0.0")
    governance_fee: FixedPoint = FixedPoint("0.0")
    # misc extra
    num_agents: int = 1
    experiment_id: int = 0
    randseed: int = 1234
    wandb_init_mode: str = "online"
