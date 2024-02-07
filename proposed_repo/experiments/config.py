"""Master config for experiments."""

from dataclasses import dataclass

from fixedpointmath import FixedPoint

from agent0.hyperdrive.interactive import InteractiveHyperdrive


@dataclass
class Config(InteractiveHyperdrive.Config):
    """This adds few extra fields and overrides some defaults for the interactive config."""

    # experiment times
    experiment_days: int = 365  # 1 year
    position_duration: int = 60 * 60 * 24 * 30  # 1 month
    checkpoint_duration: int = 60 * 60 * 24  # 1 day
    initial_liquidity: FixedPoint = FixedPoint(200_000_000)
    # trading amounts
    daily_volume_percentage_of_liquidity: FixedPoint = FixedPoint("0.10")
    minimum_trade_hold_days: FixedPoint = FixedPoint(0)
    agent_budget: FixedPoint = FixedPoint(200_000_000)
    # rates
    initial_variable_rate: FixedPoint = FixedPoint("0.05")
    initial_fixed_rate: FixedPoint = FixedPoint("0.045")
    # fees
    curve_fee: FixedPoint = FixedPoint("0.0")
    flat_fee: FixedPoint = FixedPoint("0.0")
    governance_lp_fee: FixedPoint = FixedPoint("0.0")
    # misc extra
    num_agents: int = 1
    experiment_id: int = 0
    randseed: int = 1234
    calc_pnl: bool = False
    wandb_init_mode: str = "online"  # "online", "offline", or "disabled"
