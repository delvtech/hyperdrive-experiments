"""Simulate an LP return.

We simulate an LP position in a pool with random trades that are profitable half the time.
The average daily trading volume is 10% of the total pool liquidity.
The variable rate is chosen to have an average of 4.5% return, about equal to 2023 staking returns.

Goal: Demonstrate that LPs backing trades can result in negative returns,
which is mitigated by profits from the yield source and fees.
"""
# %%
from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
from fixedpointmath import FixedPoint

import wandb
from agent0.hyperdrive.interactive import InteractiveHyperdrive, LocalChain
from agent0.hyperdrive.state import HyperdriveActionType

# %%
wandb.login()

# %%
experiment_name = "some name"
experiment_notes = "Running sweeps to find best fees"
experiment_tags = ["sweeps", "fees"]

experiment_params = {"fixed_rate": 0.05, "curve_fee": 0.01}

## Initialize sweep config
sweep_config = {"method": "random"}

## Set goals
metric = {"name": "pnl", "goal": "maximize"}
sweep_config["metric"] = metric

## what to sweep over
grid_parameters = {"fixed_rate": {"values": [0, 0.01, 0.1]}}

random_parameters = {
    "variable_rate": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.05,
    },
    "curve_fee": {
        "distribution": "normal",
        "mu": 0.01,
        "sigma": 0.001,
    },
}

constant_parameters = {"position_duration": {"value": 60 * 60 * 24 * 365}}

parameters_dict = {}
parameters_dict.update(grid_parameters)
parameters_dict.update(random_parameters)
parameters_dict.update(constant_parameters)

sweep_config["parameters"] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project=experiment_name)


def run_experiment(config=None):
    with wandb.init(config=config, project=experiment_name):
        config = wandb.config
        start_time = time.time()

        ## Experiment settings
        @dataclass
        class ExperimentConfig:
            # experiment times
            experiment_days: int = 60 * 60 * 24 * 30  # 1 month
            position_duration: int = config["position_duration"]
            checkpoint_duration: int = 60 * 60 * 24 * 7  # 1 week
            initial_liquidity: FixedPoint(2 * 10**8)
            # trading amounts
            daily_volume_percentage_of_liquidity: float = 0.10
            opens_per_day: int = 6  # how often to open a trade
            minimum_trade_hold_days: float = 1  # minimum number of days to keep a trade open
            agent_budget: FixedPoint = FixedPoint(10_000_000)
            # starting fixed rate equal to variable should encourage net 0 profit for rob
            variable_rate: FixedPoint = config["variable_rate"]
            fixed_rate: FixedPoint = config["fixed_rate"]
            # fees
            curve_fee: FixedPoint = config["curve_fee"]
            flat_fee: FixedPoint = FixedPoint("0")
            governance_fee: FixedPoint = FixedPoint("0")
            # misc extra
            experiment_id: int = 0
            randseed: int = 0

        exp = ExperimentConfig()
        log_dict = deepcopy(config)
        log_dict.update(asdict(exp))
        wandb.log(log_dict)

        ## Interactive Hyperdrive config has a subset of experiment config
        hyperdrive_config = InteractiveHyperdrive.Config(
            position_duration=exp.position_duration,
            checkpoint_duration=exp.checkpoint_duration,
            initial_liquidity=exp.initial_liquidity,
            initial_fixed_apr=exp.fixed_rate,
            initial_variable_rate=exp.variable_rate,
            curve_fee=exp.curve_fee,
            flat_fee=exp.flat_fee,
            governance_lp_fee=exp.governance_fee,
            calc_pnl=False,
        )

        ## Initialize primary objects
        rng = np.random.default_rng(seed=exp.randseed)
        chain = LocalChain(LocalChain.Config(db_port=5_433, chain_port=10_000))
        interactive_hyperdrive = InteractiveHyperdrive(hyperdrive_config, chain)

        ## Initialize agents
        deployer_privkey = chain.get_deployer_account_private_key()
        larry = interactive_hyperdrive.init_agent(base=exp.agent_budget, name="larry", private_key=deployer_privkey)
        rob = interactive_hyperdrive.init_agent(base=exp.agent_budget, name="rob")

        # Trades are randomly long or short; trade amounts are fixed
        current_trading_volume = 0
        base_amount_per_open = exp.initial_liquidity * exp.daily_volume_percentage_of_liquidity / exp.opens_per_day
        lp_present_value = []
        for day in range(exp.experiment_days):
            # Open a long or short trade for a predetermined amount
            for _ in range(exp.opens_per_day):
                trade_type = rng.choice([HyperdriveActionType.OPEN_LONG, HyperdriveActionType.OPEN_SHORT], size=1)
                match trade_type:
                    case HyperdriveActionType.OPEN_LONG:
                        open_event = rob.open_long(base=base_amount_per_open)
                    case HyperdriveActionType.OPEN_SHORT:
                        pool_state = interactive_hyperdrive.interface.current_pool_state
                        amount_to_trade_bonds = interactive_hyperdrive.interface.calc_bonds_out_given_shares_in_down(
                            amount_in=base_amount_per_open / pool_state.pool_info.vault_share_price,
                            pool_state=pool_state,
                        )
                        open_event = rob.open_short(bonds=amount_to_trade_bonds)
                # update present value after a trade
                lp_present_value.append(interactive_hyperdrive.interface.calc_present_value())
            # Optionally close trades
            position_duration_days = int(interactive_hyperdrive.interface.pool_config.position_duration / 60 / 60 / 24)
            close_events = []
            for long in rob.wallet.longs.values():
                mint_time = long.maturity_time - interactive_hyperdrive.interface.pool_config.position_duration
                current_block_time = interactive_hyperdrive.interface.get_block_timestamp(
                    interactive_hyperdrive.interface.get_current_block()
                )
                days_passed = int((current_block_time - mint_time) / 60 / 60 / 24)
                if days_passed > exp.minimum_trade_hold_days:
                    gonna_close = rng.choice([True, False], size=1)
                    if gonna_close or days_passed > position_duration_days:
                        close_events.append(rob.close_long(long.maturity_time, long.balance))
                        # update present value after a trade
                        lp_present_value.append(interactive_hyperdrive.interface.calc_present_value())
            for short in rob.wallet.shorts.values():
                mint_time = short.maturity_time - interactive_hyperdrive.interface.pool_config.position_duration
                current_block_time = interactive_hyperdrive.interface.get_block_timestamp(
                    interactive_hyperdrive.interface.get_current_block()
                )
                days_passed = int((current_block_time - mint_time) / 60 / 60 / 24)
                if days_passed > exp.minimum_trade_hold_days:
                    gonna_close = rng.choice([True, False], size=1)
                    if gonna_close or days_passed > position_duration_days:
                        close_events.append(rob.close_short(short.maturity_time, short.balance))
                        # update present value after a trade
                        lp_present_value.append(interactive_hyperdrive.interface.calc_present_value())
            wandb.log({"pnl": lp_present_value[-1], "day": day})
            wandb.log({"something_else": lp_present_value[-1]})
        # Close everything up in the end
        rob.liquidate(randomize=True)
        # update present value after the remaining trades
        lp_present_value.append(interactive_hyperdrive.interface.calc_present_value())

        ## Run analysis
        # total lp profits (lp present value) per day
        # vault share price per day
        # fees per day
        # profit from backing trades per day
        #
        pool_state = interactive_hyperdrive.get_pool_state().to_parquet("pool_state.parquet")
        end_time = time.time()
        wandb.log({"exp_time": end_time - start_time})


wandb.agent(sweep_id, run_experiment, count=500)

wandb.finish()
# %%
