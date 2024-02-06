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
from agent0.hyperdrive.agent import HyperdriveActionType
from agent0.hyperdrive.interactive import InteractiveHyperdrive, LocalChain
from agent0.hyperdrive.policies import PolicyZoo


@dataclass
class LpPnlConfig:
    # experiment times
    experiment_days: int = 365  # 1 year
    position_duration: int = 60 * 60 * 24 * 30  # 1 month
    checkpoint_duration: int = 60 * 60 * 24  # 1 day
    initial_liquidity: FixedPoint = FixedPoint(200_000_000)
    # trading amounts
    daily_volume_percentage_of_liquidity: FixedPoint = FixedPoint("0.10")
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


def lp_pnl_experiment(config=None):
    start_time = time.time()
    experiment_notes = "Compute lp PNL given fees."
    experiment_tags = ["fees", "lp pnl"]
    mode = "online" if config is None else config["wandb_init_mode"]

    with wandb.init(config=config, notes=experiment_notes, tags=experiment_tags, mode=mode) as run:
        ## Setup config
        run_config = run.config
        exp_config = LpPnlConfig()
        # merge overlapping run config settings into experiment config, with casting since wandb communicates with dicts
        for key, value in run_config.items():
            if hasattr(exp_config, key):
                exp_type = type(asdict(exp_config)[key])
                setattr(exp_config, key, exp_type(value))
        # if in a sweep, add sweep id to the run config dict
        if hasattr(run, "sweep_id"):
            run_config["sweep_id"] = run.sweep_id
        # log a combo of the two configs
        log_dict = deepcopy(asdict(exp_config))
        log_dict.update(run_config)
        run.log(log_dict)

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
        total_daily_volume = exp_config.initial_liquidity * exp_config.daily_volume_percentage_of_liquidity

        ## Initialize primary objects
        rng = np.random.default_rng(seed=exp_config.randseed)
        chain = LocalChain(LocalChain.Config(db_port=5_433, chain_port=10_000))
        interactive_hyperdrive = InteractiveHyperdrive(chain, hyperdrive_config)

        ## Initialize agents
        # TODO: Directly compute and log liquidity for larry, instead of using lp share price.
        # deployer_privkey = chain.get_deployer_account_private_key()
        # larry = interactive_hyperdrive.init_agent(
        #     base=exp_config.agent_budget, name="larry", private_key=deployer_privkey
        # )
        agents = [
            interactive_hyperdrive.init_agent(
                base=exp_config.agent_budget,
                policy=PolicyZoo.lp_and_arb,
                policy_config=PolicyZoo.lp_and_arb.Config(rng=rng, lp_portion=FixedPoint("0.0")),
                name=f"agent_{id}",
            )
            for id in range(exp_config.num_agents)
        ]

        ## Run the experiment loop
        lp_present_value = []
        trade_events = []
        for day in range(exp_config.experiment_days):
            current_daily_volume = FixedPoint(0)
            while current_daily_volume < total_daily_volume:
                trader = agents[rng.integers(len(agents))]
                position_duration_days = int(
                    interactive_hyperdrive.interface.pool_config.position_duration / 60 / 60 / 24
                )

                # loop over longs & close them if they're mature enough
                for long in trader.wallet.longs.values():
                    mint_time = long.maturity_time - interactive_hyperdrive.interface.pool_config.position_duration
                    current_block_time = interactive_hyperdrive.interface.get_block_timestamp(
                        interactive_hyperdrive.interface.get_current_block()
                    )
                    days_passed = int((current_block_time - mint_time) / 60 / 60 / 24)
                    if days_passed > exp_config.minimum_trade_hold_days:
                        gonna_close = rng.choice([True, False], size=1)
                        if gonna_close or days_passed > position_duration_days:
                            trade_events.append(trader.close_long(long.maturity_time, long.balance))
                            # update present value and volume after a trade
                            lp_present_value.append(interactive_hyperdrive.interface.calc_present_value())
                            current_daily_volume += long.balance

                # loop over shorts & close them if they're mature enough
                for short in trader.wallet.shorts.values():
                    mint_time = short.maturity_time - interactive_hyperdrive.interface.pool_config.position_duration
                    current_block_time = interactive_hyperdrive.interface.get_block_timestamp(
                        interactive_hyperdrive.interface.get_current_block()
                    )
                    days_passed = int((current_block_time - mint_time) / 60 / 60 / 24)
                    if days_passed > exp_config.minimum_trade_hold_days:
                        gonna_close = rng.choice([True, False], size=1)
                        if gonna_close or days_passed > position_duration_days:
                            trade_events.append(trader.close_short(short.maturity_time, short.balance))
                            # update present value and volume after a trade
                            lp_present_value.append(interactive_hyperdrive.interface.calc_present_value())
                            current_daily_volume += short.balance

                # execute a profitable or not profitable trade (50% chance for either)
                profitable = rng.choice([True, False], size=1)
                if profitable:  # do an arb trade half the time
                    trade_events.append(trader.execute_policy_action())
                else:  # not guaranteed to lose money, but more likely to given the arb trades
                    base_amount_to_open = total_daily_volume - current_daily_volume
                    match rng.choice([HyperdriveActionType.OPEN_LONG, HyperdriveActionType.OPEN_SHORT], size=1):
                        case HyperdriveActionType.OPEN_LONG:
                            trade_events.append(trader.open_long(base=base_amount_to_open))
                        case HyperdriveActionType.OPEN_SHORT:
                            pool_state = interactive_hyperdrive.interface.current_pool_state
                            amount_to_trade_bonds = (
                                interactive_hyperdrive.interface.calc_bonds_out_given_shares_in_down(
                                    amount_in=base_amount_to_open / pool_state.pool_info.vault_share_price,
                                    pool_state=pool_state,
                                )
                            )
                            trade_events.append(trader.open_short(bonds=amount_to_trade_bonds))
                current_daily_volume += trade_events[-1].base_amount
                # update present value after a trade
                lp_present_value.append(interactive_hyperdrive.interface.calc_present_value())

            run.log({"pnl": lp_present_value[-1], "day": day})
        ## Liquidate trades at the end
        for agent in agents:
            agent.liquidate(randomize=True)

        ## Run analytics
        # update present value after the remaining trades
        lp_present_value.append(interactive_hyperdrive.interface.calc_present_value())
        run.log({"pnl": lp_present_value[-1], "day": day + 1})
        # Save experiment pool state onto w&b
        pool_state = interactive_hyperdrive.get_pool_state().to_parquet("pool_state.parquet")
        state_artifact = wandb.Artifact(
            "pool-state",
            type="dataset",
            description="Final state of the Hyperdrive pool after the experiment was completed",
            metadata=log_dict,
        )
        state_artifact.add(pool_state, "pool-state")
        run.log_artifact(state_artifact)

        ## Log final time
        end_time = time.time()
        run.log({"experiment_time": end_time - start_time})

    chain.cleanup()
