"""Run some random trades."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import wandb
from fixedpointmath import FixedPoint
from hyperlogs import ExtendedJSONEncoder

from agent0.hyperdrive.agent import HyperdriveActionType
from agent0.hyperdrive.interactive import InteractiveHyperdrive, LocalChain
from agent0.hyperdrive.policies import PolicyZoo
from utils import convert_run_config


def random_trades(config=None):
    start_time = time.time()
    experiment_notes = "Execute a minimum volume of random trades."
    experiment_tags = ["fees", "lp pnl"]
    mode = "online" if config is None else config["wandb_init_mode"]

    with wandb.init(config=config, notes=experiment_notes, tags=experiment_tags, mode=mode) as run:
        ## Setup config
        run_config = run.config
        exp_config = convert_run_config(run_config)
        # if in a sweep, add sweep id to the run config dict
        if hasattr(run, "sweep_id"):
            run_config["sweep_id"] = run.sweep_id
        # log a combo of the two configs
        log_dict = deepcopy(asdict(exp_config))
        log_dict.update(run_config)
        run.log(json.loads(json.dumps(log_dict, cls=ExtendedJSONEncoder)))

        ## Initialize primary objects
        total_daily_volume = exp_config.initial_liquidity * exp_config.daily_volume_percentage_of_liquidity
        rng = np.random.default_rng(seed=exp_config.randseed)
        chain = LocalChain(LocalChain.Config(db_port=5_433, chain_port=10_000))
        interactive_hyperdrive = InteractiveHyperdrive(chain, exp_config)

        ## Initialize agents
        agents = [
            interactive_hyperdrive.init_agent(
                base=exp_config.agent_budget,
                policy=PolicyZoo.random,
                policy_config=PolicyZoo.random.Config(
                    rng=rng,
                    allowable_actions=[
                        HyperdriveActionType.OPEN_LONG,
                        HyperdriveActionType.CLOSE_LONG,
                        HyperdriveActionType.OPEN_SHORT,
                        HyperdriveActionType.CLOSE_SHORT,
                    ],
                ),
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
                agent = agents[rng.integers(len(agents))]
                trade_outcomes = agent.execute_policy_action()
                for trade_outcome in trade_outcomes:
                    current_daily_volume += trade_outcome.base_amount
                    trade_events.append(trade_outcome)
                lp_present_value.append(interactive_hyperdrive.interface.calc_present_value())
            run.log({"pnl": float(lp_present_value[-1]), "day": day})

        ## Liquidate trades at the end
        for agent in agents:
            agent.liquidate(randomize=True)

        ## Run analytics
        # update present value after the remaining trades
        lp_present_value.append(interactive_hyperdrive.interface.calc_present_value())
        run.log({"pnl": float(lp_present_value[-1]), "day": day + 1})
        # Save experiment pool state onto w&b
        try:  # this will only work if wandb is enabled
            pool_state = interactive_hyperdrive.get_pool_state().to_parquet("pool_state.parquet")
            state_artifact = wandb.Artifact(
                "pool-state",
                type="dataset",
                description="Final state of the Hyperdrive pool after the experiment was completed",
                metadata=log_dict,
            )
            state_artifact.add(pool_state, "pool-state")
            run.log_artifact(state_artifact)
        except ValueError:
            pass

        ## Log final time
        end_time = time.time()
        run.log({"experiment_time": end_time - start_time})

    chain.cleanup()
