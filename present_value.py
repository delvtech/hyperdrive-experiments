"""Calculate the present value of an LP's capital in the pool for many pool states."""

# %%
from agent0 import AccountKeyConfig
from agent0.base.make_key import make_private_key
from agent0.hyperdrive.agents import HyperdriveAgent
from agent0.hyperdrive.exec import fund_agents
from eth_account.account import Account
from ethpy import EthConfig
from ethpy.hyperdrive import HyperdriveInterface, deploy_hyperdrive_from_factory
from fixedpointmath import FixedPoint
from hypertypes.IHyperdriveTypes import Fees, PoolConfig

# pylint: disable=invalid-name

# %%
# Experiment parameters

# address to local Anvil deployment process
anvil_address = "127.0.0.1:8545"

# ABI folder should contain JSON and Bytecode files for the following contracts:
# ERC20Mintable, MockERC4626, ForwarderFactory, ERC4626HyperdriveDeployer, ERC4626HyperdriveFactory
abi_dir = "packages/hyperdrive/src/abis/"

# Deployer is the pre-funded account 0 on the Delv devnet
deployer_private_key: str = (
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
)

# Factory initializaiton parameters
initial_variable_rate = FixedPoint("0.05")
curve_fee = FixedPoint("0.1")  # 10%
flat_fee = FixedPoint("0.0005")  # 0.05%
governance_fee = FixedPoint("0.15")  # 15%
max_curve_fee = FixedPoint("0.3")  # 30%
max_flat_fee = FixedPoint("0.0015")  # 0.15%
max_governance_fee = FixedPoint("0.30")  # 30%

# Pool initialization parameters
initial_fixed_rate = FixedPoint("0.05")  # 5%
initial_liquidity = FixedPoint(100_000_000)  # 100M ETH
initial_share_price = FixedPoint(1)
minimum_share_reserves = FixedPoint(10)
minimum_transaction_amount = FixedPoint("0.001")
time_stretch = (
    FixedPoint("0.04665") * (initial_fixed_rate * FixedPoint(100))
) / FixedPoint("5.24592")
position_duration = 604800  # 1 week
checkpoint_duration = 3600  # 1 hour
oracle_size = 10
update_gap = 3600  # 1 hour

# Derived values
fees = Fees(curve_fee.scaled_value, flat_fee.scaled_value, governance_fee.scaled_value)
max_fees = Fees(
    max_curve_fee.scaled_value,
    max_flat_fee.scaled_value,
    max_governance_fee.scaled_value,
)
initial_pool_config = PoolConfig(
    "",  # will be determined in the deploy function
    initial_share_price.scaled_value,
    minimum_share_reserves.scaled_value,
    minimum_transaction_amount.scaled_value,
    position_duration,
    checkpoint_duration,
    time_stretch.scaled_value,
    "",  # will be determined in the deploy function
    "",  # will be determined in the deploy function
    fees,
    oracle_size,
    update_gap,
)

# %%
# Initialize the hyperdrive interafce and deploy the chain
hyperdrive_chain = deploy_hyperdrive_from_factory(
    anvil_address,
    abi_dir,
    deployer_private_key,
    initial_liquidity,
    initial_variable_rate,
    initial_fixed_rate,
    initial_pool_config,
    max_fees,
)
hyperdrive_interface = HyperdriveInterface(
    EthConfig(artifacts_uri="not used", rpc_uri=anvil_address, abi_dir=abi_dir),
    hyperdrive_chain.hyperdrive_contract_addresses,
    hyperdrive_chain.web3,
)

# %%
# create & fund agents
num_agents = 8
eth_budgets = [
    1 * 10**18,
] * num_agents
base_budgets = [
    50_000 * 10**18,
] * num_agents

agents: list[HyperdriveAgent] = []
for _ in range(num_agents):
    user_private_key = make_private_key()
    agents.append(HyperdriveAgent(Account().from_key(user_private_key)))

account_config = AccountKeyConfig(
    deployer_private_key,
    [agent._private_key for agent in agents],  # pylint: disable=protected-access
    eth_budgets,
    base_budgets,
)
fund_agents(
    HyperdriveAgent(hyperdrive_chain.deploy_account),
    hyperdrive_interface.config,
    account_config,
    hyperdrive_chain.hyperdrive_contract_addresses,
)

# %%
# execute trades
lp_agent_index = 0
trade_result = hyperdrive_interface.async_add_liquidity(
    agent=agents[lp_agent_index],
    trade_amount=FixedPoint(1000),
    min_apr=FixedPoint("0.001"),
    max_apr=FixedPoint("1.0"),
)

trade_result = hyperdrive_interface.async_open_long(
    agent=agents[1], trade_amount=FixedPoint(100), slippage_tolerance=FixedPoint("0.1")
)

trade_result = hyperdrive_interface.async_open_short(
    agent=agents[2], trade_amount=FixedPoint(10), slippage_tolerance=FixedPoint("0.1")
)


# %%
# calculate present value

# TODO: We need access to the present value calculation.
# I think the best way to do this is request that it gets added to hyperdrive-rs


# %%
# close a trade & estimate new present value
