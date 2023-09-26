"""Calculate the present value of an LP's capital in the pool for many pool states."""

# %%
from agent0.base.make_key import make_private_key
from agent0.hyperdrive.agents import HyperdriveAgent
from eth_account.account import Account

# %%
# spin up hyperdrive chain & initialize interface

# %%
# create agents
user_private_key = make_private_key(
    extra_entropy="FAKE USER"
)  # argument value can be any str
user_account = HyperdriveAgent(Account().from_key(user_private_key))

# %%
# execute trades

# %%
# calculate present value

# %%
# close a trade & estimate new present value
