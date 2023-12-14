# hyperdrive-experiments

Test bed for understanding the Hyperdrive AMM

## INSTALL

1. Install the following applications:

- [Pyenv install instructions](https://github.com/pyenv/pyenv#installation) for controlling your python version.
- [Docker](docs.docker.com/get-docker) for running frontend, databases, dashboards, etc.
- [Anvil](<[url](https://book.getfoundry.sh/reference/anvil/)>) for running a simulated block chain.

2. Setup Python 3.10

```bash
pyenv install 3.10
pyenv local 3.10
python -m venv .venv
source .venv/bin/activate
```

3. Install agent0:

```bash
python -m pip install --upgrade pip
python -m pip install "agent0[with-dependencies] @ git+https://git@github.com/delvtech/agent0.git@v0.11.0#subdirectory=lib/agent0"
python -m pip install --upgrade -r https://raw.githubusercontent.com/delvtech/agent0/v0.11.0/requirements-dev.txt
```
