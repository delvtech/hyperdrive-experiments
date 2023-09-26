# hyperdrive-experiments

Test bed for understanding the Hyperdrive AMM

## INSTALL

1. Install the following applications:

- [Pyenv install instructions](https://github.com/pyenv/pyenv#installation) for controlling your python version.
- [Docker](docs.docker.com/get-docker) for running frontend, databases, dashboards, etc.
- [Anvil](<[url](https://book.getfoundry.sh/reference/anvil/)>) for running a simulated block chain (devnet).

2. Setup Python 3.10

```bash
pyenv install 3.10
pyenv local 3.10
python -m venv .venv
source .venv/bin/activate
```

3. Clone & link hyperdrive:

```bash
git clone --branch v0.0.13 git@github.com:delvtech/hyperdrive.git ../hyperdrive
ln -s ../hyperdrive hyperdrive_solidity
```

4. Clone & install elf-simulations:

```bash
git clone --branch v0.5.0 git@github.com:delvtech/elf-simulations.git ../elf-simulations
python -m pip install --upgrade pip
python -m pip install --upgrade -r ../elf-simulations/requirements.txt
python -m pip install --upgrade -r ../elf-simulations/requirements-dev.txt
```
