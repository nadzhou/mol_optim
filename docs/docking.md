# Docking (optional)

**Before you invest in this:** docking was measured on this target and does *not* rank
known EGFR compounds correctly. It is a spot-check that never runs in the training loop.
See [plan.md](../plan.md) for the measurement.

## Install

```bash
pip install -e ".[docking]"
```

That pulls `vina`, `meeko`, `scipy`, and `gemmi`. Build the receptor once, then run the
gating test — a redock of the crystallographic ligand:

```bash
python -m mol_optim.fetch_structure
pytest tests/test_docking.py
```

Open Babel types the receptor rather than Meeko, whose polymer path fails on this entry's
terminal residues. Install it however your platform prefers; it is a runtime tool, not a
Python dependency.

## If the vina wheel does not build

AutoDock Vina publishes no wheel for some platforms, Apple Silicon among them, and its
`setup.py` then has two problems: it looks for Boost only in a conda prefix or
`/usr/local`, and it compiles with `-std=c++11`, which Boost 1.92 headers no longer
accept. Both are fixable from the source distribution — point it at wherever your Boost
actually lives and raise the standard:

```bash
pip download vina --no-deps --no-binary :all: -d /tmp/vina
tar xzf /tmp/vina/vina-*.tar.gz -C /tmp/vina
sed -i.bak 's/-std=c++11/-std=c++17/' /tmp/vina/vina-*/setup.py
CONDA_PREFIX=/path/to/boost/prefix CONDA_DEFAULT_ENV=local pip install /tmp/vina/vina-*/
```

`CONDA_PREFIX` is read as a search root whether or not conda is involved, so it is the
lever for a Boost installed anywhere else. Set it to the prefix containing `include/boost`
and `lib`.
