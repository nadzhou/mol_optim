# mol_optim

A reinforcement learning agent that improves a molecule by editing it — add an atom here,
close a ring there — and, at every step, a measurement of whether it is learning chemistry
or just gaming the thing that scores it.

## What this does

The setting is lead optimization, not design from nothing: the agent starts from a
compound that already works and makes small changes to it. The molecule stays an RDKit
graph throughout — there is no SMILES round-trip, no wrapper class, and no training
framework.

Three pieces, each measured before the next one is built:

- **A graph encoder pretrained on ZINC.** Masked-atom prediction over 249,455 molecules,
  which gives the network a starting point that knows what ordinary chemistry looks like.
- **A pIC50 regressor fitted to BindingDB EGFR data.** 10,850 compounds on a scaffold
  split, five networks in an ensemble. This is the reward.
- **A DQN over molecular edits.** Add an atom, add a bond, remove a bond, or stop, with
  the candidate set enumerated fresh at every state.

The reward arrives as a plain function named in a config file, so fitting a new one and
pointing the agent at it is an edit to that file, not a subclass.

## The result

| Reward | Agent | Random |
|---|---:|---:|
| EGFR pIC50 | 0.859 | 0.331 |

Both on the same scale, predicted pIC50 divided by 10. Pretraining on ZINC improves the
regressor's test MAE from 0.868 to 0.806, and the GNN encoder carries 56k parameters
against the published fingerprint MLP's 2.7M.

**The finding worth your time is that the agent games whatever it is scored on.** Against
the fitted pIC50 regressor it found a molecule scoring 9.95 that looks entirely plausible,
and only a substructure audit shows every one of its top molecules carries a
nitrogen–nitrogen bond.

A reward curve that climbs is not evidence of lead optimization. It is evidence that the
loop optimizes its reward. Telling those apart is what this repo is for, and
[results/](results/README.md) has every figure with the number it carries.

## How to run it

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -m "not slow"   # 104 tests, about 14 seconds
```

Then the whole pipeline, from download to figures — one command, one config file:

```bash
mol-optim configs/config.toml
```

About 25 minutes of compute, plus a 605 MB download the first time. Each fetch is pinned
by checksum, so a silently changed upstream file fails loudly instead of quietly moving
your numbers. Runs write into `runs/`, which is not tracked; the figures are written
straight into `results/`, so regenerating them is the same command as running the
pipeline.

Every knob is in that file: which steps run, which datasets, which agents in which order,
and every hyperparameter. A different experiment is a different config file, not a
different command line. [docs/running.md](docs/running.md) says what each step does, with
timings, and what is worth changing.

## Where things are

- [mol_optim/](mol_optim/README.md) — what each module is, and where to start reading
- [configs/config.toml](configs/config.toml) — the pipeline, as data
- [docs/running.md](docs/running.md) — what each step does, and what to change
- [results/](results/README.md) — the figures and the numbers
- [CONTRIBUTING.md](CONTRIBUTING.md) — how commits work here, and the size budget

## License

[GPL-3.0-or-later](LICENSE).

## Prior work

[MolDQN-pytorch](https://github.com/aksub99/MolDQN-pytorch) is the working base;
[google-research/mol_dqn](https://github.com/google-research/google-research/tree/master/mol_dqn)
is the original TF1 code and the source of the published hyperparameters. Neither is
imported here. Reimplementing rather than importing measured what the former's
defaults cost — the published gamma, ring sizes,
buffer size, update interval and gradient clipping are what this repo uses, not the
port's.
