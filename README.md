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

The reward arrives as a plain function, so swapping the fitted regressor for RDKit's QED
is a flag rather than a subclass. That flag is what makes the result below legible.

## The result

| Reward | Agent | Random |
|---|---:|---:|
| QED, the control | 0.895 | 0.145 |
| EGFR pIC50, the real target | 0.859 | 0.331 |

Each row is on its own reward's scale. Supporting numbers: the GNN encoder matches a
fingerprint baseline on QED using 56k parameters against 2.7M, and pretraining on ZINC
improves the regressor's test MAE from 0.868 to 0.806.

**The finding worth your time is that the agent games whatever it is scored on.** Against
QED it built fused strained heterocycles, enol ethers and N-hydroxyls — molecules that
score 0.93 and cannot exist. Anyone can see those are wrong, which is the point of keeping
QED around. Against the fitted pIC50 regressor it found a molecule scoring 9.95 that looks
entirely plausible, and only a substructure audit shows every one of its top molecules
carries a nitrogen–nitrogen bond.

A reward curve that climbs is not evidence of lead optimization. It is evidence that the
loop optimizes its reward. Telling those apart is what this repo is for, and
[results/](results/README.md) has every figure with the number it carries.

## How to run it

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -m "not slow"   # 166 tests, about 15 seconds
```

Then the whole pipeline, from download to figures:

```bash
./run_all.sh
```

About 25 minutes of compute, plus a 605 MB download the first time. Each fetch is
pinned by checksum, so a silently changed upstream file fails loudly instead of quietly
moving your numbers. Runs write into `runs/`, which is not tracked; the plots worth
keeping are copied into `results/`. [docs/running.md](docs/running.md) breaks the script
into its five steps with timings, and explains the flags worth changing.

## Where things are

- [mol_optim/](mol_optim/README.md) — what each module is, and where to start reading
- [docs/running.md](docs/running.md) — every command, with timings
- [results/](results/README.md) — the figures and the numbers

## Working on this repo with Claude Code

**Claude does not write to git. The maintainer makes every commit.**

No `git commit`, no `git merge`, no `git rebase`, no rewriting history, no `git push`, no
branches or tags, no pull requests. Claude edits files in the working tree and stops
there — `git add` and everything after it is done by hand.

This includes merge commits and the co-author trailer that comes with them: nothing
Claude does should put a second name on a commit or in the repo's contributor list.

## License

[GPL-3.0-or-later](LICENSE).

## Prior work

[MolDQN-pytorch](https://github.com/aksub99/MolDQN-pytorch) is the working base;
[google-research/mol_dqn](https://github.com/google-research/google-research/tree/master/mol_dqn)
is the original TF1 code and the source of the published hyperparameters. Neither is
imported here. Reimplementing rather than importing measured what the former's
defaults cost: 0.19 of QED.
