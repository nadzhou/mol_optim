# mol_optim

A reinforcement learning agent that improves a molecule by editing it — add an atom here,
close a ring there — and, at every step, a measurement of whether it is learning chemistry
or just gaming the thing that scores it.

The setting is lead optimization, not design from nothing: the agent starts from a compound
that already works and makes small changes to it. The molecule stays an RDKit graph
throughout.

Steps 0–4 are done and measured; Step 5, a fitted pIC50 model as the reward, has a pilot
run. Design and build order: [plan.md](plan.md). Coding rules: [CLAUDE.md](CLAUDE.md).

## Results

[**Every figure, in build order, with the number it carries →**](results/README.md)

| Step | Reward | Agent | Random |
|---|---|---:|---:|
| 1–2 | QED | 0.895 | 0.145 |
| 5 | EGFR pIC50, pilot | 0.859 | 0.331 |

Each row is on its own reward's scale. The GNN encoder matches the fingerprint baseline on
QED using 56k parameters against 2.7M, and pretraining on ZINC improves the EGFR regressor
from 0.868 to 0.806 test MAE.

The finding worth your time is that the agent games every scorer it is given. Against QED
it built strained fused heterocycles, which anyone can see are wrong; against the pIC50
regressor it found a molecule that looks entirely plausible — which is the harder
problem.

## Quickstart

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Fetch the data. Each runs once and is pinned by checksum, so a silently changed upstream
file fails loudly instead of quietly moving your numbers.

```bash
python -m mol_optim.zinc             # 12 MB  → data/zinc.tab
python -m mol_optim.fetch_bindingdb  # 593 MB → data/egfr_ic50.sdf
```

Check it works, then train:

```bash
pytest -m "not slow"
python -m mol_optim.train_dqn --episodes 5000 --reward qed \
  --log runs/dqn.csv --checkpoint runs/dqn.pt --top-k runs/dqn_top
```

166 tests, about 15 seconds. Runs write into `runs/`, which is not tracked; the plots worth
keeping are copied into `results/`.

## Where things are

- [mol_optim/](mol_optim/README.md) — what each module is, and where to start reading
- [docs/running.md](docs/running.md) — every command, with timings
- [plan.md](plan.md) — the design, and the test gating each step
- [results/](results/README.md) — the figures and the numbers

## License

[GPL-3.0-or-later](LICENSE).

## Prior work

[MolDQN-pytorch](https://github.com/aksub99/MolDQN-pytorch) is the working base;
[google-research/mol_dqn](https://github.com/google-research/google-research/tree/master/mol_dqn)
is the original TF1 code and the source of the published hyperparameters. Neither is
imported here — Step 1 of plan.md measures what the former's defaults cost: 0.19 of QED.
