# mol_optim

RL molecular derivatization. Design and build order: [plan.md](plan.md). Coding style:
[CLAUDE.md](CLAUDE.md).

Current position: Step 3. Atom-level graph edits, GNN state encoder, and TDC's GSK3B
oracle as the reward — a published bioactivity model, so a flat reward curve means the
loop is wrong and not the reward. The state is an RDKit molecular graph end to end —
molecules become text only in `report.py`, where a person looks at them.

## Setup

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
.venv/bin/pip install rdkit torch pytest hypothesis matplotlib
.venv/bin/python -m mol_optim.fetch_gsk3b
```

The last line downloads TDC's published GSK3B oracle once (28 MB) and writes the 2 MB
`models/gsk3b_forest.npz` this repo reads. PyTDC itself is not a dependency — see
`fetch_gsk3b.py` for why.

## Tests

```bash
.venv/bin/python -m pytest -m "not slow"
```

89 tests, under 6 seconds. The full run adds four 5000-episode training runs and a
step-latency measurement.

## Runs

```bash
.venv/bin/python -m mol_optim.train_dqn --episodes 5000 --reward gsk3b --log runs/dqn.csv --checkpoint runs/dqn.pt --top-k runs/dqn_top
```

`--reward qed` is the Step 1 and Step 2 target; `--reward gsk3b` is Step 3's oracle.
`baseline_random.py` takes the same flag.

```bash
.venv/bin/python -m mol_optim.plot_run runs/dqn.csv --out runs/curve.png --random-baseline 0.146
```

`--top-k` writes the best distinct molecules of a run as a drawing and an SDF.

## Layout

| File | What it is |
|---|---|
| `config.py` | every hyperparameter, one frozen dataclass, passed explicitly |
| `graph_key.py` | the state key: a canonical name taken off the graph, no SMILES |
| `environment.py` | the MDP: candidate enumeration by RWMol edits, `step` |
| `featurize.py` | molecular graph to network tensors: int8 codes, one-hot at the network |
| `encoder.py` | the GNN: message passing over atoms and bonds, mean pooled |
| `rewards.py` | QED, the Step 1 and Step 2 target |
| `oracle_gsk3b.py` | the Step 3 reward: TDC's GSK3B random forest, walked by hand |
| `fetch_gsk3b.py` | run once: TDC's pickle in, `models/gsk3b_forest.npz` out |
| `replay_buffer.py` | ours; ragged, because the target is a max over a candidate set |
| `dqn.py` | the Q network: the encoder plus a head that reads steps remaining |
| `train_dqn.py` | the training loop, flat |
| `baseline_random.py` | tier 0 of the ladder — the number DQN has to beat |
| `molio.py` | SDF in and out; the only place a graph becomes bytes |
| `report.py` | top-k molecules as a drawing and an SDF |
| `plot_run.py` | reward and loss curves from a run log |

The reference implementations live outside this repo, one directory up:
`MolDQN-pytorch/` (the working base) and `google-research/mol_dqn/` (the original TF1
code, and the published hyperparameters this project follows). Nothing here imports them.
