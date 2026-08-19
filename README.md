# mol_optim

RL molecular derivatization. Design and build order: [plan.md](plan.md). Coding style:
[CLAUDE.md](CLAUDE.md).

Current position: Step 3b. Atom-level graph edits, GNN state encoder, TDC's GSK3B
oracle as the reward — a published bioactivity model, so a flat reward curve means the
loop is wrong and not the reward — and the encoder pretrained on ZINC by masked-atom
prediction. The state is an RDKit molecular graph end to end; molecules become text
only in `report.py`, where a person looks at them, and in `zinc.py`, where ZINC's
published SMILES are read once.

## Setup

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
.venv/bin/pip install rdkit torch pytest hypothesis matplotlib
.venv/bin/python -m mol_optim.fetch_gsk3b
.venv/bin/python -m mol_optim.zinc
```

The third line downloads TDC's published GSK3B oracle once (28 MB) and writes the 2 MB
`models/gsk3b_forest.npz` this repo reads. PyTDC itself is not a dependency — see
`fetch_gsk3b.py` for why. The fourth downloads ZINC 250k (12 MB) to `data/zinc.tab`,
the molecules the encoder is pretrained on. Both files are pinned by SHA-256.

## Tests

```bash
.venv/bin/python -m pytest -m "not slow"
```

102 tests, under 5 seconds. The full run adds four 5000-episode training runs, a
step-latency measurement, and a short ZINC pretraining run.

## Runs

```bash
.venv/bin/python -m mol_optim.train_dqn --episodes 5000 --reward gsk3b --log runs/dqn.csv --checkpoint runs/dqn.pt --top-k runs/dqn_top
```

`--reward qed` is the Step 1 and Step 2 target; `--reward gsk3b` is Step 3's oracle.
`baseline_random.py` takes the same flag. `--pretrained-encoder models/zinc_encoder.pt`
starts the run from the Step 3b encoder instead of a random one.

```bash
.venv/bin/python -m mol_optim.pretrain --epochs 10 --checkpoint models/zinc_encoder.pt --log runs/pretrain_zinc.csv
```

Masked-atom pretraining on ZINC: about 7 minutes and 3.3 GB of memory, since all
249,455 parsed molecules are held for the whole run. It writes the one encoder
checkpoint that both the RL agent and the Step 4 regressor start from, with the
featurization hash inside it — a checkpoint whose featurization does not match refuses
to load.

```bash
.venv/bin/python -m mol_optim.plot_pretrain runs/pretrain_zinc.csv --out runs/pretrain_curve.png
```

Loss and accuracy per epoch against the two baselines the run measured for itself: the
element prior, and the same molecules with their atom features shuffled.

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
| `zinc.py` | run once: ZINC 250k in, molecular graphs out |
| `pretrain.py` | masked-atom pretraining on ZINC; writes the shared encoder |
| `oracle_gsk3b.py` | the Step 3 reward: TDC's GSK3B random forest, walked by hand |
| `fetch_gsk3b.py` | run once: TDC's pickle in, `models/gsk3b_forest.npz` out |
| `replay_buffer.py` | ours; ragged, because the target is a max over a candidate set |
| `dqn.py` | the Q network: the encoder plus a head that reads steps remaining |
| `train_dqn.py` | the training loop, flat |
| `baseline_random.py` | tier 0 of the ladder — the number DQN has to beat |
| `molio.py` | SDF in and out; the only place a graph becomes bytes |
| `report.py` | top-k molecules as a drawing and an SDF |
| `plot_run.py` | reward and loss curves from a run log |
| `plot_pretrain.py` | loss and accuracy curves from a pretraining log |

The reference implementations live outside this repo, one directory up:
`MolDQN-pytorch/` (the working base) and `google-research/mol_dqn/` (the original TF1
code, and the published hyperparameters this project follows). Nothing here imports them.
