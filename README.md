# mol_optim

RL molecular derivatization. Design and build order: [plan.md](plan.md). Coding style:
[CLAUDE.md](CLAUDE.md).

Current position: Step 4. Atom-level graph edits, GNN state encoder, TDC's GSK3B
oracle as the sanity reward, the encoder pretrained on ZINC by masked-atom prediction,
and a pIC50 regressor fitted to BindingDB's EGFR IC50 measurements on a scaffold split.
The state is an RDKit molecular graph end to end; molecules become text only in
`report.py`, where a person looks at them, and at the two places published data arrives
as SMILES — `zinc.py` and `fetch_bindingdb.py`.

## Setup

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
.venv/bin/pip install rdkit torch pytest hypothesis matplotlib
.venv/bin/python -m mol_optim.fetch_gsk3b
.venv/bin/python -m mol_optim.zinc
.venv/bin/python -m mol_optim.fetch_bindingdb
```

The third line downloads TDC's published GSK3B oracle once (28 MB) and writes the 2 MB
`models/gsk3b_forest.npz` this repo reads. PyTDC itself is not a dependency — see
`fetch_gsk3b.py` for why. The fourth downloads ZINC 250k (12 MB) to `data/zinc.tab`,
the molecules the encoder is pretrained on, pinned by SHA-256. The fifth downloads a
dated BindingDB snapshot (593 MB, pinned by the MD5 BindingDB publishes beside it) and
writes `data/egfr_ic50.sdf` — every EGFR wild-type IC50 measurement in it, qualified
values dropped, converted to pIC50, duplicate compounds median-aggregated.

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

```bash
.venv/bin/python -m mol_optim.train_regressor --pretrained-encoder models/zinc_encoder.pt --checkpoint models/egfr_regressor.pt
```

Five networks on the EGFR scaffold split, about 11 minutes, reporting test MAE, RMSE and
Spearman against the null of predicting the training mean. Drop `--pretrained-encoder`
for the from-scratch null.

```bash
.venv/bin/python -m mol_optim.plot_regressor models/egfr_regressor.pt --out runs/regressor.png
```

## Docking (optional)

`docking.py` and `fetch_structure.py` need extras the training loop does not:

```bash
brew install boost swig open-babel
```

AutoDock Vina publishes no wheel for Apple Silicon and its `setup.py` looks for Boost
only in a conda prefix or `/usr/local`, then compiles with `-std=c++11`, which Boost 1.92
headers no longer support. Point it at Homebrew and raise the standard:

```bash
pip download vina --no-deps --no-binary :all: -d /tmp/vina && tar xzf /tmp/vina/vina-*.tar.gz -C /tmp/vina && sed -i '' 's/-std=c++11/-std=c++17/' /tmp/vina/vina-*/setup.py && CONDA_DEFAULT_ENV=brew CONDA_PREFIX=$(brew --prefix) pip install meeko scipy gemmi /tmp/vina/vina-*/
```

Then build the receptor once:

```bash
.venv/bin/python -m mol_optim.fetch_structure
```

Open Babel types the receptor rather than Meeko, whose polymer path fails on this
entry's terminal residues. The gating test is the redock:

```bash
.venv/bin/pytest tests/test_docking.py
```

## Layout

| File | What it is |
|---|---|
| `config.py` | every hyperparameter, one frozen dataclass, passed explicitly |
| `graph_key.py` | the state key: a canonical name taken off the graph, no SMILES |
| `environment.py` | the MDP: candidate enumeration by RWMol edits, `step` |
| `featurize.py` | molecular graph to network tensors: int8 codes, one-hot at the network |
| `encoder.py` | the GNN: message passing over atoms and bonds, mean pooled |
| `rewards.py` | QED, the Step 1 and Step 2 target |
| `bindingdb.py` | the EGFR dataset: pIC50 units, aggregation, loading |
| `fetch_bindingdb.py` | run once: BindingDB's 9 GB table in, one target's compounds out |
| `vocabulary.py` | the fragment vocabulary: precedented decorations cut from the actives |
| `splits.py` | scaffold split, and holding the seed scaffolds out of training |
| `regressor.py` | the pIC50 network and the ensemble that predicts with a spread |
| `train_regressor.py` | the regressor training loop, ensemble, and test report |
| `zinc.py` | run once: ZINC 250k in, molecular graphs out |
| `pretrain.py` | masked-atom pretraining on ZINC; writes the shared encoder |
| `finetune_zinc.py` | the Step 3b question: is that checkpoint a better place to start? |
| `oracle_gsk3b.py` | the Step 3 reward: TDC's GSK3B random forest, walked by hand |
| `fetch_gsk3b.py` | run once: TDC's pickle in, `models/gsk3b_forest.npz` out |
| `replay_buffer.py` | ours; ragged, because the target is a max over a candidate set |
| `dqn.py` | the Q network: the encoder plus a head that reads steps remaining |
| `train_dqn.py` | the training loop, flat |
| `baseline_random.py` | tier 0 of the ladder — the number DQN has to beat |
| `molio.py` | SDF in and out; the only place a graph becomes bytes |
| `report.py` | top-k molecules as a drawing and an SDF |
| `audit.py` | what the agent built: motif counts and whether the scaffold survived |
| `docking.py` | AutoDock Vina against a prepared receptor — measured not to rank here |
| `fetch_structure.py` | run once: a PDB entry in, a Vina-ready receptor out |
| `plot_run.py` | reward and loss curves from a run log |
| `plot_pretrain.py` | loss and accuracy curves from a pretraining log |
| `plot_regressor.py` | predicted against measured, and whether disagreement predicts error |
| `seeds.py` | the chemotypes the RL run starts from, held out of the regressor |

The reference implementations live outside this repo, one directory up:
`MolDQN-pytorch/` (the working base) and `google-research/mol_dqn/` (the original TF1
code, and the published hyperparameters this project follows). Nothing here imports them.
