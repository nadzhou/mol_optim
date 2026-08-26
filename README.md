# mol_optim

A reinforcement learning agent that improves a molecule by editing it — add an atom here,
close a ring there — and, at every step, a measurement of whether it is really learning
chemistry or just gaming the thing that scores it.

The setting is lead optimization, not design from nothing. The agent starts from a
compound that already works and makes small changes to it, which is what a medicinal
chemist does. The molecule stays an RDKit graph the whole way through; it only becomes
text where a person reads it (`report.py`) or where published data arrives as SMILES
(`zinc.py`, `fetch_bindingdb.py`).

**Where it is now.** Steps 0–4 are done and measured: atom-level graph edits, a GNN state
encoder, TDC's GSK3B oracle as a sanity reward, that encoder pretrained on ZINC by
masked-atom prediction, and a pIC50 regressor fitted to BindingDB's EGFR measurements on a
scaffold split. Step 5 — the regressor as the reward — has a pilot run, not a deliverable.
The design and the full build order are in [plan.md](plan.md); the coding rules are in
[CLAUDE.md](CLAUDE.md).

## Results

**[Every figure, in build order, with the number it carries →](results/README.md)**

The short version. On QED the agent reaches 0.895 against a random baseline of 0.145.
Swapping the fingerprint encoder for a GNN ties that score using 56k parameters instead of
2.7M. On TDC's GSK3B oracle it reaches 0.610 against a random floor of 0.077. Pretraining
on ZINC names masked atoms correctly 92.9% of the time against a 73.6% prior, and — the
comparison that mattered — carries over to real assay data: the EGFR regressor tests at
0.806 MAE and 0.642 Spearman, against 0.868 and 0.582 for the same network trained from
scratch.

The more interesting result is that the agent games every scorer it is given. Against QED
it built strained fused heterocycles. Against the GSK3B oracle it stacked one real
hinge-binding motif three times over. Against the pIC50 regressor it found a molecule that
looks entirely plausible, which is the harder version of the problem. Each of those is
written up where it happened.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install rdkit torch pytest hypothesis matplotlib
```

Then fetch the three datasets. Each runs once and is pinned by checksum, so a silently
changed upstream file fails loudly instead of quietly moving your numbers.

```bash
python -m mol_optim.fetch_gsk3b      # 28 MB  → models/gsk3b_forest.npz
python -m mol_optim.zinc             # 12 MB  → data/zinc.tab
python -m mol_optim.fetch_bindingdb  # 593 MB → data/egfr_ic50.sdf
```

The first converts TDC's published GSK3B oracle into the 2 MB array this repo reads —
PyTDC itself is not a dependency, and `fetch_gsk3b.py` explains why. The second is ZINC
250k, the molecules the encoder pretrains on. The third scans a dated BindingDB snapshot
for every wild-type EGFR IC50 measurement, drops qualified values, converts to pIC50, and
median-aggregates compounds measured more than once.

Every command below assumes the virtual environment is active.

## Tests

```bash
pytest -m "not slow"
```

177 tests in about 17 seconds. They are the point, not an afterthought: ML bugs are silent,
so a wrong reward curve looks like a slow learner. The slow suite adds four 5000-episode
training runs, a step-latency measurement, and a short ZINC pretraining run.

## Running things

Train the agent. `--reward qed` is the Step 1 and 2 target, `--reward gsk3b` is Step 3's
oracle, and `baseline_random.py` takes the same flag to give you the number to beat.

```bash
python -m mol_optim.train_dqn --episodes 5000 --reward gsk3b \
  --log runs/dqn.csv --checkpoint runs/dqn.pt --top-k runs/dqn_top
```

`--top-k` writes the best distinct molecules as a drawing and an SDF.
`--pretrained-encoder models/zinc_encoder.pt` starts from the pretrained encoder rather
than a random one.

Pretrain that encoder — about 7 minutes and 3.3 GB of memory, since all 249,455 molecules
stay in memory for the whole run. The checkpoint carries a hash of the featurization, and
refuses to load if it does not match.

```bash
python -m mol_optim.pretrain --epochs 10 \
  --checkpoint models/zinc_encoder.pt --log runs/pretrain_zinc.csv
```

Train the pIC50 regressor — five networks on the EGFR scaffold split, about 11 minutes,
reporting test MAE, RMSE and Spearman against the null of predicting the training mean.
Drop `--pretrained-encoder` for the from-scratch comparison.

```bash
python -m mol_optim.train_regressor \
  --pretrained-encoder models/zinc_encoder.pt --checkpoint models/egfr_regressor.pt
```

Draw any of it:

```bash
python -m mol_optim.plot_run runs/dqn.csv --out runs/curve.png --random-baseline 0.146
python -m mol_optim.plot_pretrain runs/pretrain_zinc.csv --out runs/pretrain_curve.png
python -m mol_optim.plot_regressor models/egfr_regressor.pt --out runs/regressor.png
```

Runs write into `runs/`, which is not tracked. The plots worth keeping are copied into
`results/`.

## Docking (optional)

Docking is an evaluation spot-check and never runs in the training loop, so its
dependencies are optional — and awkward, because AutoDock Vina ships no wheel for Apple
Silicon. Its `setup.py` looks for Boost only in a conda prefix or `/usr/local`, then
compiles with `-std=c++11`, which Boost 1.92 headers no longer accept. Point it at
Homebrew and raise the standard:

```bash
brew install boost swig open-babel
pip download vina --no-deps --no-binary :all: -d /tmp/vina && tar xzf /tmp/vina/vina-*.tar.gz -C /tmp/vina && sed -i '' 's/-std=c++11/-std=c++17/' /tmp/vina/vina-*/setup.py && CONDA_DEFAULT_ENV=brew CONDA_PREFIX=$(brew --prefix) pip install meeko scipy gemmi /tmp/vina/vina-*/
```

Build the receptor once with `python -m mol_optim.fetch_structure`. Open Babel types it
rather than Meeko, whose polymer path fails on this entry's terminal residues. The gating
test is the redock: `pytest tests/test_docking.py`.

Worth knowing before you invest in this: docking was measured on this target and does
**not** rank known EGFR compounds correctly. See plan.md.

## Layout

Start with `environment.py` and `train_dqn.py` — between them they hold the whole loop.

| File | What it is |
|---|---|
| `determinism.py` | Step 0: seeds every source of randomness the loop touches |
| `config.py` | every hyperparameter, one frozen dataclass, passed explicitly |
| `graph_key.py` | the state key: a canonical name taken off the graph, no SMILES |
| `environment.py` | the MDP: candidate enumeration by RWMol edits, `step` |
| `featurize.py` | molecular graph to network tensors: int8 codes, one-hot at the network |
| `encoder.py` | the GNN: message passing over atoms and bonds, mean pooled |
| `rewards.py` | QED, the Step 1 and Step 2 target |
| `reward_pic50.py` | the Step 5 reward: the regressor, behind three guardrails |
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
| `results.py` | what a run returns; shared by the DQN loop and the baseline |
| `molio.py` | SDF in and out; the only place a graph becomes bytes |
| `report.py` | top-k molecules as a drawing and an SDF |
| `audit.py` | what the agent built: motif counts and whether the scaffold survived |
| `docking.py` | AutoDock Vina against a prepared receptor — measured not to rank here |
| `fetch_structure.py` | run once: a PDB entry in, a Vina-ready receptor out |
| `plot_run.py` | reward and loss curves from a run log |
| `plot_pretrain.py` | loss and accuracy curves from a pretraining log |
| `plot_regressor.py` | predicted against measured, and whether disagreement predicts error |
| `seeds.py` | the chemotypes the RL run starts from, held out of the regressor |

## Prior work

Two reference implementations, cloned one directory up and imported by nothing here:
[MolDQN-pytorch](https://github.com/aksub99/MolDQN-pytorch), the working base, and
[google-research/mol_dqn](https://github.com/google-research/google-research/tree/master/mol_dqn),
the original TF1 code and the source of the published hyperparameters this project
follows. CLAUDE.md cites the first as a set of anti-examples, and Step 1 of plan.md
measures what its defaults cost: 0.19 of QED.
