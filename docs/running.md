# Running things

One path through the pipeline, in the order you would actually run it. All commands
assume the virtual environment is active (`source .venv/bin/activate`).

`./run_all.sh` runs every step below in sequence. This file exists for when you want to
run one of them on its own, or change a flag.

Runs write into `runs/`, which is not tracked; the plots worth keeping are copied into
[`results/`](../results/README.md).

## 1. Fetch the data

```bash
python -m mol_optim.zinc             # 12 MB  → data/zinc.tab
python -m mol_optim.fetch_bindingdb  # 593 MB → data/egfr_ic50.sdf
```

Each downloads once and is pinned by checksum, so a silently changed upstream file fails
loudly instead of quietly moving your numbers. BindingDB arrives as a 9 GB table and is
filtered down to one target's compounds.

The fragment vocabulary is committed as `data/egfr_fragments.sdf`, because a run is not
reproducible without the action space it was measured on. Rebuild it with
`python -m mol_optim.vocabulary` if you change the dataset.

## 2. Pretrain the encoder on ZINC

```bash
python -m mol_optim.pretrain --epochs 10 \
  --checkpoint models/zinc_encoder.pt --log runs/pretrain_zinc.csv
```

About 7 minutes and 3.3 GB of memory, since all 249,455 molecules stay in memory for the
whole run. The checkpoint carries a hash of the featurization and refuses to load if it
does not match — which is what stops a silently changed feature table from being loaded
into a network trained against the old one.

## 3. Train the pIC50 regressor

```bash
python -m mol_optim.train_regressor \
  --pretrained-encoder models/zinc_encoder.pt --checkpoint models/egfr_regressor.pt
```

Five networks on the EGFR scaffold split, about 11 minutes, reporting test MAE, RMSE and
Spearman against the null of predicting the training mean. Drop `--pretrained-encoder`
for the from-scratch comparison — that is the ablation, and it is worth about 0.06 MAE.

## 4. Train the agent

```bash
python -m mol_optim.train_dqn --episodes 1000 --reward pic50 \
  --seed-molecule 0 --max-steps 6 \
  --pretrained-encoder models/zinc_encoder.pt \
  --regressor models/egfr_regressor.pt \
  --log runs/pilot.csv --checkpoint runs/pilot.pt --top-k runs/pilot_top
```

About 5 minutes. `--seed-molecule` indexes the five seed scaffolds the run can start
from, and each is evaluated against its own measured pIC50. `--max-steps 6` is
deliberate: at 40 edits the agent leaves the regressor's applicability domain and the
prediction stops meaning anything.

`baseline_random.py` takes the same flags and gives you the number to beat:

```bash
python -m mol_optim.baseline_random --episodes 1000 --reward pic50 \
  --seed-molecule 0 --max-steps 6 --regressor models/egfr_regressor.pt
```

`--reward qed` swaps in RDKit's drug-likeness score. That is the control, not the target:
it needs no regressor and no seed molecule, and what it shows is an agent building
molecules that score 0.93 and cannot exist.

## 5. Look at what it built

A climbing reward curve is not the result. These are:

```bash
python -m mol_optim.audit runs/pilot_top.sdf --seed-molecule 0
python -m mol_optim.plot_run runs/pilot.csv --out runs/pilot.png --random-baseline 0.331
```

`audit` counts the substructures past runs have gone wrong in and checks the seed's
scaffold survived. `--top-k` above already wrote the best distinct molecules as a drawing
and an SDF.

The other two plotting scripts read the checkpoints from steps 2 and 3:

```bash
python -m mol_optim.plot_pretrain runs/pretrain_zinc.csv --out runs/pretrain_curve.png
python -m mol_optim.plot_regressor models/egfr_regressor.pt --out runs/regressor.png
```

## Tests

```bash
pytest -m "not slow"   # 166 tests, about 15 seconds
pytest                 # adds two 5000-episode runs and a ZINC pretraining run
```
