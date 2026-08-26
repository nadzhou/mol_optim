# Running things

All commands assume the virtual environment is active (`source .venv/bin/activate`).
Runs write into `runs/`, which is not tracked; the plots worth keeping are copied into
[`results/`](../results/README.md).

## Train the agent

```bash
python -m mol_optim.train_dqn --episodes 5000 --reward gsk3b \
  --log runs/dqn.csv --checkpoint runs/dqn.pt --top-k runs/dqn_top
```

`--reward qed` is the Step 1 and 2 target; `--reward gsk3b` is Step 3's oracle.
`baseline_random.py` takes the same flag and gives you the number to beat. `--top-k`
writes the best distinct molecules as a drawing and an SDF. `--pretrained-encoder
models/zinc_encoder.pt` starts from the pretrained encoder rather than a random one.

5000 episodes took 93 minutes with the fingerprint encoder on QED, 147 with the GNN on
QED, and 128 with the GNN on the GSK3B oracle. Throughput falls over a run — about 70
steps/s down to 23 — because the agent builds larger molecules and the candidate sets
grow with them.

## Pretrain the encoder

```bash
python -m mol_optim.pretrain --epochs 10 \
  --checkpoint models/zinc_encoder.pt --log runs/pretrain_zinc.csv
```

About 7 minutes and 3.3 GB of memory, since all 249,455 molecules stay in memory for the
whole run. The checkpoint carries a hash of the featurization and refuses to load if it
does not match — which is what stops a silently changed feature table from being loaded
into a network trained against the old one.

## Train the pIC50 regressor

```bash
python -m mol_optim.train_regressor \
  --pretrained-encoder models/zinc_encoder.pt --checkpoint models/egfr_regressor.pt
```

Five networks on the EGFR scaffold split, about 11 minutes, reporting test MAE, RMSE and
Spearman against the null of predicting the training mean. Drop `--pretrained-encoder`
for the from-scratch comparison.

## Draw any of it

```bash
python -m mol_optim.plot_run runs/dqn.csv --out runs/curve.png --random-baseline 0.146
python -m mol_optim.plot_pretrain runs/pretrain_zinc.csv --out runs/pretrain_curve.png
python -m mol_optim.plot_regressor models/egfr_regressor.pt --out runs/regressor.png
```

## Tests

```bash
pytest -m "not slow"   # 177 tests, about 17 seconds
pytest                 # adds four 5000-episode runs and a ZINC pretraining run
```
