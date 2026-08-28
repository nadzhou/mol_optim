# Running things

One path through the pipeline, in the order you would actually run it, ending at
`results/report.md`. Step 4b is optional — it is the algorithm comparison, not part
of the pipeline. All commands
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

## 4b. The same MDP under PPO

```bash
python -m mol_optim.train_ppo --updates 63 --reward pic50 \
  --seed-molecule 0 --max-steps 6 \
  --pretrained-encoder models/zinc_encoder.pt \
  --regressor models/egfr_regressor.pt \
  --log runs/ppo_pic50_seed0.csv --top-k runs/ppo_pic50_seed0_top
```

About 3 minutes. 63 updates of 16 episodes is 1008 episodes — the same environment
budget as step 4, which is what makes the two comparable; PPO's own gradient steps are
counted per update, not per episode.

This is the algorithm arm. Holding the reward fixed and swapping DQN for PPO asks
whether the motifs in the audit are a property of the reward surface or of the
algorithm that searched it. `--reward qed` is not available here: the value head reads
a state graph, and a QED run starts from the empty molecule, which has none.

## 4c. Constraining the action space

```bash
python -m mol_optim.train_dqn --episodes 1000 --reward pic50 \
  --seed-molecule 0 --max-steps 3 \
  --pretrained-encoder models/zinc_encoder.pt --regressor models/egfr_regressor.pt \
  --fragments data/egfr_fragments.sdf --forbid-acyclic-nn \
  --log runs/dqn_frag_seed0.csv --top-k runs/dqn_frag_seed0_top
```

`--fragments` adds fragment-attachment actions from the vocabulary alongside the
atom-level edits; `--forbid-acyclic-nn` drops candidates carrying a non-ring N-N bond.
Together they take the action space at the seed from 40 candidates to 534.

**`--max-steps 3`, not 6.** A fragment attachment adds up to 12 heavy atoms where an
atom edit adds one, so six of them reach 59 heavy atoms from a 19-atom seed and leave
the regressor's applicability domain entirely — the reward goes to zero and the run
learns nothing. Three fragment edits is the same size change as roughly six atom edits.

Slower per step: 534 candidates are scored every step, and the DQN target is a max over
each replayed next state's candidate set, so a gradient step reads ~69,000 graphs
against the baseline's ~5,000. Expect ~30 minutes rather than ~5.

## 5. Look at what it built

A climbing reward curve is not the result. These are:

```bash
python -m mol_optim.audit runs/pilot_top.sdf --seed-molecule 0
python -m mol_optim.plot_run runs/pilot.csv --out runs/pilot.png --random-baseline 0.331
```

`audit` counts the substructures past runs have gone wrong in and checks the seed's
scaffold survived. `--top-k` above already wrote the best distinct molecules as a drawing
and an SDF.

The other two plotting scripts read the checkpoints from steps 2 and 3. Point them at
`results/` directly — a figure copied by hand is a figure that goes stale:

```bash
python -m mol_optim.plot_pretrain runs/pretrain_zinc.csv \
  --out results/zinc-pretraining/pretrain_curve.png
python -m mol_optim.plot_regressor models/egfr_regressor.pt \
  --out results/pic50-regressor/regressor.png
```

## 6. The report

```bash
python -m mol_optim.report
```

Seconds. Reads every `runs/pilot_pic50_seed*_top.sdf`, the QED control's top-k and both
regressor checkpoints, and writes [`results/report.md`](../results/report.md): per seed,
the top-k with predicted pIC50 against that seed's own measured value, SA score, Tanimoto
to the seed, and the audit columns beside them. Nothing is recomputed — if a seed has no
run, the report says so rather than quietly reporting four of five.

It takes checkpoints already on disk, so re-running it after any step above is cheap and
is the way to keep `results/` honest.

## Tests

```bash
pytest -m "not slow"   # 106 tests, about 12 seconds
pytest                 # adds two 5000-episode runs and a ZINC pretraining run
```
