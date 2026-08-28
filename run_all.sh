#!/usr/bin/env bash
# The whole pipeline, from an empty checkout to results/report.md.
#
# About 25 minutes of compute after the downloads. Each step writes a checkpoint the next
# one reads, so you can comment out what you have already run.
#
# docs/running.md is this same sequence, one step at a time, with the flags explained.

set -euo pipefail
cd "$(dirname "$0")"

mkdir -p runs models

echo "==> fetching ZINC (12 MB) and BindingDB EGFR (593 MB)"
python -m mol_optim.zinc
python -m mol_optim.fetch_bindingdb

echo "==> pretraining the encoder on ZINC (~7 min)"
python -m mol_optim.pretrain --epochs 10 \
  --checkpoint models/zinc_encoder.pt --log runs/pretrain_zinc.csv

echo "==> training the pIC50 regressor (~11 min)"
python -m mol_optim.train_regressor \
  --pretrained-encoder models/zinc_encoder.pt \
  --checkpoint models/egfr_regressor.pt

echo "==> the random floor the agent has to beat (~2 min)"
python -m mol_optim.baseline_random --episodes 1000 --reward pic50 \
  --seed-molecule 0 --max-steps 6 \
  --regressor models/egfr_regressor.pt | tee runs/random_pic50.out

echo "==> training the agent against predicted pIC50 (~5 min)"
python -m mol_optim.train_dqn --episodes 1000 --reward pic50 \
  --seed-molecule 0 --max-steps 6 \
  --pretrained-encoder models/zinc_encoder.pt \
  --regressor models/egfr_regressor.pt \
  --log runs/pilot.csv --checkpoint runs/pilot.pt --top-k runs/pilot_pic50_seed0_top | tee runs/pilot.out

echo "==> what did it actually build?"
python -m mol_optim.audit runs/pilot_pic50_seed0_top.sdf --seed-molecule 0 | tee runs/pilot_audit.out

echo "==> drawing the curves, straight into results/"
python -m mol_optim.plot_pretrain runs/pretrain_zinc.csv \
  --out results/zinc-pretraining/pretrain_curve.png
python -m mol_optim.plot_regressor models/egfr_regressor.pt \
  --out results/pic50-regressor/regressor.png
python -m mol_optim.plot_run runs/pilot.csv \
  --out results/pic50-agent/pilot_pic50_seed0.png --random-baseline 0.331

echo "==> the report"
python -m mol_optim.report --out results/report.md

echo
echo "done. results/report.md is the deliverable; the audit above and the tables in it"
echo "are the part that says whether the reward curve means anything."
