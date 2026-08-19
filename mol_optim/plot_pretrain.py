"""Curves from a ZINC pretraining log. Reading, not training.

    .venv/bin/python -m mol_optim.plot_pretrain \
        runs/pretrain_zinc.csv --out runs/pretrain_curve.png

Its own script rather than a flag on plot_run.py: a pretraining log and a DQN log share
no columns, and one plotter with a branch for each would be two plotters sharing an
argument parser.

Every reference line is read out of the log, not passed in. The prior — the loss of
guessing ZINC's element distribution and never looking at the graph — is what says
whether the run learned anything, and a baseline typed on the command line is a baseline
that can be typed wrong.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on this machine; write files only
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with open(args.log) as log_file:
        rows = list(csv.DictReader(log_file))
    def column(name: str) -> np.ndarray:
        return np.array([float(row[name]) for row in rows])

    epochs = column("epoch") + 1  # the log counts from 0; a reader counts passes

    figure, (loss_axes, accuracy_axes) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, height_ratios=[2, 1]
    )

    # Train is a mean over the epoch's batches and held-out is measured once at the end
    # of it, which is why the first epoch's train loss sits above its held-out loss.
    loss_axes.plot(
        epochs,
        column("train_loss"),
        label="train (mean over the epoch)",
        marker="o",
        ms=3,
    )
    loss_axes.plot(epochs, column("holdout_loss"), label="held out", marker="o", ms=3)
    loss_axes.plot(
        epochs,
        column("control_loss"),
        label="held out, atom features shuffled",
        marker="o",
        ms=3,
        color="0.55",
    )
    prior_loss = column("prior_loss")[0]
    loss_axes.axhline(
        prior_loss,
        color="0.35",
        ls="--",
        label=f"element prior, no graph read ({prior_loss:.3f})",
    )

    accuracy_axes.plot(
        epochs, column("holdout_accuracy"), label="held out", marker="o", ms=3
    )
    accuracy_axes.plot(
        epochs,
        column("control_accuracy"),
        label="atom features shuffled",
        marker="o",
        ms=3,
        color="0.55",
    )
    prior_accuracy = column("prior_accuracy")[0]
    accuracy_axes.axhline(
        prior_accuracy,
        color="0.35",
        ls="--",
        label=f"always the commonest element ({prior_accuracy:.3f})",
    )

    loss_axes.set_ylabel("masked-element cross-entropy (nats)")
    loss_axes.set_ylim(0, 1.05 * max(prior_loss, column("train_loss").max()))
    loss_axes.legend(loc="center right", fontsize=9)
    loss_axes.grid(alpha=0.25)
    accuracy_axes.set_ylabel("masked atoms named correctly")
    accuracy_axes.set_xlabel("epoch over ZINC")
    accuracy_axes.set_ylim(0.7, 1.0)
    accuracy_axes.set_xticks(epochs)  # epochs are whole passes; 1.25 of one is not a thing
    accuracy_axes.legend(loc="center right", fontsize=9)
    accuracy_axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
