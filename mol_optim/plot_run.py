"""Reward and loss curves from a training log. Reading, not training.

    .venv/bin/python -m mol_optim.plot_run runs/dqn_qed_aligned.csv --out runs/curve.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on this machine; write files only
import matplotlib.pyplot as plt
import numpy as np


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean; the first `window - 1` points are means of what exists so far."""
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    full = (cumulative[window:] - cumulative[:-window]) / window
    head = cumulative[1:window] / np.arange(1, window)
    return np.concatenate([head, full])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path, nargs="+", help="training CSVs to overlay")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument(
        "--random-baseline",
        type=float,
        default=None,
        help="final mean reward of the random rollout, drawn as a floor",
    )
    args = parser.parse_args()

    figure, (reward_axes, loss_axes) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, height_ratios=[2, 1]
    )

    for log_path in args.logs:
        with open(log_path) as log_file:
            rows = list(csv.DictReader(log_file))
        episodes = np.array([int(row["episode"]) for row in rows])
        rewards = np.array([float(row["reward"]) for row in rows])
        losses = np.array([float(row["mean_loss"]) for row in rows])

        line = reward_axes.plot(
            episodes,
            rolling_mean(rewards, args.window),
            label=f"{log_path.stem} ({args.window}-episode mean)",
        )[0]
        # Raw per-episode reward underneath: the spread is as informative as the mean.
        reward_axes.plot(episodes, rewards, color=line.get_color(), alpha=0.12, lw=0.7)
        loss_axes.plot(episodes, rolling_mean(losses, args.window), color=line.get_color())

    if args.random_baseline is not None:
        reward_axes.axhline(
            args.random_baseline,
            color="0.35",
            ls="--",
            label=f"random baseline ({args.random_baseline:.3f})",
        )

    reward_axes.set_ylabel("terminal QED")
    reward_axes.set_ylim(0, 1)
    reward_axes.legend(loc="lower right", fontsize=9)
    reward_axes.grid(alpha=0.25)
    loss_axes.set_ylabel("MSE loss")
    loss_axes.set_xlabel("episode")
    loss_axes.set_yscale("log")
    loss_axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
