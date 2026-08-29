"""Reward and loss curves from a training log. Reading, not training."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on this machine; write files only
import matplotlib.pyplot as plt
import numpy as np

from mol_optim import config


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean; the first `window - 1` points are means of what exists so far."""
    # A run shorter than the window is the running mean throughout. Without this the
    # default window of 100 cannot plot a run of fewer than 100 episodes.
    window = min(window, len(values))
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    full = (cumulative[window:] - cumulative[:-window]) / window
    head = cumulative[1:window] / np.arange(1, window)
    return np.concatenate([head, full])


def run(settings: config.Settings, spec: config.PlotSpec) -> None:
    figure, (reward_axes, loss_axes) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, height_ratios=[2, 1]
    )

    for log_path in spec.inputs:
        with open(log_path) as log_file:
            rows = list(csv.DictReader(log_file))
        episodes = np.array([int(row["episode"]) for row in rows])
        rewards = np.array([float(row["reward"]) for row in rows])
        losses = np.array([float(row["mean_loss"]) for row in rows])

        line = reward_axes.plot(
            episodes,
            rolling_mean(rewards, spec.window),
            label=f"{log_path.stem} ({spec.window}-episode mean)",
        )[0]
        # Raw per-episode reward underneath: the spread is as informative as the mean.
        reward_axes.plot(episodes, rewards, color=line.get_color(), alpha=0.12, lw=0.7)
        loss_axes.plot(episodes, rolling_mean(losses, spec.window), color=line.get_color())

    if spec.random_baseline is not None:
        reward_axes.axhline(
            spec.random_baseline,
            color="0.35",
            ls="--",
            label=f"random baseline ({spec.random_baseline:.3f})",
        )
    # What the agent gets for taking the no-op every step. Anything below this line is an
    # agent that has damaged its own starting molecule.
    if spec.seed_reward is not None:
        reward_axes.axhline(
            spec.seed_reward,
            color="firebrick",
            ls=":",
            label=f"seed molecule, unedited ({spec.seed_reward:.3f})",
        )

    reward_axes.set_ylabel(spec.ylabel)
    reward_axes.set_ylim(0, 1)
    reward_axes.legend(loc="lower right", fontsize=9)
    reward_axes.grid(alpha=0.25)
    loss_axes.set_ylabel("MSE loss")
    loss_axes.set_xlabel("episode")
    loss_axes.set_yscale("log")
    loss_axes.grid(alpha=0.25)
    figure.tight_layout()
    spec.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(spec.out, dpi=150)
    print(f"wrote {spec.out}")
