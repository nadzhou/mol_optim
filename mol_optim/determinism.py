"""Seeding. Step 0 of plan.md — nothing below it is trustworthy without this."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seeds every source of randomness the training loop touches.

    PYTHONHASHSEED is set for child processes only; it cannot affect this already
    running interpreter. String hash randomization would otherwise reach us through
    set iteration order, so environment.valid_actions returns a sorted tuple instead
    of a set — that is the actual fix, and this line is belt-and-braces for scripts
    we spawn.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
