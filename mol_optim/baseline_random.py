"""Tier 0 of the algorithm ladder: uniform random over the candidate set.

This is the number the DQN has to beat. A reward curve that looks like progress but
ties this is a broken agent, and nothing else in the test suite catches that.
"""

import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
from rdkit import Chem

from mol_optim import (
    config,
    determinism,
    environment,
    results,
    rewards,
    seeds,
)


def rollout(cfg: config.Config, reward_fn: Callable[[Chem.Mol], float]) -> results.Run:
    determinism.seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    episode_rewards: list[float] = []
    episode_molecules: list = []
    started = time.perf_counter()

    for _ in range(cfg.episodes):
        episode = environment.reset(cfg)
        while True:
            choice = int(rng.integers(len(episode.valid_actions)))
            result = environment.step(episode, choice, reward_fn, cfg)
            if result.terminated:
                break
        episode_rewards.append(result.reward)
        episode_molecules.append(result.state)

    return results.Run(
        episode_rewards=tuple(episode_rewards),
        episode_molecules=tuple(episode_molecules),
        seconds=time.perf_counter() - started,
    )


def run(settings: config.Settings, spec: config.AgentSpec) -> results.Run:
    reward_fn = rewards.build(spec, settings.bindingdb.path)
    init_mol = seeds.molecule(settings.bindingdb.path, spec.seed_molecule)
    return rollout(replace(spec.cfg, init_mol=init_mol), reward_fn)
