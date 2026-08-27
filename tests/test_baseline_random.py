"""Tier 0: the uniform-random rollout the DQN's number is measured against.

The baseline is the reference the whole result rests on, so a broken one is worse than
no baseline: a random rollout that quietly stops after one edit reports a floor far
below the real one and makes any agent look like it learned. These are fast — 40-odd
episodes on ethanol with QED, no network.
"""

import pytest

from mol_optim import baseline_random, config, graph_key, results, rewards
from tests.molecules import NAMED

SMALL = config.Config(
    init_mol=NAMED["ethanol"], episodes=20, max_steps_per_episode=4, seed=0
)


@pytest.fixture(scope="module")
def run() -> results.Run:
    return baseline_random.rollout(SMALL, rewards.qed)


def test_the_same_seed_gives_the_same_rollout():
    # The floor in the README is a single number from a single seed. If it moves between
    # runs, the margin the agent beats it by is not a measurement.
    first = baseline_random.rollout(SMALL, rewards.qed)
    second = baseline_random.rollout(SMALL, rewards.qed)
    assert first.episode_rewards == second.episode_rewards
    assert [graph_key.canonical_hash(mol) for mol in first.episode_molecules] == [
        graph_key.canonical_hash(mol) for mol in second.episode_molecules
    ]