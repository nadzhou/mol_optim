"""Tier 0: the uniform-random rollout the DQN's number is measured against.

The baseline is the reference the whole result rests on, so a broken one is worse than
no baseline: a random rollout that quietly stops after one edit reports a floor far
below the real one and makes any agent look like it learned. These are fast — 40-odd
episodes on ethanol with a size reward, no network.
"""

import pytest

from rdkit import Chem

from mol_optim import config
from mol_optim.chem import fragments, graph_key
from mol_optim.agents import random_walk
from mol_optim.report import results
from tests import molecules
from tests.molecules import NAMED

LIBRARY = tuple(
    fragments.Fragment(mol=Chem.MolFromSmiles(s), smiles=s, count=1)
    for s in ("*C", "*OC", "*O", "*c1ccccc1", "*N(C)C")
)

SMALL = config.Config(
    init_mol=NAMED["ethanol"], episodes=20, max_steps_per_episode=4, seed=0
)


@pytest.fixture(scope="module")
def run() -> results.Run:
    return random_walk.rollout(SMALL, molecules.size_reward, LIBRARY)


def test_the_same_seed_gives_the_same_rollout():
    # The floor in the README is a single number from a single seed. If it moves between
    # runs, the margin the agent beats it by is not a measurement.
    first = random_walk.rollout(SMALL, molecules.size_reward, LIBRARY)
    second = random_walk.rollout(SMALL, molecules.size_reward, LIBRARY)
    assert first.episode_rewards == second.episode_rewards
    assert [graph_key.canonical_hash(mol) for mol in first.episode_molecules] == [
        graph_key.canonical_hash(mol) for mol in second.episode_molecules
    ]