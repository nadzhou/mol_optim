"""The reward's guardrails."""

from pathlib import Path

import pytest
from rdkit import Chem

from mol_optim.chem import seeds, splits
from mol_optim.datasets import bindingdb, zinc
from mol_optim.nets import regressor
from mol_optim.env import rewards
from tests import conftest

CHECKPOINT = Path("models/egfr_regressor.pt")


@pytest.fixture(scope="module")
def reward() -> rewards.Reward:
    # The reward carries the training set with it, for the nearest-neighbour distance.
    conftest.require(conftest.BINDINGDB_PATH, conftest.BUILD_IT)
    return rewards.load(CHECKPOINT, conftest.BINDINGDB_PATH)


def test_molecules_far_from_training_are_zeroed(reward):
    # Methane is nothing like a kinase inhibitor. The model has no opinion there, and an
    # agent paid for one would go looking for exactly that place.
    methane = Chem.MolFromSmiles("C")
    assert rewards.nearest_training_similarity(reward, methane) < reward.domain_floor
    assert rewards.score(reward, methane) == 0.0


def test_the_reward_is_capped_at_the_best_thing_ever_measured(reward, compounds):
    scores = rewards.score_many(reward, [c.mol for c in compounds[:200]])
    assert scores.max() <= reward.ceiling
    assert reward.ceiling == pytest.approx(11.10, abs=0.01)


def test_no_molecule_scores_below_zero(reward, compounds):
    # The environment discounts by discount_factor ** steps_remaining, which flips sign
    # on a negative reward and pays the agent for taking longer.
    scores = rewards.score_many(reward, [c.mol for c in compounds[:200]])
    assert scores.min() >= 0.0


@pytest.mark.slow
def test_ranks_known_inhibitors_above_random_zinc(reward, compounds):
    # The one that means something chemically. Everything else checks plumbing.
    _, test = splits.scaffold_split(
        compounds, 0.2, seeds.held_out_scaffolds(seeds.choose(compounds))
    )
    actives = [c.mol for c in test if c.pic50 >= 8.0]
    conftest.require(conftest.ZINC_PATH, conftest.BUILD_IT)
    decoys = list(zinc.molecules(conftest.ZINC_PATH, limit=500))

    active_scores = rewards.score_many(reward, actives)
    decoy_scores = rewards.score_many(reward, decoys)
    assert active_scores.mean() > decoy_scores.mean() + 1.0
    assert regressor.roc_auc(active_scores, decoy_scores) > 0.8
