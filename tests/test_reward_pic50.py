"""The reward's guardrails."""

from pathlib import Path

import pytest
from rdkit import Chem

from mol_optim import regressor, reward_pic50, seeds, splits, zinc

CHECKPOINT = Path("models/egfr_regressor.pt")


@pytest.fixture(scope="module")
def reward() -> reward_pic50.Reward:
    return reward_pic50.load(CHECKPOINT)


def test_a_training_compound_is_its_own_nearest_neighbour(reward, compounds):
    by_key = {c.mol.GetProp("_Name"): c for c in compounds}
    trained_on = by_key[list(by_key)[0]]
    if reward_pic50.nearest_training_similarity(reward, trained_on.mol) < 1.0:
        trained_on = next(
            c for c in compounds if reward_pic50.nearest_training_similarity(reward, c.mol) == 1.0
        )
    assert reward_pic50.nearest_training_similarity(reward, trained_on.mol) == 1.0


def test_molecules_far_from_training_are_zeroed(reward):
    # Methane is nothing like a kinase inhibitor. The model has no opinion there, and an
    # agent paid for one would go looking for exactly that place.
    methane = Chem.MolFromSmiles("C")
    assert reward_pic50.nearest_training_similarity(reward, methane) < reward.domain_floor
    assert reward_pic50.score(reward, methane) == 0.0


def test_the_reward_is_capped_at_the_best_thing_ever_measured(reward, compounds):
    scores = reward_pic50.score_many(reward, [c.mol for c in compounds[:200]])
    assert scores.max() <= reward.ceiling
    assert reward.ceiling == pytest.approx(11.10, abs=0.01)


def test_the_same_molecule_scores_the_same_twice(reward, compounds):
    # Catches dropout left on at eval and nondeterministic pooling.
    mol = compounds[0].mol
    assert reward_pic50.score(reward, mol) == reward_pic50.score(reward, mol)


def test_no_molecule_scores_below_zero(reward, compounds):
    # The environment discounts by discount_factor ** steps_remaining, which flips sign
    # on a negative reward and pays the agent for taking longer.
    scores = reward_pic50.score_many(reward, [c.mol for c in compounds[:200]])
    assert scores.min() >= 0.0


def test_an_empty_molecule_scores_zero(reward):
    assert reward_pic50.score(reward, None) == 0.0
    assert reward_pic50.score(reward, Chem.RWMol().GetMol()) == 0.0


@pytest.mark.slow
def test_ranks_known_inhibitors_above_random_zinc(reward, compounds):
    # The one that means something chemically. Everything else checks plumbing.
    _, test = splits.scaffold_split(
        compounds, 0.2, seeds.held_out_scaffolds(seeds.choose(compounds))
    )
    actives = [c.mol for c in test if c.pic50 >= 8.0]
    decoys = list(zinc.molecules(limit=500))

    active_scores = reward_pic50.score_many(reward, actives)
    decoy_scores = reward_pic50.score_many(reward, decoys)
    assert active_scores.mean() > decoy_scores.mean() + 1.0
    assert regressor.roc_auc(active_scores, decoy_scores) > 0.8
