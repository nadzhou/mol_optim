"""Discount arithmetic. An off-by-one here reweights all training."""

import pytest

from rdkit import Chem

from mol_optim import config
from mol_optim.chem import fragments
from mol_optim.env import environment
from tests import molecules
from tests.molecules import NAMED

LIBRARY = tuple(
    fragments.Fragment(mol=Chem.MolFromSmiles(s), smiles=s, count=1)
    for s in ("*C", "*OC", "*O", "*c1ccccc1", "*N(C)C")
)

ASPIRIN = NAMED["aspirin"]


def test_discount_applied_from_steps_remaining():
    # Hand-computed: a 5-step episode, reward collected after 3 steps, so two steps of
    # discount remain.
    cfg = config.Config(init_mol=ASPIRIN, max_steps_per_episode=5, discount_factor=0.9)
    episode = environment.reset(cfg, LIBRARY)
    for _ in range(3):
        result = environment.step(episode, 0, molecules.size_reward, cfg, LIBRARY)
    assert result.reward == pytest.approx(molecules.size_reward(result.state) * 0.9**2)
    assert not result.terminated


def test_terminal_reward_is_undiscounted():
    cfg = config.Config(init_mol=ASPIRIN, max_steps_per_episode=2, discount_factor=0.9)
    episode = environment.reset(cfg, LIBRARY)
    for _ in range(2):
        result = environment.step(episode, 0, molecules.size_reward, cfg, LIBRARY)
    assert result.terminated
    assert result.reward == pytest.approx(molecules.size_reward(result.state))


def test_episode_terminates_at_exactly_max_steps():
    cfg = config.Config(init_mol=ASPIRIN, max_steps_per_episode=4)
    episode = environment.reset(cfg, LIBRARY)
    for step_index in range(4):
        result = environment.step(episode, 0, molecules.size_reward, cfg, LIBRARY)
        assert result.terminated == (step_index == 3)
    assert episode.num_steps_taken == 4


def test_the_reward_function_is_what_the_environment_calls():
    # The reward arrives as an argument, not as a subclass override. Swapping it is how
    # the pIC50 regressor gets put in.
    cfg = config.Config(init_mol=ASPIRIN, max_steps_per_episode=1, discount_factor=0.9)
    episode = environment.reset(cfg, LIBRARY)
    result = environment.step(episode, 0, lambda mol: 7.0, cfg, LIBRARY)
    assert result.reward == pytest.approx(7.0)
