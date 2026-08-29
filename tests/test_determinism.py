"""Seeding. Every measurement in this repo is untrustworthy without it."""

from mol_optim import config
from mol_optim.chem import graph_key
from mol_optim.env import environment
from mol_optim.agents import dqn, random_walk
from tests import molecules
from tests.molecules import NAMED

SMALL = config.Config(
    init_mol=NAMED["ethanol"],
    episodes=3,
    max_steps_per_episode=4,
    batch_size=8,
    update_interval=2,
    replay_buffer_size=100,
)


def hashes(run) -> tuple[str, ...]:
    return tuple(graph_key.canonical_hash(mol) for mol in run.episode_molecules)


def test_dqn_run_is_bitwise_reproducible():
    first, second = dqn.train(SMALL, molecules.size_reward), dqn.train(SMALL, molecules.size_reward)
    assert first.episode_rewards == second.episode_rewards
    assert hashes(first) == hashes(second)


def test_random_rollout_is_bitwise_reproducible():
    first, second = random_walk.rollout(SMALL, molecules.size_reward), random_walk.rollout(SMALL, molecules.size_reward)
    assert first.episode_rewards == second.episode_rewards
    assert hashes(first) == hashes(second)


def test_different_seeds_give_different_runs():
    # Guards the two tests above from passing because the seed is ignored and every run
    # is identical.
    seeded = config.Config(
        init_mol=NAMED["ethanol"], episodes=3, max_steps_per_episode=4, seed=1
    )
    assert hashes(random_walk.rollout(SMALL, molecules.size_reward)) != hashes(
        random_walk.rollout(seeded, molecules.size_reward)
    )


def test_candidate_order_is_stable_across_calls():
    # Candidates come back ordered by canonical hash. Set iteration order over strings
    # moves with PYTHONHASHSEED, which would make the argmax depend on the shell.
    cfg = config.Config(init_mol=NAMED["aspirin"])
    first = [
        graph_key.canonical_hash(mol)
        for mol in environment.valid_actions(cfg.init_mol, cfg)
    ]
    second = [
        graph_key.canonical_hash(mol)
        for mol in environment.valid_actions(cfg.init_mol, cfg)
    ]
    assert first == second == sorted(first)
