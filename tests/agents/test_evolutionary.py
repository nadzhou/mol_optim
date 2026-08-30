"""The baseline's one load-bearing property: it stays inside the DQN's reachable set.

If an individual can drift further than max_steps_per_episode edits from the seed, the
comparison is between two different search spaces and says nothing about search strength.
Budget parity is the other half — the same number of reward evaluations as the DQN, or
the gap is a budget difference wearing an algorithm's name.
"""

from dataclasses import replace

from rdkit import Chem

from mol_optim import config
from mol_optim.agents import evolutionary
from mol_optim.chem import fragments, graph_key
from mol_optim.env import environment
from tests import molecules
from tests.molecules import NAMED

LIBRARY = tuple(
    fragments.Fragment(mol=Chem.MolFromSmiles(s), smiles=s, count=1)
    for s in ("*C", "*OC", "*O", "*c1ccccc1", "*N(C)C")
)

CFG = config.Config(
    init_mol=NAMED["ethanol"], max_steps_per_episode=3, episodes=100, seed=0
)


def _reachable_within(seed, depth, cfg):
    seen = {graph_key.canonical_hash(seed)}
    frontier = [seed]
    for _ in range(depth):
        next_frontier = []
        for mol in frontier:
            for candidate in environment.valid_actions(mol, LIBRARY):
                key = graph_key.canonical_hash(candidate)
                if key not in seen:
                    seen.add(key)
                    next_frontier.append(candidate)
        frontier = next_frontier
    return seen


def test_every_molecule_it_builds_is_inside_the_budget():
    run = evolutionary.search(CFG, molecules.size_reward, LIBRARY)
    reachable = _reachable_within(CFG.init_mol, CFG.max_steps_per_episode, CFG)
    for mol in run.episode_molecules:
        assert graph_key.canonical_hash(mol) in reachable


def test_it_spends_the_same_reward_budget_as_the_dqn():
    run = evolutionary.search(CFG, molecules.size_reward, LIBRARY)
    assert len(run.episode_rewards) == CFG.episodes


def test_it_is_reproducible_and_the_seed_changes_it():
    same = evolutionary.search(CFG, molecules.size_reward, LIBRARY).episode_rewards
    assert same == evolutionary.search(CFG, molecules.size_reward, LIBRARY).episode_rewards
    other = evolutionary.search(
        replace(CFG, seed=1), molecules.size_reward, LIBRARY
    ).episode_rewards
    assert other != same
