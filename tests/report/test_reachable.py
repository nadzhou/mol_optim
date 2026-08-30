"""The edit lower bound: it has to be sound, or the ceiling it reports is fiction.

Soundness is the whole property. A bound that is too high says an analog is out of
reach when it is not, and the conclusion in docs/where_this_stands.md turns on exactly
that call. So the last test walks the real action space two edits deep and checks the
bound never exceeds the distance BFS actually found.
"""

from rdkit import Chem

from mol_optim import config
from mol_optim.chem import graph_key
from mol_optim.env import environment
from mol_optim.report import reachable
from tests.molecules import NAMED

CFG = config.Config()


def test_no_edits_between_one_molecule_and_itself():
    assert reachable.edit_lower_bound(NAMED["aspirin"], NAMED["aspirin"], CFG) == 0


def test_an_element_the_action_space_cannot_add_is_unreachable():
    # Fluorine is not in atom_types, and no sequence of edits introduces one.
    assert (
        reachable.edit_lower_bound(
            Chem.MolFromSmiles("c1ccccc1"), Chem.MolFromSmiles("Fc1ccccc1"), CFG
        )
        is None
    )


def test_an_element_the_seed_already_carries_is_only_unreachable_in_excess():
    seed = Chem.MolFromSmiles("Fc1ccccc1")
    assert reachable.edit_lower_bound(seed, Chem.MolFromSmiles("Fc1ccccc1C"), CFG) == 1
    assert (
        reachable.edit_lower_bound(seed, Chem.MolFromSmiles("Fc1ccccc1F"), CFG) is None
    )


def test_removals_and_additions_both_count():
    # One O out, two C in.
    assert (
        reachable.edit_lower_bound(
            Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCCC"), CFG
        )
        == 3
    )


def test_bound_never_exceeds_the_distance_the_action_space_actually_walks():
    seed = NAMED["ethanol"]
    seen = {graph_key.canonical_hash(seed)}
    frontier = [seed]
    for depth in (1, 2):
        next_frontier = []
        for mol in frontier:
            for candidate in environment.valid_actions(mol, CFG):
                key = graph_key.canonical_hash(candidate)
                if key not in seen:
                    seen.add(key)
                    next_frontier.append(candidate)
        frontier = next_frontier
        for reached in frontier:
            bound = reachable.edit_lower_bound(seed, reached, CFG)
            assert bound is not None and bound <= depth
