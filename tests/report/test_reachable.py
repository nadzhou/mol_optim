from rdkit import Chem

from mol_optim.chem import fragments, graph_key
from mol_optim.env import environment
from tests.molecules import NAMED

LIBRARY = tuple(
    fragments.Fragment(mol=Chem.MolFromSmiles(s), smiles=s, count=1)
    for s in ("*C", "*OC", "*O")
)


def walk(seed: Chem.Mol, max_depth: int) -> list[set[str]]:
    """Hashes newly reached at each depth, the way reachable._enumerate counts them."""
    seen = {graph_key.canonical_hash(graph_key.normalize(seed))}
    frontier = [seed]
    levels = []
    for _ in range(max_depth):
        reached, next_frontier = set(), []
        for mol in frontier:
            for candidate in environment.valid_actions(mol, LIBRARY):
                key = graph_key.canonical_hash(candidate)
                if key not in seen:
                    seen.add(key)
                    reached.add(key)
                    next_frontier.append(candidate)
        levels.append(reached)
        frontier = next_frontier
    return levels


def test_each_state_is_counted_at_one_depth_only():
    levels = walk(NAMED["ethanol"], 2)
    assert levels[0] and levels[1]
    assert not (levels[0] & levels[1]), "a state counted at two depths inflates the ceiling"


def test_a_deeper_walk_reaches_a_superset():
    shallow = walk(NAMED["ethanol"], 1)
    deep = walk(NAMED["ethanol"], 2)
    assert shallow[0] == deep[0]


def test_the_seed_is_never_counted_as_reached():
    seed = NAMED["ethanol"]
    seed_key = graph_key.canonical_hash(graph_key.normalize(seed))
    for level in walk(seed, 2):
        assert seed_key not in level


def test_one_substituent_swap_is_one_edit():
    """Toluene to anisole is a single action, not the six atom edits it used to be."""
    reached = walk(Chem.MolFromSmiles("Cc1ccccc1"), 1)[0]
    anisole = graph_key.canonical_hash(
        graph_key.normalize(Chem.MolFromSmiles("COc1ccccc1"))
    )
    assert anisole in reached
