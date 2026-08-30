"""Environment invariants.

The action space is the foundation: if it emits an invalid graph, everything downstream
is garbage, and nothing crashes when it does. Random *action sequences* are where the
bugs are, so the fuzz test walks whole episodes.
"""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from rdkit import Chem

from mol_optim import config
from mol_optim.chem import graph_key
from mol_optim.env import environment
from tests import molecules
from tests.molecules import NAMED, START_MOLECULES


def assert_action_set_is_sane(actions: tuple[Chem.Mol, ...]) -> None:
    assert actions, "no valid actions — the agent would deadlock here"
    for mol in actions:
        Chem.SanitizeMol(Chem.Mol(mol))  # raises on bad valence
        assert len(Chem.GetMolFrags(mol)) == 1, "disconnected candidate"
        rebuilt = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol))
        assert graph_key.canonical_hash(rebuilt) == graph_key.canonical_hash(
            mol
        ), "candidate's key depends on how it was built"
    hashes = [graph_key.canonical_hash(mol) for mol in actions]
    assert len(set(hashes)) == len(hashes), "duplicate candidates inflate the argmax"
    assert hashes == sorted(hashes), "candidate order is not reproducible"


@pytest.mark.parametrize("start", START_MOLECULES, ids=lambda m: m.GetProp("_Name"))
def test_action_set_is_sane_for_each_start_molecule(start):
    assert_action_set_is_sane(environment.valid_actions(start, config.Config()))


@given(
    start_index=st.integers(min_value=0, max_value=len(START_MOLECULES) - 1),
    moves=st.lists(st.integers(min_value=0, max_value=10**6), min_size=1, max_size=4),
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_random_action_sequences_keep_every_invariant(start_index, moves):
    cfg = config.Config(
        init_mol=START_MOLECULES[start_index], max_steps_per_episode=len(moves)
    )
    episode = environment.reset(cfg)
    for move in moves:
        assert_action_set_is_sane(episode.valid_actions)
        result = environment.step(
            episode, move % len(episode.valid_actions), molecules.size_reward, cfg
        )
    assert_action_set_is_sane(episode.valid_actions)
    assert episode.num_steps_taken == len(moves)
    assert result.terminated


@pytest.mark.parametrize(
    "start",
    [m for m in START_MOLECULES if m.GetNumAtoms() > 1],
    ids=lambda m: m.GetProp("_Name"),
)
def test_adding_an_atom_is_reversible(start):
    # Every action that grows the molecule by one atom must have the original graph back
    # in its own action set. Asymmetric edit logic traps the agent.
    #
    # Single-atom starts are excluded: undoing methane -> HCN splits into two lone atoms,
    # and the environment keeps the larger fragment, which for a tie is the wrong one.
    cfg = config.Config()
    start_hash = graph_key.canonical_hash(start)
    for action in environment.valid_actions(start, cfg):
        if action.GetNumAtoms() != start.GetNumAtoms() + 1:
            continue
        undo_hashes = {
            graph_key.canonical_hash(mol)
            for mol in environment.valid_actions(action, cfg)
        }
        assert start_hash in undo_hashes


def action_hashes(mol, cfg) -> set[str]:
    return {graph_key.canonical_hash(m) for m in environment.valid_actions(mol, cfg)}


def test_allowed_ring_sizes_is_threaded_through():
    # Hexane: bonding atom 0 to atom 5 closes a six-ring, atom 0 to atom 2 a three-ring.
    hexane = NAMED["hexane"]
    six_only = action_hashes(hexane, config.Config(allowed_ring_sizes=(6,)))
    three_only = action_hashes(hexane, config.Config(allowed_ring_sizes=(3,)))
    cyclohexane = graph_key.canonical_hash(NAMED["cyclohexane"])
    propylcyclopropane = graph_key.canonical_hash(NAMED["propylcyclopropane"])
    assert cyclohexane in six_only and cyclohexane not in three_only
    assert propylcyclopropane in three_only and propylcyclopropane not in six_only


def test_step_rejects_a_step_past_termination():
    cfg = config.Config(init_mol=NAMED["ethanol"], max_steps_per_episode=1)
    episode = environment.reset(cfg)
    assert environment.step(episode, 0, molecules.size_reward, cfg).terminated
    with pytest.raises(ValueError, match="terminated"):
        environment.step(episode, 0, molecules.size_reward, cfg)


def test_empty_start_offers_one_atom_of_each_type():
    cfg = config.Config(init_mol=None, atom_types=("C", "O", "N"))
    actions = environment.valid_actions(None, cfg)
    assert len(actions) == 3
    assert {mol.GetAtomWithIdx(0).GetSymbol() for mol in actions} == {"C", "O", "N"}
    assert all(mol.GetNumAtoms() == 1 for mol in actions)


def test_a_high_valence_atom_type_does_not_break_the_bond_orders():
    """Sulfur's maximum valence is 6, but the deepest bond is a triple.

    The bond-order loop used to be bounded by the largest valence in atom_types, so a
    methane state with S in the alphabet indexed past SINGLE/DOUBLE/TRIPLE and raised.
    Nothing caught it because the default alphabet is C, O, N, whose maximum is 4.
    """
    cfg = config.Config(atom_types=("C", "O", "N", "F", "Cl", "S", "Br"))
    assert_action_set_is_sane(environment.valid_actions(NAMED["ethanol"], cfg))
    assert_action_set_is_sane(environment.valid_actions(Chem.MolFromSmiles("C"), cfg))


def test_widening_atom_types_only_adds_candidates():
    """The C/O/N candidates have to survive verbatim, or the recorded runs stop meaning
    anything: a wider alphabet is a superset of the narrower one, not a different set."""
    narrow = config.Config(atom_types=("C", "O", "N"))
    wide = config.Config(atom_types=("C", "O", "N", "F", "Cl", "S", "Br"))
    keys = lambda cfg: {
        graph_key.canonical_hash(mol)
        for mol in environment.valid_actions(NAMED["aspirin"], cfg)
    }
    assert keys(narrow) < keys(wide)


@pytest.mark.parametrize(
    "smiles",
    [
        "C1CC#CCC1",  # sp carbon cannot have 180 degrees inside a six-ring
        "COOC",  # peroxide, and so trioxide
        "COOOC",
        "N#COc1ccccc1",  # cyanate ester
    ],
)
def test_is_plausible_rejects_motifs_absent_from_both_corpora(smiles):
    assert not environment.is_plausible(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize(
    "smiles",
    [
        "CNNc1ccccc1",  # hydrazine: 1.7% of ZINC and of the EGFR set, so not this filter's job
        "C#Cc1ccccc1",  # an alkyne is fine when it is not inside the ring
        "COc1ccccc1",
        "OC=Cc1ccccc1",  # enol, uncommon but measured
    ],
)
def test_is_plausible_keeps_real_chemistry(smiles):
    assert environment.is_plausible(Chem.MolFromSmiles(smiles))


def test_valid_actions_never_offers_an_implausible_candidate():
    """The filter runs inside _deduplicated, so it covers every action generator."""
    cfg = config.Config(atom_types=("C", "O", "N"))
    for candidate in environment.valid_actions(Chem.MolFromSmiles("COc1ccccc1"), cfg):
        assert environment.is_plausible(candidate)
