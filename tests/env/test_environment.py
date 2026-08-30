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
from mol_optim.chem import fragments, graph_key
from mol_optim.env import environment
from tests import molecules
from tests.molecules import NAMED, START_MOLECULES

LIBRARY = tuple(
    fragments.Fragment(mol=Chem.MolFromSmiles(s), smiles=s, count=1)
    for s in ("*C", "*OC", "*O", "*c1ccccc1", "*N(C)C")
)


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
    assert_action_set_is_sane(environment.valid_actions(start, LIBRARY))


@given(
    start_index=st.integers(min_value=0, max_value=len(START_MOLECULES) - 1),
    moves=st.lists(st.integers(min_value=0, max_value=10**6), min_size=1, max_size=3),
)
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_random_action_sequences_keep_every_invariant(start_index, moves):
    cfg = config.Config(
        init_mol=START_MOLECULES[start_index], max_steps_per_episode=len(moves)
    )
    episode = environment.reset(cfg, LIBRARY)
    for move in moves:
        assert_action_set_is_sane(episode.valid_actions)
        result = environment.step(
            episode, move % len(episode.valid_actions), molecules.size_reward, cfg, LIBRARY
        )
    assert_action_set_is_sane(episode.valid_actions)
    assert episode.num_steps_taken == len(moves)
    assert result.terminated


def test_attaching_a_substituent_is_reversible():
    """Every attach must have the original graph back in its own action set, or the
    agent is trapped by asymmetric edit logic.

    Attaches only. A swap is reversible just when the group it removed is also in the
    library, which is a property of the library, not of the action space.
    """
    start = NAMED["aspirin"]
    start_hash = graph_key.canonical_hash(graph_key.normalize(start))
    attached = 0
    for atom in start.GetAtoms():
        if atom.GetNumImplicitHs() < 1:
            continue
        for fragment in LIBRARY:
            grown = fragments.attach(start, atom.GetIdx(), fragment)
            if grown is None:
                continue
            attached += 1
            undo = {
                graph_key.canonical_hash(m)
                for m in environment.valid_actions(graph_key.normalize(grown), LIBRARY)
            }
            assert start_hash in undo, f"cannot undo {fragment.smiles}"
    assert attached, "the library never grew the molecule"


def test_a_bigger_library_only_adds_candidates():
    """A wider library is a superset of the narrower one, not a different set."""
    narrow = LIBRARY[:2]
    keys = lambda lib: {
        graph_key.canonical_hash(m)
        for m in environment.valid_actions(NAMED["aspirin"], lib)
    }
    assert keys(narrow) < keys(LIBRARY)


def test_step_rejects_a_step_past_termination():
    cfg = config.Config(init_mol=NAMED["ethanol"], max_steps_per_episode=1)
    episode = environment.reset(cfg, LIBRARY)
    assert environment.step(episode, 0, molecules.size_reward, cfg, LIBRARY).terminated
    with pytest.raises(ValueError, match="terminated"):
        environment.step(episode, 0, molecules.size_reward, cfg, LIBRARY)


def test_reset_refuses_an_empty_start():
    """There is nothing to hang a substituent off before a molecule exists."""
    with pytest.raises(ValueError, match="molecule to start from"):
        environment.reset(config.Config(init_mol=None), LIBRARY)


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
    """Attaching *O to an oxygen would make a peroxide; _deduplicated drops it."""
    for candidate in environment.valid_actions(NAMED["ethanol"], LIBRARY):
        assert environment.is_plausible(candidate)
