"""Step 1 environment invariants.

The action space is the foundation: if it emits an invalid graph, everything downstream
is garbage, and nothing crashes when it does. Random *action sequences* are where the
bugs are, so the fuzz test walks whole episodes.
"""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from rdkit import Chem

from mol_optim import config, environment, graph_key, rewards
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
            episode, move % len(episode.valid_actions), rewards.qed, cfg
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


def test_no_modification_present_only_when_allowed():
    start = NAMED["ethanol"]
    start_hash = graph_key.canonical_hash(start)
    assert start_hash in action_hashes(start, config.Config(allow_no_modification=True))
    assert start_hash not in action_hashes(
        start, config.Config(allow_no_modification=False)
    )


def test_removal_actions_present_only_when_allowed():
    start, ethane = NAMED["ethanol"], NAMED["ethane"]
    with_removal = action_hashes(start, config.Config(allow_removal=True))
    without_removal = action_hashes(start, config.Config(allow_removal=False))
    assert without_removal < with_removal
    ethane_hash = graph_key.canonical_hash(ethane)
    assert ethane_hash in with_removal and ethane_hash not in without_removal


def test_allowed_ring_sizes_is_threaded_through():
    # Hexane: bonding atom 0 to atom 5 closes a six-ring, atom 0 to atom 2 a three-ring.
    hexane = NAMED["hexane"]
    six_only = action_hashes(hexane, config.Config(allowed_ring_sizes=(6,)))
    three_only = action_hashes(hexane, config.Config(allowed_ring_sizes=(3,)))
    cyclohexane = graph_key.canonical_hash(NAMED["cyclohexane"])
    propylcyclopropane = graph_key.canonical_hash(NAMED["propylcyclopropane"])
    assert cyclohexane in six_only and cyclohexane not in three_only
    assert propylcyclopropane in three_only and propylcyclopropane not in six_only


def test_bonds_between_rings_is_threaded_through():
    # Two benzenes joined by a single bond; a new bond would fuse them.
    biphenyl = NAMED["biphenyl"]
    forbidden = action_hashes(biphenyl, config.Config(allow_bonds_between_rings=False))
    allowed = action_hashes(biphenyl, config.Config(allow_bonds_between_rings=True))
    assert forbidden < allowed


def test_step_rejects_a_candidate_that_does_not_exist():
    cfg = config.Config(init_mol=NAMED["ethanol"], max_steps_per_episode=2)
    episode = environment.reset(cfg)
    with pytest.raises(ValueError, match="No candidate"):
        environment.step(episode, len(episode.valid_actions), rewards.qed, cfg)


def test_step_rejects_a_step_past_termination():
    cfg = config.Config(init_mol=NAMED["ethanol"], max_steps_per_episode=1)
    episode = environment.reset(cfg)
    assert environment.step(episode, 0, rewards.qed, cfg).terminated
    with pytest.raises(ValueError, match="terminated"):
        environment.step(episode, 0, rewards.qed, cfg)


def test_empty_start_offers_one_atom_of_each_type():
    cfg = config.Config(init_mol=None, atom_types=("C", "O", "N"))
    actions = environment.valid_actions(None, cfg)
    assert len(actions) == 3
    assert {mol.GetAtomWithIdx(0).GetSymbol() for mol in actions} == {"C", "O", "N"}
    assert all(mol.GetNumAtoms() == 1 for mol in actions)
