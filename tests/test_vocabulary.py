"""The fragment vocabulary. plan.md "Action space — fragment edits over a precedented
vocabulary".

The vocabulary is the action space, so a bug here is not a worse reward — it is a
different project. These check the two things the rest of the loop assumes: a fragment
can actually be attached, and the vocabulary contains only chemistry that was cut out of
a measured inhibitor.
"""

from pathlib import Path

import pytest
from rdkit import Chem

from mol_optim import graph_key, vocabulary

VOCABULARY = Path("data/egfr_fragments.sdf")


def test_every_fragment_sanitizes_on_its_own(fragments):
    # The dummy is gone and the attachment atom got its hydrogen back, so each fragment
    # is a real molecule. If it is not, every candidate built from it is invalid too.
    for fragment in fragments:
        Chem.SanitizeMol(Chem.Mol(fragment.mol))


def test_every_attachment_atom_can_take_a_bond(fragments):
    for fragment in fragments:
        atom = fragment.mol.GetAtomWithIdx(fragment.attachment_idx)
        assert atom.GetNumImplicitHs() > 0, Chem.MolToSmiles(fragment.mol)


def test_the_attachment_index_points_into_the_fragment(fragments):
    for fragment in fragments:
        assert 0 <= fragment.attachment_idx < fragment.mol.GetNumAtoms()


def test_no_fragment_carries_a_nitrogen_nitrogen_bond(fragments):
    # The Step 5 finding, asserted. The agent put an N-N bond in 100% of its episodes
    # against a reward with no term for it; this is why it cannot do that here.
    for fragment in fragments:
        catenated = [
            bond
            for bond in fragment.mol.GetBonds()
            if bond.GetBeginAtom().GetAtomicNum() == 7
            and bond.GetEndAtom().GetAtomicNum() == 7
            and not bond.GetIsAromatic()
        ]
        assert not catenated, Chem.MolToSmiles(fragment.mol)


def test_one_molecule_at_two_positions_is_two_fragments(fragments):
    # Fragments are hashed with the BRICS dummy still attached, so ortho- and
    # para-fluorophenyl are two entries. Collapsing them would silently delete half the
    # regiochemistry the vocabulary exists to carry.
    by_graph: dict[str, set[int]] = {}
    for fragment in fragments:
        key = graph_key.canonical_hash(fragment.mol)
        by_graph.setdefault(key, set()).add(
            list(Chem.CanonicalRankAtoms(fragment.mol))[fragment.attachment_idx]
        )
    assert any(len(positions) > 1 for positions in by_graph.values())


def test_the_vocabulary_round_trips_through_an_sdf(fragments, tmp_path):
    # The attachment index is an index, so anything that renumbers atoms on the way to
    # disk points it at the wrong atom. An SDF is an atom table and does not.
    path = tmp_path / "fragments.sdf"
    vocabulary.write(path, fragments)
    for original, reloaded in zip(fragments, vocabulary.load(path)):
        assert graph_key.canonical_hash(original.mol) == graph_key.canonical_hash(
            reloaded.mol
        )
        assert original.attachment_idx == reloaded.attachment_idx
        assert original.count == reloaded.count


def test_load_says_how_to_build_a_vocabulary_that_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="mol_optim.vocabulary"):
        vocabulary.load(tmp_path / "absent.sdf")


def test_fragments_are_ordered_by_how_often_they_were_cut(fragments):
    counts = [fragment.count for fragment in fragments]
    assert counts == sorted(counts, reverse=True)


@pytest.mark.slow
def test_building_twice_gives_the_same_vocabulary(compounds):
    # 15 s a build. Ties are broken by graph key, so a run must not depend on the order
    # Counter happens to hand back.
    first = vocabulary.build(compounds)
    second = vocabulary.build(compounds)
    assert [graph_key.canonical_hash(f.mol) for f in first] == [
        graph_key.canonical_hash(f.mol) for f in second
    ]
    assert [f.attachment_idx for f in first] == [f.attachment_idx for f in second]


@pytest.mark.slow
def test_the_committed_vocabulary_is_what_build_produces(compounds, fragments):
    # Catches a vocabulary edited by hand, or one built from an older dataset.
    rebuilt = vocabulary.build(compounds, size=len(fragments))
    assert [graph_key.canonical_hash(f.mol) for f in rebuilt] == [
        graph_key.canonical_hash(f.mol) for f in fragments
    ]
