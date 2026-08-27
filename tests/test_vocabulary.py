"""The fragment vocabulary: the action space, cut from the target's own actives.

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


def test_no_fragment_carries_a_nitrogen_nitrogen_bond(fragments):
    # The pIC50 run's finding, asserted. The agent put an N-N bond in 100% of its
    # episodes
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


@pytest.mark.slow
def test_the_committed_vocabulary_is_what_build_produces(compounds, fragments):
    # Catches a vocabulary edited by hand, or one built from an older dataset.
    rebuilt = vocabulary.build(compounds, size=len(fragments))
    assert [graph_key.canonical_hash(f.mol) for f in rebuilt] == [
        graph_key.canonical_hash(f.mol) for f in fragments
    ]
