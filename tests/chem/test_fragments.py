"""The substituent action space. See mol_optim/chem/fragments.py."""

from rdkit import Chem

from mol_optim.chem import fragments, graph_key
from mol_optim.env import environment


def _fragment(smiles: str) -> fragments.Fragment:
    return fragments.Fragment(mol=Chem.MolFromSmiles(smiles), smiles=smiles, count=1)


def test_library_counts_compounds_not_occurrences():
    """p-xylene wears two methyls and must still count once toward methyl."""
    molecules = [Chem.MolFromSmiles(s) for s in ("Cc1ccc(C)cc1", "Cc1ccccc1")]
    library = fragments.library(molecules, min_count=1)

    methyl = next(f for f in library if f.smiles == "*C")
    assert methyl.count == 2


def test_library_drops_fragments_below_min_count():
    molecules = [Chem.MolFromSmiles(s) for s in ("Cc1ccccc1", "CCc1ccccc1")]
    library = fragments.library(molecules, min_count=2)

    assert [f.smiles for f in library] == ["*C"]


def test_library_is_ordered_by_count():
    molecules = [Chem.MolFromSmiles(s) for s in ("Cc1ccccc1", "Cc1ccccc1", "CCc1ccccc1")]
    counts = [f.count for f in fragments.library(molecules, min_count=1)]

    assert counts == sorted(counts, reverse=True)


def test_library_excludes_ring_bonds():
    """Cutting a ring bond does not separate the molecule, so benzene has no substituent."""
    assert fragments.library([Chem.MolFromSmiles("c1ccccc1")], min_count=1) == ()


def test_attach_bonds_the_fragment_at_the_dummy():
    benzene = Chem.MolFromSmiles("c1ccccc1")
    grown = fragments.attach(benzene, 0, _fragment("*OC"))

    assert graph_key.canonical_hash(grown) == graph_key.canonical_hash(
        Chem.MolFromSmiles("COc1ccccc1")
    )


def test_attach_returns_none_when_the_atom_has_no_room():
    """The carbonyl carbon of acetone has no free valence."""
    acetone = Chem.MolFromSmiles("CC(=O)C")
    assert fragments.attach(acetone, 1, _fragment("*C")) is None


def test_detach_keeps_the_larger_side():
    anisole = Chem.MolFromSmiles("COc1ccccc1")
    methoxy_bond = next(
        bond
        for bond in anisole.GetBonds()
        if not bond.IsInRing() and bond.GetBeginAtom().GetSymbol() == "C"
    )
    trimmed = fragments.detach(anisole, methoxy_bond)

    assert trimmed.GetNumHeavyAtoms() < anisole.GetNumHeavyAtoms()


def test_substitutions_reach_a_target_in_one_action():
    """The point of the action space: one substituent swap is one action, not six edits."""
    toluene = Chem.MolFromSmiles("Cc1ccccc1")
    library = (_fragment("*OC"),)
    reached = {
        graph_key.canonical_hash(graph_key.normalize(m))
        for m in fragments.substitutions(toluene, library)
    }

    # Swap the methyl for a methoxy, in a single action.
    assert graph_key.canonical_hash(Chem.MolFromSmiles("COc1ccccc1")) in reached


def test_substitutions_include_plain_removal():
    toluene = Chem.MolFromSmiles("Cc1ccccc1")
    reached = {
        graph_key.canonical_hash(graph_key.normalize(m))
        for m in fragments.substitutions(toluene, ())
    }

    assert graph_key.canonical_hash(Chem.MolFromSmiles("c1ccccc1")) in reached


def test_fragment_action_space_is_deduplicated_and_ordered():
    """environment._deduplicated is what both action spaces share; order is the hash."""
    library = (_fragment("*C"), _fragment("*OC"))
    first = environment.valid_actions(Chem.MolFromSmiles("c1ccccc1"), library)
    again = environment.valid_actions(Chem.MolFromSmiles("c1ccccc1"), library)

    hashes = [graph_key.canonical_hash(m) for m in first]
    assert hashes == sorted(hashes)
    assert len(hashes) == len(set(hashes))
    assert hashes == [graph_key.canonical_hash(m) for m in again]
