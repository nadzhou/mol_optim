"""The state key. Everything downstream trusts that one graph has one name."""

import pytest
from rdkit import Chem

from mol_optim import graph_key
from tests.molecules import NAMED, START_MOLECULES


@pytest.mark.parametrize("mol", START_MOLECULES, ids=lambda m: m.GetProp("_Name"))
def test_hash_is_invariant_to_atom_ordering(mol):
    # The hash is built from RDKit's canonical ranking, so renumbering the atoms must
    # not touch it. If it does, the key is really an atom-order key and dedup is a lie.
    reversed_order = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    assert graph_key.canonical_hash(reversed_order) == graph_key.canonical_hash(mol)


@pytest.mark.parametrize("mol", START_MOLECULES, ids=lambda m: m.GetProp("_Name"))
def test_hash_survives_a_molblock_round_trip(mol):
    # A molblock has no aromatic flags, so reading one back re-perceives aromaticity
    # from scratch. Report output is SDF, so a molecule that comes back under a
    # different name would break every lookup against a saved run.
    rebuilt = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol))
    assert graph_key.canonical_hash(rebuilt) == graph_key.canonical_hash(mol)


@pytest.mark.parametrize("mol", START_MOLECULES, ids=lambda m: m.GetProp("_Name"))
def test_hash_is_the_same_for_a_graph_rebuilt_atom_by_atom(mol):
    # The property the environment actually depends on: two edit paths that arrive at
    # the same graph must arrive at the same name. Rebuilding in reverse atom order,
    # with no coordinates and no aromatic flags, is as different a path as there is.
    builder = Chem.RWMol()
    new_index = {}
    for old_index in reversed(range(mol.GetNumAtoms())):
        atom = mol.GetAtomWithIdx(old_index)
        fresh = Chem.Atom(atom.GetAtomicNum())
        fresh.SetFormalCharge(atom.GetFormalCharge())
        new_index[old_index] = builder.AddAtom(fresh)
    for bond in mol.GetBonds():
        builder.AddBond(
            new_index[bond.GetBeginAtomIdx()],
            new_index[bond.GetEndAtomIdx()],
            bond.GetBondType(),
        )
    rebuilt = builder.GetMol()
    Chem.SanitizeMol(rebuilt)
    assert graph_key.canonical_hash(rebuilt) == graph_key.canonical_hash(mol)


def test_the_key_is_constitutional_so_stereoisomers_share_it():
    # Pinning a known limitation, not endorsing it. The atom-level action space cannot
    # create a stereocentre, and ranking without chirality is what keeps the key stable
    # across an SDF round trip. Step 4 brings in real inhibitors and this must change
    # then — when it does, this test fails and asks to be rewritten.
    assert graph_key.canonical_hash(NAMED["alanine_L"]) == graph_key.canonical_hash(
        NAMED["alanine_D"]
    )


def test_distinct_molecules_get_distinct_hashes():
    # The alanines are one constitution under two configurations; they share a key on
    # purpose, and test_the_key_is_constitutional_so_stereoisomers_share_it covers them.
    constitutions = [
        mol for name, mol in NAMED.items() if not name.startswith("alanine")
    ]
    hashes = {graph_key.canonical_hash(mol) for mol in constitutions}
    assert len(hashes) == len(constitutions)


def test_hash_separates_molecules_that_differ_only_in_bond_order():
    # Ethane against ethene: same atoms, same connectivity, different bond.
    ethane = NAMED["ethane"]
    ethene = Chem.RWMol(ethane)
    ethene.GetBondBetweenAtoms(0, 1).SetBondType(Chem.BondType.DOUBLE)
    Chem.SanitizeMol(ethene)
    assert graph_key.canonical_hash(ethene) != graph_key.canonical_hash(ethane)


def test_normalize_is_idempotent():
    for mol in START_MOLECULES:
        once = graph_key.normalize(mol)
        assert graph_key.canonical_hash(graph_key.normalize(once)) == (
            graph_key.canonical_hash(once)
        )
