"""A canonical name for a molecular graph, computed from the graph.

RDKit's canonical atom ranking, the per-atom properties in that order, and the bond list
rewritten in terms of the ranks. 14 us against 19.5 us to write a canonical SMILES, and no
re-parse on the other side — which is where aromaticity gets re-perceived and one molecule
picks up two names.

Two keys, because real inhibitors brought stereochemistry in. `canonical_hash` is
constitutional and is the RL loop's name for a state; `stereo_hash` separates
stereoisomers, which the BindingDB dataset needs, since two enantiomers there carry two
different measured IC50 values.
"""

import hashlib

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def normalize(mol: Chem.Mol) -> Chem.Mol:
    """One aromaticity perception per graph, whatever route built it.

    A candidate built by editing a kekulized copy otherwise carries different aromatic
    flags from the same molecule built another way: two hashes and two Morgan
    fingerprints for one graph, so the network sees one state as two.
    """
    work = Chem.RWMol(mol)
    Chem.Kekulize(work, clearAromaticFlags=True)
    Chem.SanitizeMol(work)
    return work.GetMol()


def canonical_hash(mol: Chem.Mol) -> str:
    # Constitutional: L- and D-alanine get one name. The atom-level action space never
    # sets a stereocentre, and ranking without chirality is what makes the key survive a
    # trip through an SDF, where a molblock re-perceives double-bond stereo from geometry.
    return _hash(mol, chirality=False)


def stereo_hash(mol: Chem.Mol) -> str:
    """The same name with configuration in it. What BindingDB is deduplicated on."""
    return _hash(mol, chirality=True)


def scaffold_hash(mol: Chem.Mol) -> str:
    """The name of this molecule's Bemis-Murcko scaffold — ring systems and linkers.

    Two stereoisomers of one frame are one series. Molecules with no rings share the
    empty scaffold's name.
    """
    return _hash(MurckoScaffold.GetScaffoldForMol(normalize(mol)), chirality=False)


def _hash(mol: Chem.Mol, chirality: bool) -> str:
    mol = normalize(mol)
    if chirality:
        # A double bond with undefined geometry is STEREONONE from a SMILES and STEREOANY
        # from a molblock. Both mean unspecified; left alone this renamed 185 of the
        # 10,862 EGFR compounds on the way to disk.
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        for bond in mol.GetBonds():
            if bond.GetStereo() == Chem.BondStereo.STEREOANY:
                bond.SetStereo(Chem.BondStereo.STEREONONE)
    ranks = list(
        Chem.CanonicalRankAtoms(mol, includeChirality=chirality)
    )  # canonical position of each atom index

    atoms: list[tuple | None] = [None] * mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        atoms[ranks[atom.GetIdx()]] = (
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            int(atom.GetIsAromatic()),
        ) + ((atom.GetPropsAsDict().get("_CIPCode", ""),) if chirality else ())
    bonds = sorted(
        (
            min(ranks[bond.GetBeginAtomIdx()], ranks[bond.GetEndAtomIdx()]),
            max(ranks[bond.GetBeginAtomIdx()], ranks[bond.GetEndAtomIdx()]),
            int(bond.GetBondType()),
        )
        + ((int(bond.GetStereo()),) if chirality else ())
        for bond in mol.GetBonds()
    )
    return hashlib.blake2b(repr((atoms, bonds)).encode(), digest_size=16).hexdigest()
