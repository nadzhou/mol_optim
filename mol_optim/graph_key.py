"""A canonical name for a molecular graph, computed from the graph.

The state of an episode is a molecular graph, and the loop needs to answer "have I
already seen this one?" thousands of times per step. That needs a canonical name.
Writing SMILES is the usual way to get one, and it is what this project used to do;
it costs a string write plus a re-parse per candidate, and the parse is where
aromaticity gets re-perceived and two names appear for one molecule.

This takes the name straight off the graph instead: RDKit's canonical atom ranking
(a graph invariant), the per-atom properties in that order, and the bond list rewritten
in terms of the ranks. 14 us against 19.5 us to write a canonical SMILES, and no parse
on the other side.

Checked against InChIKey over 7105 generated molecules: zero cases of two molecules
sharing a hash. The 26 cases where InChIKey merges what this splits are tautomer pairs
(N=N-OH against HN-N=O) — InChI normalizes mobile hydrogens by design. Those are
distinct graphs with distinct edits available, so splitting them is what an MDP over
graph edits wants.

The key is *constitutional*: L- and D-alanine get the same name. The atom-level action
space never sets a stereocentre, so nothing here can produce two stereoisomers to tell
apart, and ranking without chirality is what makes the key survive a trip through an SDF
— reading a molblock back re-perceives double-bond stereo from the coordinates, which
otherwise shifts the canonical ranking and renames the molecule.

Step 4 brought in real inhibitors, so there are now two keys. `canonical_hash` stays
constitutional and stays the RL loop's name for a state. `stereo_hash` separates
stereoisomers, which the BindingDB dataset needs: two enantiomers there carry two
different measured IC50 values.
"""

import hashlib

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def normalize(mol: Chem.Mol) -> Chem.Mol:
    """One aromaticity perception per graph, whatever route built it.

    A candidate built by editing a kekulized copy can carry different aromatic flags
    from the same molecule built another way. That is two hashes for one graph — and,
    worse, two different Morgan fingerprints, so the network sees one state as two.
    Kekulizing and re-sanitizing puts every molecule through one perception path.
    """
    work = Chem.RWMol(mol)
    Chem.Kekulize(work, clearAromaticFlags=True)
    Chem.SanitizeMol(work)
    return work.GetMol()


def canonical_hash(mol: Chem.Mol) -> str:
    """A 32-character name for this graph. Same graph, same name; different, different."""
    return _hash(mol, chirality=False)


def stereo_hash(mol: Chem.Mol) -> str:
    """The same name with configuration in it: L- and D-alanine differ here.

    What the BindingDB dataset is deduplicated and split on.
    """
    return _hash(mol, chirality=True)


def scaffold_hash(mol: Chem.Mol) -> str:
    """The name of this molecule's Bemis-Murcko scaffold — ring systems and linkers.

    Constitutional on purpose: two stereoisomers of one frame are one series. Molecules
    with no rings share the empty scaffold's name.
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
