"""Reward functions. Pure: a graph in, a float out. The discount lives in environment.step."""

from rdkit import Chem
from rdkit.Chem import QED


def qed(mol: Chem.Mol | None) -> float:
    """Quantitative Estimate of Drug-likeness, 0..1. The Step 1 target."""
    if mol is None or mol.GetNumAtoms() == 0:
        return 0.0
    return QED.qed(mol)
