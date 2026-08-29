"""BindingDB IC50 values as pIC50 on graphs. `fetch_bindingdb.py` writes what this reads.

The unit conversion lives here rather than inline because nM against uM shifts every
label by a constant, and that trains a regressor which looks fine on its own test set
and ranks nothing correctly.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

from mol_optim import graph_key, molio


@dataclass(frozen=True)
class Compound:
    mol: Chem.Mol
    pic50: float
    num_measurements: int  # how many BindingDB rows the median was taken over
    pic50_spread: float  # max - min across those rows; 0.0 for a single measurement
    # Carried, not recomputed: the split and the leakage tests ask for it repeatedly.
    scaffold: str


def to_pic50(ic50_nm: float) -> float:
    """IC50 in nanomolar to pIC50 = -log10(IC50 in molar). 1 nM is 9.0, 1 uM is 6.0."""
    if ic50_nm <= 0.0:
        raise ValueError(f"IC50 must be positive, got {ic50_nm} nM")
    return 9.0 - math.log10(ic50_nm)


def median(values: list[float]) -> float:
    # Median, not mean: duplicates are the same compound measured in different labs, and
    # the disagreements reach 8 logs. One bad row should move the label by nothing.
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def load(path: Path) -> tuple[Compound, ...]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run the 'bindingdb' step first.")
    named = molio.read_named(path)
    return tuple(
        Compound(
            mol=mol,
            pic50=float(mol.GetProp("pic50")),
            num_measurements=int(mol.GetProp("num_measurements")),
            pic50_spread=float(mol.GetProp("pic50_spread")),
            scaffold=graph_key.scaffold_hash(mol),
        )
        for mol in named.values()
    )
