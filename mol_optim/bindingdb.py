"""BindingDB IC50 values as pIC50 on graphs. plan.md Step 4.

`fetch_bindingdb.py` writes the dataset this reads. The unit conversion lives here
rather than inline because nM against uM shifts every label by a constant, and that
trains a regressor which looks fine on its own test set and ranks nothing correctly.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

from mol_optim import graph_key, molio

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "egfr_ic50.sdf"


@dataclass(frozen=True)
class Compound:
    """One compound with one label, after aggregation."""

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
    """The middle value, or the mean of the middle two.

    Median, not mean: duplicates are the same compound in different labs, and the
    disagreements reach 8 logs. One bad row should move the label by nothing.
    """
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def load(path: Path = DATASET_PATH) -> tuple[Compound, ...]:
    """The dataset, as compounds. Raises if it has not been built yet."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Build it once with: python -m mol_optim.fetch_bindingdb"
        )
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
