"""Molecules to and from files. The only place a graph turns into bytes.

SDF, not SMILES: an SDF record is an atom table plus a bond table, which is the graph
written down, not a linear re-encoding of it that has to be parsed back.
"""

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


def read_named(path: Path) -> dict[str, Chem.Mol]:
    """Every molecule in an SDF, keyed by its record name."""
    molecules = {}
    for mol in Chem.SDMolSupplier(str(path)):
        if mol is None:
            raise ValueError(f"unreadable record in {path}")
        molecules[mol.GetProp("_Name")] = mol
    return molecules


def write(path: Path, molecules: tuple[Chem.Mol, ...], properties: dict[str, list]) -> None:
    """Writes molecules to an SDF, with one column of `properties` per molecule."""
    writer = Chem.SDWriter(str(path))
    for index, mol in enumerate(molecules):
        record = Chem.Mol(mol)
        # Generated molecules carry no coordinates; without these the SDF opens as a
        # pile of atoms stacked on the origin.
        AllChem.Compute2DCoords(record)
        for name, values in properties.items():
            record.SetProp(name, str(values[index]))
        writer.write(record)
    writer.close()
