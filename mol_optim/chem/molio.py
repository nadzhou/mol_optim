from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


def read(path: Path) -> tuple[Chem.Mol, ...]:
    molecules = []
    for mol in Chem.SDMolSupplier(str(path)):
        if mol is None:
            raise ValueError(f"unreadable record in {path}")
        molecules.append(mol)
    return tuple(molecules)


def read_named(path: Path) -> dict[str, Chem.Mol]:
    return {mol.GetProp("_Name"): mol for mol in read(path)}


def write(path: Path, molecules: tuple[Chem.Mol, ...], properties: dict[str, list]) -> None:
    writer = Chem.SDWriter(str(path))
    for index, mol in enumerate(molecules):
        record = Chem.Mol(mol)
        # Generated molecules carry no coordinates; without these the SDF opens as a
        # pile of atoms stacked on the origin. A molecule that already has geometry —
        # a crystal ligand, a docked pose — keeps it: computing 2D coordinates over
        # measured ones silently replaces the only thing that made it worth writing.
        if record.GetNumConformers() == 0:
            AllChem.Compute2DCoords(record)
        for name, values in properties.items():
            record.SetProp(name, str(values[index]))
        writer.write(record)
    writer.close()
