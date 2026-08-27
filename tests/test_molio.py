"""SDF in and out — the only place a graph turns into bytes."""

from rdkit import Chem
from rdkit.Chem import AllChem

from mol_optim import molio
from tests.molecules import NAMED


def test_measured_geometry_survives_the_round_trip(tmp_path):
    # A crystal ligand or a docked pose is worth writing only for its coordinates, and
    # computing 2D ones over them replaces exactly that.
    # Heavy atoms only: an SDF read back drops the explicit hydrogens, so comparing a
    # molecule that carries them compares two different atom counts.
    mol = Chem.Mol(NAMED["aspirin"])
    AllChem.EmbedMolecule(mol, randomSeed=0)
    before = mol.GetConformer().GetPositions()

    path = tmp_path / "posed.sdf"
    molio.write(path, (mol,), {})
    after = molio.read(path)[0].GetConformer().GetPositions()

    assert abs(before - after).max() < 1e-3


def test_properties_are_written_alongside_the_molecules(tmp_path):
    path = tmp_path / "props.sdf"
    molio.write(path, (NAMED["benzene"], NAMED["toluene"]), {"rank": [1, 2]})
    assert [mol.GetProp("rank") for mol in molio.read(path)] == ["1", "2"]
