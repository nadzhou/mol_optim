"""Docking against the prepared 1M17 receptor. plan.md, the Step 8 spot-check.

The gating test is the redock. Every way of getting the receptor typing, the box or the
ligand preparation wrong still returns a plausible negative number, so a score proves
nothing on its own — only putting a known ligand back where the crystal found it does.
"""

from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolAlign

from mol_optim import docking, molio

RECEPTOR = Path("data/structures/1M17_receptor.pdbqt")
LIGAND = Path("data/structures/1M17_ligand.sdf")


@pytest.fixture(scope="module")
def site() -> docking.Site:
    return docking.site_from(RECEPTOR, LIGAND)


def test_the_box_is_centred_on_the_co_crystal_ligand(site):
    centroid = molio.read(LIGAND)[0].GetConformer().GetPositions().mean(axis=0)
    assert site.center == pytest.approx(tuple(centroid))


def test_the_box_is_bigger_than_the_ligand_it_contains(site):
    positions = molio.read(LIGAND)[0].GetConformer().GetPositions()
    extent = positions.max(axis=0) - positions.min(axis=0)
    assert min(site.size) > max(extent)


def test_a_missing_receptor_says_how_to_build_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="mol_optim.fetch_structure"):
        docking.site_from(tmp_path / "absent.pdbqt", LIGAND)


def test_ligand_efficiency_divides_by_heavy_atoms():
    ligand = molio.read(LIGAND)[0]
    pose = docking.Pose(mol=ligand, score=-8.7)
    assert docking.ligand_efficiency(pose) == pytest.approx(-8.7 / 29)


@pytest.mark.slow
def test_redocking_reproduces_the_crystal_pose(site):
    # 6 s. The one test here that says the setup is right rather than merely running.
    pytest.importorskip("vina")
    crystal = molio.read(LIGAND)[0]
    # Geometry thrown away: dock() embeds from the graph, so the crystal coordinates
    # cannot leak into the answer.
    from_scratch = Chem.Mol(crystal)
    from_scratch.RemoveAllConformers()

    pose = docking.dock(docking.engine(site), from_scratch)
    assert pose is not None
    assert rdMolAlign.GetBestRMS(Chem.RemoveHs(pose.mol), crystal) < 2.0
    # Erlotinib against its own co-crystal receptor. A number far from this means the
    # receptor or the box moved.
    assert pose.score == pytest.approx(-7.3, abs=1.0)


@pytest.mark.slow
def test_an_element_autodock_cannot_type_returns_none(site):
    pytest.importorskip("vina")
    # Phenylboronic acid. Boron is real medicinal chemistry — bortezomib is a boronic
    # acid — and AutoDock has no type for it. Vina raises a TypeError from deep inside
    # its PDBQT parser, and one of these in a batch would otherwise end the whole run.
    # The caller must see None, not a score of zero, which would rank above every real
    # binder.
    boronic = Chem.MolFromSmiles("OB(O)c1ccccc1")
    assert docking.dock(docking.engine(site), boronic) is None
