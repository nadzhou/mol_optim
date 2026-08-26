"""AutoDock Vina against a prepared receptor. plan.md, "The failure mode to design around".

Built to be the one signal in this project that does not come from the pIC50 regressor.
It is not that signal: on 30 BindingDB compounds spanning seven logs, the Vina score
ranks measured pIC50 at Spearman +0.31 — the wrong sign, since Vina is negative-is-better
— and real EGFR actives land 0.04 kcal/mol from random ZINC molecules. The setup itself
is right: `test_redocking_reproduces_the_crystal_pose` puts erlotinib back within 1.3 A.
So this is kept, wired, and tested, and its output is not treated as evidence about
whether a molecule binds. See plan.md, "The docking spot-check ... does not rank EGFR
compounds".

Ligands go in as `Chem.Mol` and come back as `Chem.Mol`. Meeko does the PDBQT conversion
in memory, so nothing here writes a molecule to disk or to a SMILES string.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from mol_optim import molio


@dataclass(frozen=True)
class Site:
    """A receptor and the box to search inside it."""

    receptor_pdbqt: Path
    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True)
class Pose:
    """One docked ligand: the graph, its docked geometry, and what Vina scored it."""

    mol: Chem.Mol  # carries a single conformer, the best-scored pose
    score: float  # kcal/mol, more negative is a better predicted binder


def site_from(receptor_pdbqt: Path, ligand_sdf: Path, box: float = 22.0) -> Site:
    """The box centred on a co-crystal ligand, which is where the site is by definition.

    22 A a side against the erlotinib extent of 15.7 A leaves about 3 A either way — room
    for a larger analog to move without opening up a second pocket for it to wander into.
    """
    if not receptor_pdbqt.exists() or not ligand_sdf.exists():
        raise FileNotFoundError(
            f"{receptor_pdbqt} or {ligand_sdf} is missing. Build them with: "
            "python -m mol_optim.fetch_structure"
        )
    ligand = molio.read(ligand_sdf)[0]
    centroid = ligand.GetConformer().GetPositions().mean(axis=0)  # [3]
    return Site(
        receptor_pdbqt=receptor_pdbqt,
        center=tuple(float(x) for x in centroid),
        size=(box, box, box),
    )


def engine(site: Site, seed: int = 0):
    """A Vina with the receptor read and its grid maps computed.

    Held open across ligands: the maps take 0.2 s and do not depend on the ligand, and
    docking one molecule is 6 s. Imported here rather than at module scope so the rest of
    the package still imports on a machine without Vina built.
    """
    from vina import Vina

    vina = Vina(sf_name="vina", seed=seed, verbosity=0)
    vina.set_receptor(str(site.receptor_pdbqt))
    vina.compute_vina_maps(center=list(site.center), box_size=list(site.size))
    return vina


def dock(
    vina, mol: Chem.Mol, exhaustiveness: int = 16, num_poses: int = 5
) -> Pose | None:
    """The best pose Vina finds, or None if the molecule could not be prepared.

    None is a real outcome, not an error: a generated molecule can fail to embed in 3D or
    fail Meeko's typing, and a caller that treats that as a score of zero would rank it
    above every real binder.
    """
    from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate

    work = Chem.AddHs(Chem.Mol(mol))
    if AllChem.EmbedMolecule(work, randomSeed=0) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(work, maxIters=1000)

    # An element outside AutoDock's table does not come back as a flag — Meeko raises
    # KeyError on silicon, and Vina raises TypeError parsing a boron it cannot type.
    # Both are ordinary medicinal chemistry and both mean the same thing here: no score.
    try:
        prepared = MoleculePreparation().prepare(work)
        if not prepared:
            return None
        pdbqt, ok, _ = PDBQTWriterLegacy.write_string(prepared[0])
        if not ok:
            return None
        vina.set_ligand_from_string(pdbqt)
    except (KeyError, TypeError, ValueError, RuntimeError):
        return None
    vina.dock(exhaustiveness=exhaustiveness, n_poses=num_poses)
    posed = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule(vina.poses(n_poses=num_poses)))[0]

    # Poses come back as conformers on one molecule, best first. Keep the first.
    best = Chem.Mol(posed)
    best.RemoveAllConformers()
    best.AddConformer(posed.GetConformer(0), assignId=True)
    return Pose(mol=best, score=float(vina.energies(n_poses=1)[0, 0]))


def ligand_efficiency(pose: Pose) -> float:
    """Score per heavy atom. The standard correction for Vina scoring large molecules well.

    Measured on this target it does not help — Spearman 0.12 against pIC50, where the raw
    score gives 0.31. Kept because reporting a raw Vina score without it invites the
    reader to make the size mistake themselves.
    """
    return pose.score / Chem.RemoveHs(pose.mol).GetNumHeavyAtoms()


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("sdf", type=Path, help="molecules to dock")
    parser.add_argument("--receptor", type=Path, default=Path("data/structures/1M17_receptor.pdbqt"))
    parser.add_argument("--ligand", type=Path, default=Path("data/structures/1M17_ligand.sdf"))
    parser.add_argument("--out", type=Path, default=None, help="write posed molecules here")
    parser.add_argument("--exhaustiveness", type=int, default=16)
    parser.add_argument(
        "--controls",
        action="store_true",
        help="also dock known actives, weak binders and ZINC — the comparison that "
        "showed this score does not rank EGFR compounds (plan.md)",
    )
    args = parser.parse_args()

    site = site_from(args.receptor, args.ligand)
    print(f"box {site.size[0]:.0f} A centred at "
          f"({site.center[0]:.2f}, {site.center[1]:.2f}, {site.center[2]:.2f})")
    vina = engine(site)

    molecules = molio.read(args.sdf)
    started = time.perf_counter()
    poses = [dock(vina, mol, args.exhaustiveness) for mol in molecules]
    scored = [pose for pose in poses if pose is not None]

    print(f"{'#':>3} {'score':>8} {'LE':>7} {'atoms':>6}")
    for index, pose in enumerate(poses):
        if pose is None:
            print(f"{index:>3}    failed to prepare")
            continue
        print(f"{index:>3} {pose.score:>8.2f} {ligand_efficiency(pose):>7.3f} "
              f"{Chem.RemoveHs(pose.mol).GetNumHeavyAtoms():>6}")
    values = np.array([pose.score for pose in scored])
    print(f"\n{len(scored)}/{len(molecules)} docked in {time.perf_counter() - started:.0f}s   "
          f"mean {values.mean():.2f}  best {values.min():.2f} kcal/mol")
    if args.out is not None:
        molio.write(
            args.out,
            tuple(pose.mol for pose in scored),
            {"vina_score": [f"{pose.score:.2f}" for pose in scored]},
        )
        print(f"wrote {args.out}")

    if args.controls:
        # A Vina score in isolation says nothing, so the number above is only readable
        # next to compounds whose affinity is known. This is what plan.md reports.
        from mol_optim import bindingdb, regressor, zinc

        compounds = bindingdb.load()
        groups = {
            args.sdf.stem: [(mol, None) for mol in molecules],
            "EGFR pIC50 >= 9": [
                (c.mol, c.pic50) for c in compounds if c.pic50 >= 9.0
            ][:15],
            "EGFR pIC50 <= 5.5": [
                (c.mol, c.pic50) for c in compounds if c.pic50 <= 5.5
            ][:15],
            "ZINC": [(mol, None) for mol in zinc.molecules(limit=15)],
        }

        measured_scores, measured_pic50 = [], []
        print(f"\n{'group':>20} {'n':>4} {'mean':>8} {'best':>8} {'LE':>8} {'atoms':>7}")
        for name, members in groups.items():
            rows = [(dock(vina, mol, args.exhaustiveness), pic50) for mol, pic50 in members]
            rows = [(pose, pic50) for pose, pic50 in rows if pose is not None]
            values = np.array([pose.score for pose, _ in rows])
            efficiency = np.array([ligand_efficiency(pose) for pose, _ in rows])
            sizes = np.array(
                [Chem.RemoveHs(pose.mol).GetNumHeavyAtoms() for pose, _ in rows]
            )
            print(f"{name:>20} {len(rows):>4} {values.mean():>8.2f} {values.min():>8.2f} "
                  f"{efficiency.mean():>8.3f} {sizes.mean():>7.1f}")
            for pose, pic50 in rows:
                if pic50 is not None:
                    measured_scores.append(pose.score)
                    measured_pic50.append(pic50)

        scores = np.array(measured_scores)
        pic50s = np.array(measured_pic50)
        print(f"\non the {len(scores)} compounds with a measured pIC50 "
              f"({pic50s.min():.1f} to {pic50s.max():.1f}):")
        print(f"  Vina score vs pIC50:         Spearman {regressor.spearman(scores, pic50s):>6.2f}")
        print("  Vina is negative-is-better, so a positive number here is the wrong sign.")
