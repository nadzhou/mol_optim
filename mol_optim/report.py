"""Showing a run's molecules to a person: a drawing and an SDF.

This is the boundary. Everything upstream is graphs; here they become a picture you can
look at and a file a chemist's software can open.
"""

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from mol_optim import graph_key, molio, results


def top_k(run: results.Run, out_stem: Path, k: int = 12) -> None:
    """Writes the k best distinct molecules of a run as `<stem>.png` and `<stem>.sdf`."""
    ranked = sorted(
        range(len(run.episode_rewards)), key=lambda i: -run.episode_rewards[i]
    )
    best: dict[str, int] = {}  # graph hash -> episode index, best first
    for index in ranked:
        best.setdefault(graph_key.canonical_hash(run.episode_molecules[index]), index)
        if len(best) == k:
            break
    indices = list(best.values())

    molecules = tuple(run.episode_molecules[i] for i in indices)
    rewards = [run.episode_rewards[i] for i in indices]
    molio.write(
        out_stem.with_suffix(".sdf"),
        molecules,
        {"reward": [f"{reward:.4f}" for reward in rewards], "episode": indices},
    )

    drawable = []
    for mol in molecules:
        flat = Chem.Mol(mol)
        AllChem.Compute2DCoords(flat)
        drawable.append(flat)
    image = Draw.MolsToGridImage(
        drawable,
        molsPerRow=4,
        subImgSize=(320, 260),
        legends=[
            f"episode {index}  reward {reward:.3f}"
            for index, reward in zip(indices, rewards)
        ],
        returnPNG=False,
    )
    image.save(out_stem.with_suffix(".png"))
