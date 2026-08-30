from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from mol_optim.chem import graph_key, molio


@dataclass(frozen=True)
class Run:
    episode_rewards: tuple[float, ...]  # terminal reward, one per episode
    episode_molecules: tuple[Chem.Mol, ...]  # final graph, one per episode
    seconds: float

    @property
    def final_mean_reward(self) -> float:
        """Mean terminal reward over the last 100 episodes (all of them, if fewer).

        The comparison the beats-random test makes, so it lives in one place.
        """
        tail = self.episode_rewards[-100:]
        return sum(tail) / len(tail)

    @property
    def best(self) -> tuple[Chem.Mol, float]:
        best_index = max(
            range(len(self.episode_rewards)), key=lambda i: self.episode_rewards[i]
        )
        return self.episode_molecules[best_index], self.episode_rewards[best_index]


def top_k(run: Run, out_stem: Path, k: int = 12) -> None:
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
