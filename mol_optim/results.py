"""What a run returns. Plain data, shared by the DQN loop and the random baseline."""

from dataclasses import dataclass

from rdkit import Chem


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
