"""Our own replay buffer — the OpenAI baselines one is TF1-era and does not install.

One transition is: the chosen candidate's graph, the steps remaining when it was
chosen, the reward, every candidate available from the resulting state, and done. The
next-state field is a *set* of candidates because the target is a max over next
candidates, not over a fixed action head — that is the MolDQN formulation, and it is
why this buffer is ragged.

Graphs are stored as featurize.Graphs, i.e. int8 codes, not one-hot floats. A 5000
transition buffer holds roughly 200k candidate graphs; the codes make that tens of MB.
"""

from dataclasses import dataclass

import numpy as np

from mol_optim import featurize


@dataclass(frozen=True)
class Batch:
    states: tuple[featurize.Graphs, ...]  # one graph each
    state_steps_remaining: np.ndarray  # [batch]
    rewards: np.ndarray  # [batch]
    next_candidates: tuple[featurize.Graphs, ...]  # a candidate set each, ragged
    next_steps_remaining: np.ndarray  # [batch]
    dones: np.ndarray  # [batch]


class ReplayBuffer:
    """FIFO over transitions, uniform sampling with replacement."""

    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity = capacity
        self.rng = rng
        self._transitions: list[tuple] = []
        self._next_index = 0

    def __len__(self) -> int:
        return len(self._transitions)

    def push(
        self,
        state: featurize.Graphs,
        state_steps_remaining: int,
        reward: float,
        next_candidates: featurize.Graphs,
        next_steps_remaining: int,
        done: bool,
    ) -> None:
        transition = (
            _owned_copy(state),
            float(state_steps_remaining),
            float(reward),
            _owned_copy(next_candidates),
            float(next_steps_remaining),
            float(done),
        )
        if len(self._transitions) < self.capacity:
            self._transitions.append(transition)
        else:
            self._transitions[self._next_index] = transition
        self._next_index = (self._next_index + 1) % self.capacity

    def sample(self, batch_size: int) -> Batch:
        if len(self._transitions) < batch_size:
            # Loud on purpose: a batch of 128 drawn from 3 transitions trains fine and
            # learns nothing. The caller waits until the buffer has filled.
            raise ValueError(
                f"Buffer holds {len(self._transitions)} transitions, need {batch_size}"
            )
        indices = self.rng.integers(0, len(self._transitions), size=batch_size)
        sampled = [self._transitions[i] for i in indices]
        return Batch(
            states=tuple(transition[0] for transition in sampled),
            state_steps_remaining=np.array(
                [transition[1] for transition in sampled], dtype=np.float32
            ),
            rewards=np.array([transition[2] for transition in sampled], dtype=np.float32),
            next_candidates=tuple(transition[3] for transition in sampled),
            next_steps_remaining=np.array(
                [transition[4] for transition in sampled], dtype=np.float32
            ),
            dones=np.array([transition[5] for transition in sampled], dtype=np.float32),
        )


def _owned_copy(graphs: featurize.Graphs) -> featurize.Graphs:
    """The buffer owns its arrays: a stored view would rewrite its own history."""
    return featurize.Graphs(
        atom_codes=graphs.atom_codes.copy(),
        bond_codes=graphs.bond_codes.copy(),
        edge_index=graphs.edge_index.copy(),
        graph_index=graphs.graph_index.copy(),
        num_graphs=graphs.num_graphs,
    )
