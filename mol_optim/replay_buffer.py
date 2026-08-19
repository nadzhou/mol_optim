"""Our own replay buffer — the OpenAI baselines one is TF1-era and does not install.

One transition is: the chosen candidate's packed fingerprint, the steps remaining when
it was chosen, the reward, every candidate available from the resulting state, and
done. The next-state field is a *set* of candidates because the target is a max over
next candidates, not over a fixed action head — that is the MolDQN formulation, and it
is why this buffer is ragged.

Fingerprints stay packed in here (see featurize); unpacking happens in the training
step, one batch at a time.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Batch:
    states: np.ndarray  # [batch, packed_length] uint8
    state_steps_remaining: np.ndarray  # [batch]
    rewards: np.ndarray  # [batch]
    next_candidates: tuple[np.ndarray, ...]  # arrays of [num_candidates, packed_length]
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
        state: np.ndarray,
        state_steps_remaining: int,
        reward: float,
        next_candidates: np.ndarray,
        next_steps_remaining: int,
        done: bool,
    ) -> None:
        # Copy on the way in: the training loop reuses its scratch arrays, and a buffer
        # holding a view would silently rewrite its own history.
        transition = (
            np.array(state, dtype=np.uint8, copy=True),
            float(state_steps_remaining),
            float(reward),
            np.array(next_candidates, dtype=np.uint8, copy=True),
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
            states=np.stack([transition[0] for transition in sampled]),
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
