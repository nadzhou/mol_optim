"""PPO's network and the one primitive a variable action set needs.

The MDP hands the agent a candidate set that changes size every step, so there is no
fixed-size action head to softmax over. The policy scores each candidate to a logit and
normalizes *within* that step's set — a segment softmax over a ragged block, the same
shape problem `replay_buffer.py` solves for the DQN target's segment max.

Same encoder as `dqn.py`, so one ZINC checkpoint still initializes both.
"""

import torch
import torch.nn as nn

from mol_optim import config, encoder, featurize


class MolPPO(nn.Module):
    """Two heads on the shared encoder: a logit per candidate, a value per state."""

    def __init__(self, cfg: config.Config):
        super().__init__()
        self.encoder = encoder.GraphEncoder(
            cfg.hidden_dim, cfg.num_message_passing_layers
        )
        self.policy = nn.Sequential(
            nn.Linear(cfg.hidden_dim + featurize.NUM_GRAPH_FEATURES, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # Separate head, shared trunk. The value of a state and the appeal of a
        # candidate are different questions asked of the same embedding.
        self.value = nn.Sequential(
            nn.Linear(cfg.hidden_dim + featurize.NUM_GRAPH_FEATURES, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _trunk(self, batch: featurize.Batch) -> torch.Tensor:
        embedded = self.encoder(batch)  # [num_graphs, hidden]
        return torch.cat([embedded, batch.graph_features], dim=-1)

    def logits(self, batch: featurize.Batch) -> torch.Tensor:
        """One unnormalized score per candidate — [num_graphs]."""
        return self.policy(self._trunk(batch)).squeeze(-1)

    def values(self, batch: featurize.Batch) -> torch.Tensor:
        """One value per state — [num_graphs]."""
        return self.value(self._trunk(batch)).squeeze(-1)


def segment_log_softmax(
    logits: torch.Tensor, owner: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Log-softmax within each segment of a ragged block — [total_candidates].

    `owner[i]` is which step candidate i belongs to. Written out rather than padding to
    a rectangle: candidate sets differ by an order of magnitude between an early
    one-atom state and a 25-atom one, so padding would be mostly mask.
    """
    # Subtract the per-segment max before exponentiating, or a large logit overflows.
    highest = torch.full((num_segments,), float("-inf"), device=logits.device)
    highest = highest.scatter_reduce(
        0, owner, logits, reduce="amax", include_self=False
    )  # [num_segments]
    shifted = logits - highest[owner]
    summed = torch.zeros(num_segments, device=logits.device).index_add_(
        0, owner, shifted.exp()
    )  # [num_segments]
    return shifted - summed.log()[owner]


def segment_entropy(
    log_probs: torch.Tensor, owner: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Entropy of each step's distribution — [num_segments].

    Reported as well as regularized: a step with 400 candidates starts near log(400),
    and watching it fall is how policy collapse shows up before the reward curve says so.
    """
    return torch.zeros(num_segments, device=log_probs.device).index_add_(
        0, owner, -log_probs * log_probs.exp()
    )
