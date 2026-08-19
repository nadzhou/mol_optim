"""The Q network: the GNN encoder plus a head that also reads the graph-level features."""

import torch
import torch.nn as nn

from mol_optim import config, encoder, featurize


class MolDQN(nn.Module):
    """Scores candidate molecules: a batch of graphs -> [num_graphs, 1].

    One scalar per candidate, no fixed-size action head — the action space is the
    candidate set the environment enumerates, and it changes size every step.
    """

    def __init__(self, cfg: config.Config):
        super().__init__()
        self.encoder = encoder.GraphEncoder(cfg.hidden_dim, cfg.num_message_passing_layers)
        # Steps remaining joins the *pooled* embedding, not the atom features: it is a
        # property of the episode, not of any atom.
        self.linear_1 = nn.Linear(cfg.hidden_dim + featurize.NUM_GRAPH_FEATURES, 128)
        self.linear_2 = nn.Linear(128, 32)
        self.linear_3 = nn.Linear(32, 1)
        self.activation = nn.ReLU()

    def forward(self, batch: featurize.Batch) -> torch.Tensor:
        embedded = self.encoder(batch)  # [num_graphs, hidden]
        x = torch.cat([embedded, batch.graph_features], dim=-1)  # [num_graphs, hidden + 2]
        x = self.activation(self.linear_1(x))
        x = self.activation(self.linear_2(x))
        return self.linear_3(x)  # [num_graphs, 1]
