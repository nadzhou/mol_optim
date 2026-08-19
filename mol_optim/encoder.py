"""The GNN encoder: a batch of molecular graphs to one vector each.

Its own module because three things share it — the Q network here, the ZINC masked-atom
pretraining at plan.md Step 3b, and the pIC50 regressor at Step 4. One checkpoint
initializes all three, and that sharing is the point.

Message passing is written out rather than taken from torch_geometric: the aggregation
is one `index_add_`, the pooling is another, and both are deterministic on CPU —
PyG's scatter kernels are not, which is exactly what Step 0 warns about.
"""

import torch
import torch.nn as nn

from mol_optim import featurize


class GraphEncoder(nn.Module):
    """[total_atoms, ...] node and edge features -> [num_graphs, hidden]."""

    def __init__(self, hidden: int, num_layers: int):
        super().__init__()
        self.atom_embedding = nn.Linear(featurize.ATOM_FEATURE_LENGTH, hidden)
        # One message and one update per round. Messages read the bond they cross, so a
        # double bond and a single bond between the same two atoms say different things.
        self.message_layers = nn.ModuleList(
            nn.Linear(hidden + featurize.BOND_FEATURE_LENGTH, hidden)
            for _ in range(num_layers)
        )
        self.update_layers = nn.ModuleList(
            nn.Linear(2 * hidden, hidden) for _ in range(num_layers)
        )
        self.activation = nn.ReLU()

    def forward(self, batch: featurize.Batch) -> torch.Tensor:
        h = self.activation(self.atom_embedding(batch.atom_features))  # [total_atoms, hidden]
        source, target = batch.edge_index[0], batch.edge_index[1]  # [total_edges] each

        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            messages = self.activation(
                message_layer(torch.cat([h[source], batch.bond_features], dim=-1))
            )  # [total_edges, hidden]
            aggregated = torch.zeros_like(h).index_add_(0, target, messages)
            h = self.activation(update_layer(torch.cat([h, aggregated], dim=-1)))

        # Mean pool. Sum would make the embedding grow with the molecule and swamp the
        # per-atom signal; the atom count reaches the network as a graph feature instead.
        summed = torch.zeros(batch.num_graphs, h.shape[1], dtype=h.dtype).index_add_(
            0, batch.graph_index, h
        )  # [num_graphs, hidden]
        counts = torch.zeros(batch.num_graphs, dtype=h.dtype).index_add_(
            0, batch.graph_index, torch.ones(len(h), dtype=h.dtype)
        )  # [num_graphs]
        return summed / counts.unsqueeze(-1)  # [num_graphs, hidden]
