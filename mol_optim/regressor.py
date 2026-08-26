"""The pIC50 regressor: the shared encoder plus a head, and an ensemble of them.

This is what the RL reward reads. Two deliberate choices. The head sees the
pooled embedding only — unlike the DQN's, it never reads size, or the agent would be
handed "add atoms, collect reward" as a direction. And the prediction is an ensemble
mean carrying its spread, which is largest where the data is thinnest.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem

from mol_optim import config, encoder, featurize


class Regressor(nn.Module):
    """A batch of molecular graphs to one predicted pIC50 each."""

    def __init__(self, cfg: config.Config):
        super().__init__()
        self.encoder = encoder.GraphEncoder(cfg.hidden_dim, cfg.num_message_passing_layers)
        self.linear_1 = nn.Linear(cfg.hidden_dim, 128)
        self.linear_2 = nn.Linear(128, 32)
        self.linear_3 = nn.Linear(32, 1)
        self.activation = nn.ReLU()

    def forward(self, batch: featurize.Batch) -> torch.Tensor:
        x = self.activation(self.linear_1(self.encoder(batch)))  # [num_graphs, 128]
        x = self.activation(self.linear_2(x))  # [num_graphs, 32]
        return self.linear_3(x).squeeze(-1)  # [num_graphs]


@dataclass(frozen=True)
class Prediction:
    """What an ensemble says about a batch of molecules."""

    mean: np.ndarray  # [num_molecules] predicted pIC50
    spread: np.ndarray  # [num_molecules] standard deviation across the ensemble


def predict(
    models: Sequence[Regressor], mols: Sequence[Chem.Mol], cfg: config.Config
) -> Prediction:
    """Ensemble mean and standard deviation, in batches of 256.

    Steps remaining is 0.0: there is no episode here, and this head never reads it.
    """
    columns = []
    for model in models:
        model.eval()
        with torch.no_grad():
            columns.append(
                torch.cat(
                    [
                        model(
                            featurize.tensors(
                                featurize.graphs(mols[start : start + 256]), 0.0, cfg
                            )
                        )
                        for start in range(0, len(mols), 256)
                    ]
                ).numpy()
            )
    stacked = np.stack(columns)  # [num_models, num_molecules]
    return Prediction(mean=stacked.mean(axis=0), spread=stacked.std(axis=0))


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    """Rank correlation: is the compound the model likes best the one that binds best?

    The only thing the RL agent uses the regressor for. MAE measures something else.
    """
    return float(np.corrcoef(_ranks(first), _ranks(second))[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged — duplicated pIC50 values are common at round numbers."""
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or ordered[index] != ordered[start]:
            ranks[order[start:index]] = (start + index - 1) / 2.0
            start = index
    return ranks


def roc_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Probability that a random positive outscores a random negative.

    The rank-sum identity, so no scikit-learn. Ties count a half, which `_ranks` handles.
    """
    pooled = np.concatenate([positive, negative])
    ranks = _ranks(pooled) + 1.0  # [len(pooled)], 1-based
    rank_sum = ranks[: len(positive)].sum()
    best_case = len(positive) * (len(positive) + 1) / 2.0
    return float((rank_sum - best_case) / (len(positive) * len(negative)))
