"""Q network for Step 1: five explicit layers, no cleverness (MolDQN's dqn.py)."""

import torch
import torch.nn as nn


class MolDQN(nn.Module):
    """Scores one candidate molecule: [batch, input_length] -> [batch, 1]."""

    def __init__(self, input_length: int):
        super().__init__()
        self.linear_1 = nn.Linear(input_length, 1024)
        self.linear_2 = nn.Linear(1024, 512)
        self.linear_3 = nn.Linear(512, 128)
        self.linear_4 = nn.Linear(128, 32)
        self.linear_5 = nn.Linear(32, 1)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.linear_1(x))
        x = self.activation(self.linear_2(x))
        x = self.activation(self.linear_3(x))
        x = self.activation(self.linear_4(x))
        return self.linear_5(x)  # [batch, 1]
