import torch
from torch import nn
from torch.utils.data import DataLoader


class MLP(nn.Module):
    """
    Multilayer Perceptron for regression.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 6))

    def forward(self, x):
        """
        Forward pass
        """
        return self.layers(x)
