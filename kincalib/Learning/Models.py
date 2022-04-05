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


class CustomMLP(nn.Module):
    def __init__(self, trial) -> None:
        super().__init__()

    @staticmethod
    def define_model(trial):
        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers = []

        in_features = 12
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            p = trial.suggest_float("dropout_l{}".format(i), 0.00, 0.5)
            layers.append(nn.Dropout(p))
            in_features = out_features

        layers.append(nn.Linear(in_features, 6))

        return nn.Sequential(*layers)
