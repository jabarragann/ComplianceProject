import json
import torch
from torch import nn
from torch.utils.data import DataLoader


class MLP(nn.Module):
    """
    Multilayer Perceptron for regression.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, x):
        """
        Forward pass
        """
        return self.layers(x)


class BestMLP1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers = []

        in_features = 12

        layers.append(nn.Linear(in_features, 114))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.0004279))
        layers.append(nn.Linear(114, 3))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class BestMLP2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers = []

        in_features = 12

        layers.append(nn.Linear(in_features, 140))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.00929993824427144))
        layers.append(nn.Linear(140, 3))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class CustomMLP(nn.Module):
    def __init__(self, trial) -> None:
        super().__init__()

    @staticmethod
    def define_model(trial):
        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        n_layers = trial.suggest_int("n_layers", 1, 5)
        layers = []

        in_features = 12
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 4, 200)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            p = trial.suggest_float("dropout_l{}".format(i), 0.00, 0.5)
            layers.append(nn.Dropout(p))
            in_features = out_features

        layers.append(nn.Linear(in_features, 3))

        return nn.Sequential(*layers)


class JointCorrectionNet(nn.Module):
    default_architecture = dict(n_layers=1, n_units_l0=140, dropout_l0=0.009299938244271444)

    def __init__(self, model_def: dict = None) -> None:
        super().__init__()

        if model_def is None:
            self.model_def = JointCorrectionNet.default_architecture
        else:
            self.model_def = model_def

        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        n_layers = self.model_def["n_layers"]

        layers = []
        in_features = 12
        for i in range(n_layers):
            out_features = self.model_def["n_units_l{}".format(i)]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            p = self.model_def["dropout_l{}".format(i)]
            layers.append(nn.Dropout(p))
            in_features = out_features

        layers.append(nn.Linear(in_features, 3))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out

    def model_def_to_json(self, root):
        json_str = json.dumps(self.model_def, indent=2)
        with open(root / "model_def.json", "w") as f:
            f.write(json_str)
