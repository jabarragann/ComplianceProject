# Python
from sklearn.preprocessing import StandardScaler
import json
import os
from tkinter import Image
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd

# Torch
import torch
from torchvision.transforms import Compose
from torch.utils.data import Dataset, DataLoader

# Custom
from kincalib.utils.Logger import Logger

log = Logger(__name__).log
np.set_printoptions(precision=4, suppress=True, sign=" ")


class JointsDataset1(Dataset):
    def __init__(self, data_path: Path, mode: str, normalizer=None, full_output:bool=False) -> None:
        """Input robot state (joint position and torque) and output corrected wrist joints.

        Parameters
        ----------
        data_path : Path
            Path to data file
        mode: str
            Either 'train', 'valid' or 'tests'
        normalizer : sklearn.preprocessing.StandardScaler
            This can be added in the constructor or later. It is required to used the __getitem__ function.
        """
        super().__init__()
        self.data_df = pd.read_csv(data_path)
        assert mode in ["train", "valid", "test"], "invalid mode"
        self.mode = mode
        self.data_df = self.data_df.loc[self.data_df["flag"] == mode]
        self.normalizer: StandardScaler = normalizer

        input_cols = ["rq1", "rq2", "rq3", "rq4", "rq5", "rq6"] + ["rt1", "rt2", "rt3", "rt4", "rt5", "rt6"]
        if full_output:
            output_cols = ["tq1","tq2","tq3","tq4", "tq5", "tq6"]
        else:
            output_cols = ["tq4", "tq5", "tq6"]

        # Filter outliers with the optimization error of q56 lower than 1.4mm
        self.data_df = self.data_df.loc[self.data_df["opt"] < 1.4 / 1000]
        self.X = self.data_df[input_cols].to_numpy()
        self.Y = self.data_df[output_cols].to_numpy()

        self.X = torch.tensor(self.X.astype(np.float32))
        self.Y = torch.tensor(self.Y.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.normalizer is None:
            log.error("dataset normalizer is None.")
            exit(0)
        x_norm = self.normalizer(self.X[idx])

        return x_norm, self.Y[idx]


@dataclass
class Normalizer:
    xdata: np.ndarray
    state_dict: dict = None

    def __post_init__(self):
        """Calculate mean and std of data"""
        if self.xdata is not None:
            self.mean = self.xdata.mean(axis=0)
            self.std = self.xdata.std(axis=0)
        elif self.state_dict is not None:
            self.load_state_dict(self.state_dict)
        else:
            # log.info("state_dict and xdata cannot be None at the same time")
            raise ValueError("state_dict and xdata cannot be None at the same time")

    def __call__(self, x):
        return (x - self.mean) / self.std

    def reverse(self, x):
        return (x * self.std) + self.mean

    def get_state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    def load_state_dict(self, scale_values_dict):
        self.mean = np.array(scale_values_dict["mean"], dtype=np.float32)
        self.std = np.array(scale_values_dict["std"], dtype=np.float32)

    def to_json(self, path: Path):
        state_dict = self.get_state_dict()
        json_str = json.dumps(state_dict, indent=2)
        with open(path, "w") as f:
            f.write(json_str)


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Datasets creation
    # ------------------------------------------------------------
    data_path = Path("data/deep_learning_data/random_dataset.txt")
    train_data = JointsDataset1(data_path, mode="train")
    # normalizer = StandardScaler()
    # normalizer.fit(train_data.X)
    normalizer = Normalizer(train_data.X)
    train_data.normalizer = normalizer

    valid_data = JointsDataset1(data_path, mode="valid", normalizer=normalizer)
    test_data = JointsDataset1(data_path, mode="test", normalizer=normalizer)

    # ------------------------------------------------------------
    # Print dataset info
    # -----------------------------------------------------------
    log.info(f"Training size {len(train_data)}")
    log.info(f"Valid size {len(valid_data)}")
    log.info(f"test size {len(test_data)}")

    x, y = train_data[0]
    log.info(f"number of inputs {x.shape}")
    log.info(f"number of outputs {y.shape}")

    x, y = train_data[:]
    log.info(f"x shape {x.shape}")
    log.info(f"y shape {y.shape}")
    log.info(f"train x mean\n{x.mean(axis=0)}")
    log.info(f"train x std\n{x.std(axis=0)}")

    x, y = valid_data[:]
    log.info(f"valid x mean\n{x.mean(axis=0)}")
    log.info(f"valid x std\n{x.std(axis=0)}")

    x, y = train_data[0:5]

    dataloader = DataLoader(train_data, batch_size=5, shuffle=True)
    for x, y in dataloader:
        log.info(x.shape)
        log.info(y.shape)
        break
