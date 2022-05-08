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
    def __init__(self, data_path: Path, mode: str, normalizer=None) -> None:
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
        output_cols = ["tq4", "tq5", "tq6"]
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
        x_norm = self.normalizer.transform(self.X[idx].reshape(-1, 12))
        x_norm = x_norm.squeeze()

        return x_norm, self.Y[idx]


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Datasets creation
    # ------------------------------------------------------------
    data_path = Path("data/deep_learning_data/random_dataset.txt")
    train_data = JointsDataset1(data_path, mode="train")
    scaler = StandardScaler()
    scaler.fit(train_data.X)
    train_data.normalizer = scaler

    valid_data = JointsDataset1(data_path, mode="valid", normalizer=scaler)
    test_data = JointsDataset1(data_path, mode="test", normalizer=scaler)

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
