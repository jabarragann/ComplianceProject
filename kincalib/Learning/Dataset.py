# Python
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


class JointsRawDataset:
    def __init__(self, input_dir: Path, mode: str):
        train_paths = [1, 2, 3]
        valid_paths = [4]
        base_dir = input_dir / Path("test_trajectories/")

        if mode == "train":
            paths = train_paths
        elif mode == "valid":
            paths = valid_paths
        else:
            raise Exception("Mode can only be train or valid")

        self.X = []
        self.Y = []
        for p in paths:
            tracker_data = pd.read_csv(base_dir / f"{p:02d}" / "result" / "tracker_joints.txt")
            robot_data = pd.read_csv(base_dir / f"{p:02d}" / "robot_jp.txt")
            robot_data = robot_data.loc[robot_data["step"].isin(tracker_data["step"])]

            tracker_data = tracker_data.iloc[:, 1:].to_numpy()
            robot_data = robot_data.drop(columns="q7")
            robot_data = robot_data.iloc[:, 2:].to_numpy()

            self.X.append(robot_data)
            self.Y.append(tracker_data)
        self.X = np.vstack(self.X).astype(np.float32)
        self.Y = np.vstack(self.Y).astype(np.float32)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        """ """
        return self.X.shape[0]


class JointsDataset(Dataset):
    def __init__(self, xdata, ydata, normalizer) -> None:
        super().__init__()
        self.X = xdata
        self.Y = ydata
        self.normalizer = normalizer

    def __len__(self):
        """ """
        return self.X.shape[0]

    def __getitem__(self, idx):
        """ """
        # Normalize features to (0,1) interval
        x_norm = self.normalizer(self.X[idx])

        return x_norm, self.Y[idx]


@dataclass
class Normalizer:
    xdata: np.ndarray
    ydata: np.ndarray

    def __post_init__(self):
        """Calculate mean and std of data"""
        self.mean = self.xdata.mean(axis=0)
        self.std = self.xdata.std(axis=0)

    def __call__(self, x):
        return (x - self.mean) / self.std

    def reverse(self, x):
        return (x * self.std) + self.mean


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Datasets creation
    # ------------------------------------------------------------
    test_path = Path("data/03_replay_trajectory/d04-rec-10-traj01")
    train_data = JointsRawDataset(test_path, mode="train")
    valid_data = JointsRawDataset(test_path, mode="valid")
    normalizer = Normalizer(*train_data[:])
    train_dataset = JointsDataset(*train_data[:], normalizer)
    valid_dataset = JointsDataset(*valid_data[:], normalizer)

    # ------------------------------------------------------------
    # Print dataset info
    # -----------------------------------------------------------
    log.info(f"Training size {len(train_dataset)}")
    log.info(f"Valid size {len(valid_dataset)}")
    x, y = train_dataset[0]
    log.info(f"number of inputs {x.shape}")
    log.info(f"number of outputs {y.shape}")

    x, y = train_dataset[:]
    log.info(f"x shape {x.shape}")
    log.info(f"y shape {y.shape}")
    log.info(f"train x mean {x.mean():0.4f}")
    log.info(f"train x std {x.std():0.4f}")
    x, y = valid_dataset[:]
    log.info(f"valid x mean {x.mean():0.4f}")
    log.info(f"valid x std {x.std():0.4f}")

    dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    for x, mask in dataloader:
        log.info(x.shape)
        break
