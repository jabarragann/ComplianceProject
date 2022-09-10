import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Torch
import optuna
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# Custom
# from kincalib.Learning.Dataset import JointsDataset, JointsRawDataset, Normalizer
from kincalib.Learning.Dataset2 import JointsDataset1, Normalizer
from kincalib.Learning.Models import MLP, CustomMLP, JointCorrectionNet
from kincalib.Learning.Trainer import TrainRegressionNet
from torchsuite.HyperparameterTuner import OptuneStudyAbstract
from torchsuite.TrainingBoard import TrainingBoard
from kincalib.Recording.DataRecord import LearningRecord

from kincalib.utils.CmnUtils import mean_std_str
from kincalib.utils.Logger import Logger
import seaborn as sns


def plot_joints_difference(robot_joints, tracker_joints):
    rad2d = 180 / np.pi
    scale = np.array([rad2d, rad2d, 1000, rad2d, rad2d, rad2d])
    joints_cols = ["q1", "q2", "q3", "q4", "q5", "q6"]
    robot_joints_2 = robot_joints.rename(lambda x: x.replace("rq", "q"), axis=1)
    tracker_joints_2 = tracker_joints.rename(lambda x: x.replace("tq", "q"), axis=1)

    diff = (robot_joints_2 - tracker_joints_2) * scale
    diff = pd.DataFrame(diff, columns=joints_cols)

    log.info(f"N={diff.shape[0]}")

    # layout
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    for i, ax in enumerate(axes.reshape(-1)):
        sns.boxplot(data=diff, x=f"q{i+1}", ax=ax)
        sns.stripplot(data=diff, x=f"q{i+1}", ax=ax, color="black")
    return axes


def main(path):

    # Load dataset file
    path = Path(path)
    dataset = pd.read_csv(path)
    dataset = dataset.loc[dataset["opt"] < 2.5 / 1000]

    robot_joints = dataset[LearningRecord.robot_joints_cols]
    tracker_joints = dataset[LearningRecord.tracker_joints_cols]

    diff_ax = plot_joints_difference(robot_joints, tracker_joints)
    plt.show()
    # Create box plots


if __name__ == "__main__":

    log = Logger(__name__).log

    default_path = "./data/deep_learning_data/rec20_data_v2.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default=default_path, help="dataset_path")
    args = parser.parse_args()

    main(args.dataset)
