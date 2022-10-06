import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Custom
from kincalib.Recording.DataRecord import LearningRecord

from kincalib.utils.Logger import Logger


def remove_outliers(robot_joints, tracker_joints):
    rad2d = 180 / np.pi
    scale = np.array([rad2d, rad2d, 1000, rad2d, rad2d, rad2d])
    joints_cols = ["q1", "q2", "q3", "q4", "q5", "q6"]
    robot_joints_2 = robot_joints.rename(lambda x: x.replace("rq", "q"), axis=1)
    tracker_joints_2 = tracker_joints.rename(lambda x: x.replace("tq", "q"), axis=1)

    diff = (robot_joints_2 - tracker_joints_2) * scale
    diff = pd.DataFrame(diff, columns=joints_cols)

    quantile_df = diff.quantile([0.25, 0.5, 0.75])
    IQR = quantile_df.loc[0.75] - quantile_df.loc[0.25]

    upper = quantile_df.loc[0.75] + IQR * 1.5
    lower = quantile_df.loc[0.25] - IQR * 1.5

    clean_diff = diff.copy()
    for c in joints_cols:
        clean_diff = clean_diff.loc[(clean_diff[c] > lower[c]) & (clean_diff[c] < upper[c])]

    return clean_diff


def main(path):

    # Load dataset file
    path = Path(path)
    dataset = pd.read_csv(path)
    dataset = dataset.loc[dataset["opt"] < 2.5 / 1000]

    robot_joints = dataset[LearningRecord.robot_joints_cols]
    tracker_joints = dataset[LearningRecord.tracker_joints_cols]

    clean_diff = remove_outliers(robot_joints, tracker_joints)

    clean_dataset = dataset.loc[clean_diff.index.to_numpy(), :]
    new_name = path.with_suffix("").name + "_clean.csv"
    new_name = path.parent / new_name

    clean_dataset.to_csv(new_name)


if __name__ == "__main__":

    log = Logger(__name__).log

    default_path = "icra2023-data/neuralnet/dataset/final_dataset.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default=default_path, help="dataset_path")
    args = parser.parse_args()

    main(args.dataset)
