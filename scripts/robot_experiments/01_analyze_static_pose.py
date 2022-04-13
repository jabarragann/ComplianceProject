# Python imports
from pathlib import Path
import time
import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Set
from rich.logging import RichHandler
from rich.progress import track

# ROS and DVRK imports
import dvrk
import rospy

# kincalib module imports
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.Logger import Logger

log = Logger("template").log
np.set_printoptions(precision=4, suppress=True, sign=" ")


def create_histogram(data, axes, title=None, xlabel=None):
    axes.hist(data, bins=30, edgecolor="black", linewidth=1.2, density=False)
    axes.grid()
    axes.set_xlabel(xlabel)
    axes.set_ylabel("joint value (deg)")
    axes.set_title(f"{title} (N={data.shape[0]:02d})")
    # axes.set_xticks([i * 5 for i in range(110 // 5)])
    # plt.show()


def main():
    # ------------------------------------------------------------
    # Load and plot
    # ------------------------------------------------------------
    path = Path("data/04_robot_measuring_experiments/static_error/collection1.txt")
    data_df = pd.read_csv(path)
    fig, axes = plt.subplots(2, 3)

    for idx, ax in enumerate(axes.reshape(-1)):
        data = data_df[f"q{idx+1}"]
        create_histogram(data * 180 / np.pi, ax, title=f"q{idx+1}")

    fig, axes = plt.subplots(1, 3)
    create_histogram(data_df["x"] * 1000, axes[0], title=f"x")
    create_histogram(data_df["y"] * 1000, axes[1], title=f"y")
    create_histogram(data_df["z"] * 1000, axes[2], title=f"z")

    plt.show()


if __name__ == "__main__":

    main()
