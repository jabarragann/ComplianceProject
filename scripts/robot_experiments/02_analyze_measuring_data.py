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
from kincalib.utils.CmnUtils import mean_std_str
import rospy

# kincalib module imports
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.Logger import Logger

log = Logger("template").log
np.set_printoptions(precision=4, suppress=True, sign=" ")

phantom_coord = {
    "1": np.array([-12.7, 12.7, 0]),
    "2": np.array([-12.7, 31.75, 0]),
    "3": np.array([-58.42, 31.75, 0]),
    "4": np.array([-53.34, 63.5, 0]),
    "5": np.array([-12.7, 63.5, 0]),
    "6": np.array([-12.7, 114.3, 0]),
    "7": np.array([-139.7, 114.3, 0]),
    "8": np.array([-139.7, 63.5, 0]),
    "9": np.array([-139.7, 12.7, 0]),
    "A": np.array([-193.675, 19.05, 25.4]),
    "B": np.array([-193.675, 44.45, 50.8]),
    "C": np.array([-193.675, 69.85, 76.2]),
    "D": np.array([-193.675, 95.25, 101.6]),
}


def main():
    # ------------------------------------------------------------
    # Load and plot
    # ------------------------------------------------------------
    loc_str = "7_C"
    loc = loc_str.strip().split("_")

    path = Path(f"data/04_robot_measuring_experiments/measuring/Collection{loc_str}.csv")
    data_df = pd.read_csv(path)

    data_l1 = data_df.loc[data_df["label"] == loc[0]][["x", "y", "z"]].to_numpy()
    data_l2 = data_df.loc[data_df["label"] == loc[1]][["x", "y", "z"]].to_numpy()
    error_vect = data_l1 - data_l2
    error_mag = np.linalg.norm(error_vect, axis=1)

    log.info(
        f"Distance between {loc[0]} and {loc[1]} in mm (True): {np.linalg.norm(phantom_coord[loc[0]]-phantom_coord[loc[1]]):0.4f}"
    )

    log.info(f"Solution (N={error_vect.shape[0]}): {mean_std_str(error_mag.mean() * 1000, error_mag.std() * 1000)}")

    plt.show()


if __name__ == "__main__":

    main()
