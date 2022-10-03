# Python imports
from pathlib import Path
import time
import argparse
import sys
from venv import create
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Set
from rich.logging import RichHandler
from rich.progress import track
import re
from collections import defaultdict
import traceback
import seaborn as sns

# ROS and DVRK imports
import dvrk
import rospy
import PyKDL

# kincalib module imports
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.ExperimentUtils import (
    load_registration_data,
    calculate_midpoints,
    separate_markerandfiducial,
)
from kincalib.Geometry.geometry import (
    Line3D,
    Circle3D,
    Plane3D,
    Plotter3D,
    Triangle3D,
    dist_circle3_plane,
)
import kincalib.utils.CmnUtils as utils

np.set_printoptions(precision=4, suppress=True, sign=" ")


def plot_circles(circles_list, colors=None):
    if colors is None:
        colors = ["black"] * len(circles_list)

    plotter = Plotter3D()
    for ci, color in zip(circles_list, colors):
        plotter.scatter_3d(ci.generate_pts(40), marker="o", color=color)
        plotter.scatter_3d(ci.samples, marker="*", color="blue")
    return plotter


def create_roll_circles(roll_df) -> List[Circle3D]:
    df = roll_df
    roll = df.q4.unique()
    marker_orig_arr = []  # shaft marker
    fid_arr = []  # Wrist fiducial
    for r in roll:
        df_temp = df.loc[df["q4"] == r]
        pose_arr, wrist_fiducials = separate_markerandfiducial(None, marker_file, df=df_temp)
        if len(pose_arr) > 0 and len(wrist_fiducials) > 0:
            marker_orig_arr.append(list(pose_arr[0].p))
            fid_arr.append(wrist_fiducials)

    marker_orig_arr = np.array(marker_orig_arr)
    fid_arr = np.array(fid_arr)
    roll_cir1 = Circle3D.from_lstsq_fit(marker_orig_arr)
    roll_cir2 = Circle3D.from_lstsq_fit(fid_arr)

    return roll_cir1, roll_cir2


def create_yaw_pitch_circles(py_df) -> List[Circle3D]:
    df = py_df
    # roll values
    roll = df.q4.unique()

    pitch_arr = []
    yaw_arr = []
    for r in roll:
        df_temp = df.loc[(df["q4"] == r) & (df["q6"] == 0.0)]
        pose_arr, wrist_fiducials = separate_markerandfiducial(None, marker_file, df=df_temp)
        pitch_arr = wrist_fiducials

        df_temp = df.loc[(df["q4"] == r) & (df["q5"] == 0.0)]
        pose_arr, wrist_fiducials = separate_markerandfiducial(None, marker_file, df=df_temp)
        yaw_arr = wrist_fiducials

        break

    pitch_cir = Circle3D.from_lstsq_fit(pitch_arr.T)
    yaw_cir = Circle3D.from_lstsq_fit(yaw_arr.T)

    return yaw_cir, pitch_cir


def main():

    # Important paths
    root = Path(args.root)

    # Load files
    dict_files = load_registration_data(root)
    keys = sorted(list(dict_files.keys()))
    k = 440
    if len(list(dict_files[k].keys())) < 2:
        log.warning(f"files for step {k} are not available")
        exit(0)

    log.info(f"Loading files from {root}")

    # Get roll circles
    roll_cir1, roll_cir2 = create_roll_circles(dict_files[k]["roll"])
    # Get pitch and yaw circles
    yaw_cir, pitch_cir = create_yaw_pitch_circles(dict_files[k]["pitch"])

    circles = [roll_cir2, pitch_cir, yaw_cir]

    solutions_pts, solutions = dist_circle3_plane(pitch_cir, roll_cir2.get_plane())
    solutions_pts2, solutions2 = dist_circle3_plane(roll_cir2, pitch_cir.get_plane())
    plotter = plot_circles(circles, ["black", "orange", "orange"])
    plotter.scatter_3d(np.array(solutions_pts).T, marker="*", marker_size=200, color="green")
    plotter.scatter_3d(np.array(solutions_pts2).T, marker="*", marker_size=200, color="red")
    log.info(f"estimated wrist fiducial in tracker frame .\n {solutions_pts}")
    log.info(f"Error of wrist fiducial position. \n{solutions_pts - solutions_pts2}")

    plt.show()


# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
marker_file = Path("./share/custom_marker_id_112.json")
parser = argparse.ArgumentParser()
# fmt:off
parser.add_argument( "-r", "--root", type=str, default="./data/03_replay_trajectory/d04-rec-06-traj01", 
                        help="root dir") 
parser.add_argument( "-l", "--log", type=str, default="DEBUG", 
                        help="log level") #fmt:on
args = parser.parse_args()
log_level = args.log
log = Logger("pitch_exp_analize2", log_level=log_level).log

if __name__ == "__main__":
    main()
