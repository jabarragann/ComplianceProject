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
from kincalib.geometry import Line3D, Circle3D, Plane3D, Plotter3D, Triangle3D, dist_circle3_plane
import kincalib.utils.CmnUtils as utils
from kincalib.Calibration.CalibrationUtils import CalibrationUtils as calib

np.set_printoptions(precision=4, suppress=True, sign=" ")


def plot_circles(circles_list, colors=None):
    if colors is None:
        colors = ["black"] * len(circles_list)

    plotter = Plotter3D()
    for ci, color in zip(circles_list, colors):
        plotter.scatter_3d(ci.generate_pts(40), marker="o", color=color)
        plotter.scatter_3d(ci.samples, marker="*", color="blue")
    return plotter


def main():

    # Important paths
    root = Path(args.root)

    # Load files
    dict_files = load_registration_data(root)
    keys = sorted(list(dict_files.keys()))
    k = 1200
    if len(list(dict_files[k].keys())) < 2:
        log.warning(f"files for step {k} are not available")
        exit(0)

    log.info(f"Loading files from {root}")
    log.info(f"Loading files from step {k}")

    # Get roll circles
    roll_cir1, roll_cir2 = calib.create_roll_circles(dict_files[k]["roll"])
    # Get pitch and yaw circles
    pitch_yaw_circles = calib.create_yaw_pitch_circles(dict_files[k]["pitch"])
    # Estimate pitch origin with roll and pitch
    m1, m2, m3 = calib.calculate_pitch_origin(
        roll_cir1, pitch_yaw_circles[0]["pitch"], pitch_yaw_circles[1]["pitch"]
    )
    pitch_orig_est1 = (m1 + m2 + m3) / 3

    for kk in range(2):
        log.info(f"Estimation for roll value {kk}")
        pitch_cir, yaw_cir = pitch_yaw_circles[kk]["pitch"], pitch_yaw_circles[kk]["yaw"]

        # Estimate pitch and yaw origin with yaw and pitch
        l1 = Line3D(ref_point=pitch_cir.center, direction=pitch_cir.normal)
        l2 = Line3D(ref_point=yaw_cir.center, direction=yaw_cir.normal)
        inter_params = []
        l3 = Line3D.perpendicular_to_skew(l1, l2, intersect_params=inter_params)
        pitch_orig_est2 = l1(inter_params[0][0])
        yaw_orig_est = l2(inter_params[0][1])
        pitch2yaw1 = np.linalg.norm(pitch_orig_est1 - yaw_orig_est)
        pitch2yaw2 = np.linalg.norm(pitch_orig_est2 - yaw_orig_est)

        # Estimate wrist fiducial in yaw origin

        # Get location of wrist fiducial
        solutions_pts, solutions = dist_circle3_plane(pitch_cir, roll_cir2.get_plane())

        # Summary of analysis
        log.info(f"Pitch orig est with roll pitch:     {pitch_orig_est1}")
        log.info(f"Pitch orig est with pitch yaw:      {pitch_orig_est2}")
        log.info(f"error:                              {pitch_orig_est1-pitch_orig_est2}")
        log.debug(f"test                               {l3(0.0)}")
        log.info(f"yaw orig est:                       {yaw_orig_est}")
        log.debug(f"test                               {l3(inter_params[0][2])}")
        log.info(f"pitch2yaw:                          {pitch2yaw1:0.05f}")
        log.info(f"pitch2yaw:                          {pitch2yaw2:0.05f}")
        log.info(f"wrist fiducial in yaw frame.\n {solutions_pts}")

    # Plot
    circles = [roll_cir2, pitch_cir, yaw_cir]
    plotter = plot_circles(circles, ["black", "orange", "orange"])
    plotter.scatter_3d(np.array(solutions_pts).T, marker="*", marker_size=100, color="green")
    plotter.scatter_3d(
        np.array([pitch_orig_est1, pitch_orig_est2, yaw_orig_est]).T,
        marker="o",
        marker_size=100,
        color="green",
    )

    plt.show()


# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
marker_file = Path("./share/custom_marker_id_112.json")
parser = argparse.ArgumentParser()
# fmt:off
parser.add_argument( "-r", "--root", type=str, default="./data/03_replay_trajectory/d04-rec-06-traj01", 
                        help="root dir") 
parser.add_argument( "-l", "--log", type=str, default="INFO", 
                        help="log level") #fmt:on
args = parser.parse_args()
log_level = args.log
log = Logger("pitch_exp_analize2", log_level=log_level).log

if __name__ == "__main__":
    main()
