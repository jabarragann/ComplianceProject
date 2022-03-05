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
from kincalib.utils.Frame import Frame
import rospy

# kincalib module imports
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.ExperimentUtils import load_registration_data, calculate_midpoints
from kincalib.geometry import Line3D, Circle3D, Plotter3D, Triangle3D, dist_circle3_plane
import kincalib.utils.CmnUtils as utils
from kincalib.Calibration.CalibrationUtils import CalibrationUtils as calib

np.set_printoptions(precision=4, suppress=True, sign=" ")


def create_histogram(data, axes, title=None, xlabel=None):
    axes.hist(data, bins=30, edgecolor="black", linewidth=1.2, density=False)
    # axes.hist(data, bins=50, range=(0, 100), edgecolor="black", linewidth=1.2, density=False)
    axes.grid()
    axes.set_xlabel(xlabel)
    axes.set_ylabel("Frequency")
    axes.set_title(f"{title} (N={data.shape[0]:02d})")
    # axes.set_xticks([i * 5 for i in range(110 // 5)])
    # plt.show()


def main():

    # Important paths
    root = Path(args.root)
    marker_file = Path("./share/custom_marker_id_112.json")
    regex = ""

    dict_files = load_registration_data(root)
    keys = sorted(list(dict_files.keys()))

    list_area = []
    list_pitch_orig_M = []
    list_roll_axis_M = []
    list_pitch_axis_M = []
    list_pitch2yaw = []
    list_fiducial_Y = []

    prev_y_ax = None
    prev_p_ax = None
    prev_r_ax = None
    for k in keys:
        if len(list(dict_files[k].keys())) < 2:
            log.warning(f"files for step {k} are not available")
            continue

        try:
            # Get roll circles
            roll_cir1, roll_cir2 = calib.create_roll_circles(dict_files[k]["roll"])
            # Get pitch and yaw circles
            pitch_yaw_circles = calib.create_yaw_pitch_circles(dict_files[k]["pitch"])
            # Estimate pitch origin with roll and pitch
            m1, m2, m3 = calib.calculate_pitch_origin(
                roll_cir1, pitch_yaw_circles[0]["pitch"], pitch_yaw_circles[1]["pitch"]
            )
            # ---------------------------
            # Marker to pitch frame error metrics
            # - Calculate pitch origin in marker Frame
            # - Calculate pitch axes in marker Frame
            # - Calculate roll axes in marker Frame
            # ---------------------------
            pitch_ori_T = (m1 + m2 + m3) / 3
            # Marker2Tracker
            T_TM1 = utils.pykdl2frame(pitch_yaw_circles[0]["marker_pose"])
            pitch_ori1_M = T_TM1.inv() @ pitch_ori_T
            pitch_ax1_M = T_TM1.inv().r @ pitch_yaw_circles[0]["pitch"].normal
            roll_axis1_M = T_TM1.inv().r @ roll_cir1.normal

            T_TM2 = utils.pykdl2frame(pitch_yaw_circles[1]["marker_pose"])
            pitch_ori2_M = T_TM2.inv() @ pitch_ori_T
            pitch_ax2_M = T_TM2.inv().r @ pitch_yaw_circles[1]["pitch"].normal
            roll_axis2_M = T_TM2.inv().r @ roll_cir1.normal

            # Estimate wrist fiducial in yaw origin
            fiducial_Y, pitch2yaw1 = calib.calculate_fiducial_from_yaw(
                pitch_ori_T, pitch_yaw_circles, roll_cir2
            )

            # Maker sure that all axis look in the same direction
            # Check if pitch axis are looking in opposite directions
            if np.dot(pitch_ax1_M, pitch_ax2_M) < 0:
                log.warning("INCONSISTENT NORMAL ASSIGNMENT")
                pitch_ax1_M *= -1
            if prev_p_ax is None:
                prev_p_ax = pitch_ax1_M
            # Then make sure that all the vectors from all the steps in the trajectory point in the same direction
            else:
                if np.dot(prev_p_ax, pitch_ax1_M) < 0:
                    log.warning("INCONSISTENT NORMAL ASSIGNMENT")
                    pitch_ax1_M *= -1
                    pitch_ax2_M *= -1
                prev_p_ax = pitch_ax1_M

            # Check if roll axis are looking in opposite directions
            if np.dot(roll_axis1_M, roll_axis2_M) < 0:
                log.warning("INCONSISTENT NORMAL ASSIGNMENT")
                roll_axis1_M *= -1
            if prev_r_ax is None:
                prev_r_ax = roll_axis1_M
            # Then make sure that all the vectors from all the steps in the trajectory point in the same direction
            else:
                if np.dot(prev_r_ax, roll_axis1_M) < 0:
                    log.warning("INCONSISTENT NORMAL ASSIGNMENT")
                    roll_axis1_M *= -1
                    roll_axis2_M *= -1
                prev_r_ax = roll_axis1_M

            # ---------------------------
            # Mid point error metrics
            # ---------------------------
            roll_axis = roll_cir1.normal
            pitch_axis1 = pitch_yaw_circles[0]["pitch"].normal
            pitch_axis2 = pitch_yaw_circles[1]["pitch"].normal

            triangle = Triangle3D([m1, m2, m3])
            # Scale sides and area to milimiters
            sides = triangle.calculate_sides(scale=1000)
            area = triangle.calculate_area(scale=1000)
            # fmt: off
            log.debug(f"Step {k} results")
            log.debug(f"MID POINT RESULTS")
            log.debug(f"triangle sides {sides} mm")
            log.debug(f"triangle area {area:0.4f} mm^2")
            log.debug(f"SHAFT RESULTS")
            log.debug(f"pitch1 dot roll in tracker {np.dot(roll_axis, pitch_axis1)}")
            log.debug(f"pitch2 dot roll in tracker {np.dot(roll_axis, pitch_axis2)}")
            log.debug(f"MARKER TO PITCH FRAME RESULTS")
            log.debug(f"pitch origin1 from marker {pitch_ori1_M.squeeze()}")
            log.debug(f"pitch origin2 from marker {pitch_ori2_M.squeeze()}")
            log.debug(f"pitch axis1  from marker {pitch_ax1_M}")
            log.debug(f"pitch axis2  from marker {pitch_ax2_M}")
            log.debug( f"Is dot product betweeen pitch axis positive? {np.dot(pitch_ax1_M,pitch_ax2_M)>0}")
            log.debug(f"roll axis1  from marker {roll_axis1_M}")
            log.debug(f"roll axis2  from marker {roll_axis2_M}")
            log.debug( f"Is dot product betweeen roll axis positive? {np.dot(roll_axis1_M,roll_axis2_M)>0}")
            log.debug(f"Fiducial in Jaw 1 {fiducial_Y[0].squeeze()}")
            log.debug(f"Fiducial in Jaw 2 {fiducial_Y[1].squeeze()}")
            # fmt: on

            # Add data to list
            if area < 3:
                list_area.append(area)

                list_pitch2yaw.append(pitch2yaw1)
                list_fiducial_Y.append(fiducial_Y[0].squeeze())
                list_fiducial_Y.append(fiducial_Y[1].squeeze())
                # Data from roll 1 position
                list_pitch_orig_M.append(pitch_ori1_M.squeeze())
                list_pitch_axis_M.append(pitch_ax1_M)
                list_roll_axis_M.append(roll_axis2_M)
                # Data from roll 2 position
                list_pitch_orig_M.append(pitch_ori1_M.squeeze())
                list_pitch_axis_M.append(pitch_ax2_M)
                list_roll_axis_M.append(roll_axis2_M)
            else:
                log.warning(f"Data in step {k} was not added to list")

        except Exception as e:
            log.error(f"Exception triggered in step {k}")
            log.debug(e)
            log.debug(traceback.print_exc())

    list_area = np.array(list_area)
    pitch_orig = np.array(list_pitch_orig_M)
    pitch_axis = np.array(list_pitch_axis_M)
    roll_axis = np.array(list_roll_axis_M)
    list_fiducial_Y = np.array(list_fiducial_Y)
    list_pitch2yaw = np.array(list_pitch2yaw)

    f_path = root / "registration_results/error_metrics"
    if not f_path.exists():
        f_path.mkdir(parents=True)

    np.save(f_path / "error_metric_pitch2yaw", list_pitch2yaw)
    np.save(f_path / "error_metric_fid_in_yaw", list_fiducial_Y)
    np.save(f_path / "error_metric_area", list_area)
    np.save(f_path / "error_metric_pitch_orig", pitch_orig)
    np.save(f_path / "error_metric_pitch_axis", pitch_axis)
    np.save(f_path / "error_metric_roll_axis", roll_axis)


def plot_results():
    root = Path(args.root)
    f_path = root / "registration_results/error_metrics"

    # Plot area histograms

    list_pitch2yaw = np.load(f_path / "error_metric_pitch2yaw.npy")
    list_area = np.load(f_path / "error_metric_area.npy")
    fiducial_Y = np.load(f_path / "error_metric_fid_in_yaw.npy")
    pitch_orig = np.load(f_path / "error_metric_pitch_orig.npy")
    pitch_orig_df = pd.DataFrame(pitch_orig, columns=["px", "py", "pz"])
    pitch_axis = np.load(f_path / "error_metric_pitch_axis.npy")
    pitch_axis_df = pd.DataFrame(pitch_axis, columns=["px", "py", "pz"])
    roll_axis = np.load(f_path / "error_metric_roll_axis.npy")
    roll_axis_df = pd.DataFrame(roll_axis, columns=["px", "py", "pz"])

    def mean_std_str_vect(mean_vect, std_vect):
        str = "["
        plus_minus_sign = "\u00B1"
        for m, s in zip(mean_vect.squeeze(), std_vect.squeeze()):
            str += f"{m:+0.04f}{plus_minus_sign}{s:0.04f},"
        return str[:-1] + "]"

    def mean_std_str(mean, std):
        str = ""
        plus_minus_sign = "\u00B1"
        str += f"{mean:+0.04f}{plus_minus_sign}{std:0.04f},"
        return str[:-1]

    plus_minus_sign = "\u00B1"
    log.info(f"Error report for {root}. (N={list_area.shape[0]})")
    log.info(f"Mean area (mm^2): {mean_std_str(list_area.mean(),list_area.std())}")
    log.info(f"pitch2yaw (m):    {mean_std_str(list_pitch2yaw.mean(),list_pitch2yaw.std())}")
    log.info(f"Rotation axis from Marker frame")
    log.info(f"Pitch_orig_M {mean_std_str_vect(pitch_orig.mean(axis=0),pitch_orig.std(axis=0))}")
    log.info(f"Pitch_axis_M {mean_std_str_vect(pitch_axis.mean(axis=0),pitch_axis.std(axis=0))}")
    log.info(f"Roll_axis_M  {mean_std_str_vect(roll_axis.mean(axis=0),roll_axis.std(axis=0))}")
    log.info(f"Fiducial from Yaw frame")
    log.info(f"fiducial_M   {mean_std_str_vect(fiducial_Y.mean(axis=0),fiducial_Y.std(axis=0))}")

    roll_pitch_dot = np.dot(roll_axis.mean(axis=0), pitch_axis.mean(axis=0))
    log.info(f"mean roll and mean pitch dot {roll_pitch_dot:0.05f}")

    # create_histogram(list_area)
    # Plot axis
    pitch_a_mean_center = pitch_axis_df - pitch_axis_df.mean()
    pitch_axis_df = pitch_a_mean_center.melt(var_name="coordinate")
    roll_a_mean_center = roll_axis_df - roll_axis_df.mean()
    roll_axis_df = roll_a_mean_center.melt(var_name="coordinate")
    pitch_o_mean_center = pitch_orig_df - pitch_orig_df.mean()
    pitch_orig_df = pitch_o_mean_center.melt(var_name="coordinate")

    # Box plots for transformation between marker and pitch
    fig, axes = plt.subplots(3, 1)
    plt.subplots_adjust(left=0.06, right=0.96, top=0.96, bottom=0.05, hspace=0.28)
    axes[0].set_title("std pitch_axis from M")
    sns.boxplot(x="coordinate", y="value", data=pitch_axis_df, ax=axes[0])
    axes[1].set_title("std roll_axis from M")
    sns.boxplot(x="coordinate", y="value", data=roll_axis_df, ax=axes[1])
    axes[2].set_title("std pitch_orig from M")
    sns.boxplot(x="coordinate", y="value", data=pitch_orig_df, ax=axes[2])
    [ax.set_xlabel("") for ax in axes]

    # Histograms for triangle area
    fig, axes = plt.subplots(2, 1, sharey=True, tight_layout=True)
    create_histogram(
        list_area, axes[0], xlabel="Triangle area mm^2", title="Pitch origin measurements"
    )
    create_histogram(list_pitch2yaw, axes[1], xlabel="m", title="pitch2yaw measurements")

    # fig, axes = plt.subplots(1, 3)
    # axes[0].set_title(f"(mean={pitch_axis_df['px'].mean():+0.04f})")
    # axes[1].set_title(f"pitch_axis (mean={pitch_axis_df['py'].mean():+0.04f})")
    # axes[2].set_title(f"(mean={pitch_axis_df['pz'].mean():+0.04f})")
    # sns.boxplot(y=pitch_axis_df["px"] - pitch_axis_df["px"].mean(), ax=axes[0])
    # sns.boxplot(y=pitch_axis_df["py"] - pitch_axis_df["py"].mean(), ax=axes[1])
    # sns.boxplot(y=pitch_axis_df["pz"] - pitch_axis_df["pz"].mean(), ax=axes[2])

    plotter = Plotter3D()
    # plotter.scatter_3d(fiducial_Y.T)
    plotter.scatter_3d(pitch_axis.T)
    plotter.scatter_3d(roll_axis.T)
    plt.show()


# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
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
    # main()
    plot_results()
