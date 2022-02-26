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

# ROS and DVRK imports
import dvrk
import rospy

# kincalib module imports
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.ExperimentUtils import separate_markerandfiducial, calculate_midpoints
from kincalib.geometry import Line3D, Circle3D, Triangle3D
import kincalib.utils.CmnUtils as utils

np.set_printoptions(precision=4, suppress=True, sign=" ")


def load_files(root: Path):
    dict_files = defaultdict(dict)  # Use step as keys. Each entry has a pitch and roll file.
    for f in (root / "pitch_roll_mov").glob("*"):
        step = int(re.findall("step[0-9]+", f.name)[0][4:])  # Not very reliable
        if "pitch" in f.name:
            dict_files[step]["pitch"] = pd.read_csv(f)
        elif "roll" in f.name:
            dict_files[step]["roll"] = pd.read_csv(f)

    return dict_files


def create_histogram(data):
    fig, axes = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axes.hist(data, bins=50, edgecolor="black", linewidth=1.2, density=False)
    # axes.hist(data, bins=50, range=(0, 100), edgecolor="black", linewidth=1.2, density=False)
    axes.grid()
    axes.set_xlabel("Triangle area mm^2")
    axes.set_ylabel("Frequency")
    axes.set_title(f"Pitch axis measurements. (N={data.shape[0]:02d})")
    # axes.set_xticks([i * 5 for i in range(110 // 5)])
    plt.show()


def main():

    # Important paths
    log_level = args.log
    log = Logger("pitch_exp_analize2", log_level=log_level).log
    root = Path(args.root)
    marker_file = Path("./share/custom_marker_id_112.json")
    regex = ""

    dict_files = load_files(root)
    keys = sorted(list(dict_files.keys()))
    x = 0
    list_area = []
    pitch_orig1 = []
    pitch_orig2 = []
    roll_axis1_M = []
    roll_axis2_M = []
    pitch_axis1_M = []
    pitch_axis2_M = []
    prev_p_ax = None
    prev_r_ax = None
    for k in keys:
        if len(list(dict_files[k].keys())) < 2:
            log.warning(f"files for step {k} are not available")
            continue

        try:

            intermediate_values = {}
            m1, m2, m3 = calculate_midpoints(
                dict_files[k]["roll"], dict_files[k]["pitch"], other_vals_dict=intermediate_values
            )
            # ---------------------------
            # Marker to pitch frame error metrics
            # - Calculate pitch origin in marker Frame
            # - Calculate pitch axes in marker Frame
            # - Calculate roll axes in marker Frame
            # ---------------------------
            pitch_ori_T = (m1 + m2 + m3) / 3
            # Marker2Tracker
            T_TM1 = utils.pykdl2frame(intermediate_values["marker_frame_pitch1"])
            pitch_ori_M1 = T_TM1.inv() @ pitch_ori_T
            pitch_orig1.append(pitch_ori_M1.squeeze())
            pitch_ax1 = T_TM1.inv().r @ intermediate_values["pitch_axis1"]
            pitch_axis1_M.append(pitch_ax1)
            roll_axis1 = T_TM1.inv().r @ intermediate_values["roll_axis"]
            roll_axis1_M.append(roll_axis1)

            T_TM2 = utils.pykdl2frame(intermediate_values["marker_frame_pitch2"])
            pitch_ori_M2 = T_TM2.inv() @ pitch_ori_T
            pitch_orig2.append(pitch_ori_M2.squeeze())
            pitch_ax2 = T_TM2.inv().r @ intermediate_values["pitch_axis2"]
            pitch_axis2_M.append(pitch_ax2)
            roll_axis2 = T_TM2.inv().r @ intermediate_values["roll_axis"]
            roll_axis2_M.append(roll_axis2)

            # Check if pitch axis are looking in opposite directions
            if np.dot(pitch_ax1, pitch_ax2) < 0:
                pitch_ax1 *= -1
            if prev_p_ax is None:
                prev_p_ax = pitch_ax1
            # Then make sure that all the vectors from all the steps in the trajectory point in the same direction
            else:
                if np.dot(prev_p_ax, pitch_ax1) < 0:
                    pitch_ax1 *= -1
                    pitch_ax2 *= -1
                prev_p_ax = pitch_ax1

            # Check if roll axis are looking in opposite directions
            if np.dot(roll_axis1, roll_axis2) < 0:
                roll_axis1 *= -1
            if prev_r_ax is None:
                prev_r_ax = roll_axis1
            # Then make sure that all the vectors from all the steps in the trajectory point in the same direction
            else:
                if np.dot(prev_r_ax, roll_axis1) < 0:
                    roll_axis1 *= -1
                    roll_axis2 *= -1
                prev_r_ax = roll_axis1

            # ---------------------------
            # Mid point error metrics
            # ---------------------------
            triangle = Triangle3D([m1, m2, m3])
            # Scale sides and area to milimiters
            sides = triangle.calculate_sides(scale=1000)
            area = triangle.calculate_area(scale=1000)
            # sides = calculate_triangle_sides(m1,m2,m3)
            # area = calculate_area(m1,m2,m3)
            # fmt: off
            log.debug(f"Step {k} results")
            log.debug(f"MID POINT RESULTS")
            log.debug(f"triangle sides {sides} mm")
            log.debug(f"triangle area {area:0.4f} mm^2")
            log.debug(f"MARKER TO PITCH RESULTS")
            log.debug(f"pitch origin1 from marker {pitch_ori_M1.squeeze()}")
            log.debug(f"pitch origin2 from marker {pitch_ori_M2.squeeze()}")
            log.debug(f"pitch axis1  from marker {pitch_ax1}")
            log.debug(f"pitch axis2  from marker {pitch_ax2}")
            log.debug( f"Is dot product betweeen pitch axis positive? {np.dot(pitch_ax1,pitch_ax2)>0}")
            log.debug(f"roll axis1  from marker {roll_axis1}")
            log.debug(f"roll axis2  from marker {roll_axis2}")
            log.debug( f"Is dot product betweeen roll axis positive? {np.dot(roll_axis1,roll_axis2)>0}")
            # fmt: on
            list_area.append(area)
        except Exception as e:
            log.error(f"Exception triggered in step {k}")
            log.debug(e)
            log.debug(traceback.print_exc())

    list_area = np.array(list_area)
    pitch_orig = np.array(pitch_orig1 + pitch_orig2)
    pitch_axis = np.array(pitch_axis1_M + pitch_axis2_M)
    roll_axis = np.array(roll_axis1_M + roll_axis2_M)
    # create_histogram(list_area)
    log.info(f"Mean area {list_area.mean():0.4f}")
    log.info(f"Std  area {list_area.std():0.4f}")
    log.info(f"Mean pitch_orig {pitch_orig.mean(axis=0)}")
    log.info(f"Std  pitch_orig {pitch_orig.std(axis=0)}")
    log.info(f"Mean pitch_axis {pitch_axis.mean(axis=0)}")
    log.info(f"Std  pitch_axis {pitch_axis.std(axis=0)}")
    log.info(f"Mean roll_axis {roll_axis.mean(axis=0)}")
    log.info(f"Std  roll_axis {roll_axis.std(axis=0)}")

    f_path = Path(__file__).parent / "results" / root.name
    log.info(f_path)
    np.save(f_path, list_area)


def plot_results():
    root = Path(args.root)
    f_path = Path(__file__).parent / "results" / root.name
    f_path = f_path.with_suffix(".npy")
    results = np.load(f_path)

    create_histogram(results)


# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
# fmt:off
parser.add_argument( "-r", "--root", type=str, default="./data/03_replay_trajectory/d04-rec-04", 
                        help="root dir") 
parser.add_argument( "-l", "--log", type=str, default="DEBUG", 
                        help="log level") #fmt:on
args = parser.parse_args()

if __name__ == "__main__":
    main()
    plot_results()
