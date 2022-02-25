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
from collections import defaultdict
import re
import json

# ROS and DVRK imports
import dvrk
import rospy
from cisst_msgs.srv import QueryForwardKinematics
from sensor_msgs.msg import JointState
import tf_conversions.posemath as pm

# kincalib module imports
from kincalib.utils.Frame import Frame
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.ExperimentUtils import separate_markerandfiducial, calculate_midpoints
from kincalib.geometry import Line3D, Circle3D, Triangle3D
import kincalib.utils.CmnUtils as utils

np.set_printoptions(precision=4, suppress=True, sign=" ")


log = Logger("registration").log


def calculate_registration(df: pd.DataFrame, root: Path):
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    # filename = Path("scripts/01_calibration_exp/results")
    # root = Path(args.root)
    # filename = filename / (root.name + "_registration_data.txt")
    # df = pd.read_csv(filename)

    # Choose entries were the area is lower is than 6mm
    df = df.loc[df["area"] < 10]
    robot_p = df[["rpx", "rpy", "rpz"]].to_numpy().T
    tracker_p = df[["tpx", "tpy", "tpz"]].to_numpy().T
    log.info(f"Points used for the registration {df.shape}")

    trans = Frame.find_transformation_direct(robot_p, tracker_p)
    error = Frame.evaluation(robot_p, tracker_p, trans)

    log.info(f"transformation between robot and tracker\n{trans}")
    log.info(f"mean square error: {1000*error:0.05f}")

    dst_f = root / "registration_results/robot2tracker_t.npy"

    np.save(dst_f, trans)
    return trans


def calculate_pitch_to_marker(registration_data, other_values_dict=None):
    # calculate pitch orig
    pitch_orig = registration_data[["mox", "moy", "moz"]].to_numpy()
    pitch_orig_mean = pitch_orig.mean(axis=0)
    pitch_orig_std = pitch_orig.std(axis=0)
    # calculate pitch axis
    pitch_axis = registration_data[["mpx", "mpy", "mpz"]].to_numpy()
    pitch_axis_mean = pitch_axis.mean(axis=0)
    pitch_axis_std = pitch_axis.std(axis=0)
    # calculate roll axis
    roll_axis = registration_data[["mrx", "mry", "mrz"]].to_numpy()
    roll_axis_mean = roll_axis.mean(axis=0)
    roll_axis_std = roll_axis.std(axis=0)

    other_values_dict["pitchaxis"] = pitch_axis_mean
    other_values_dict["rollaxis"] = roll_axis_mean
    other_values_dict["pitchorigin"] = pitch_orig_mean

    pitch_y_axis = np.cross(roll_axis_mean, pitch_axis_mean)
    # calculate transformation
    pitch2marker_T = np.identity(4)
    pitch2marker_T[:3, 0] = pitch_axis_mean
    pitch2marker_T[:3, 1] = pitch_y_axis
    pitch2marker_T[:3, 2] = roll_axis_mean
    pitch2marker_T[:3, 3] = pitch_orig_mean
    return Frame.init_from_matrix(pitch2marker_T)


def pitch_orig_in_robot(
    robot_df: pd.DataFrame, service_name: str = "/PSM1/local/query_cp"
) -> pd.DataFrame:
    # ------------------------------------------------------------
    # Connect to DVRK fk service
    # ------------------------------------------------------------
    rospy.wait_for_service(service_name, timeout=1)
    kinematics_service = rospy.ServiceProxy(service_name, QueryForwardKinematics)
    log.info("connected to {:} ...".format(service_name))

    # ------------------------------------------------------------
    # Itereate over date and calculate pitch origin in robot coordinates
    # ------------------------------------------------------------
    cols = ["step", "rpx", "rpy", "rpz"]
    df_results = pd.DataFrame(columns=cols)
    for idx in range(robot_df.shape[0]):
        joints = JointState()
        joints.position = robot_df.iloc[idx][["q1", "q2", "q3", "q4"]].to_list()
        msg = kinematics_service(joints)
        msg = pm.fromMsg(msg.cp.pose).p
        data = [robot_df.iloc[idx]["step"]] + list(msg)
        data = np.array(data).reshape((1, -1))
        new_df = pd.DataFrame(data, columns=cols)
        df_results = df_results.append(new_df)

    return df_results


def pitch_orig_in_tracker(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate the pitch origin in tracker coordinates and the pitch in frame in marker
    coordinates. There exists a rigid transformation between the marker in the shaft and
    the pitch frame.

    Args:
        root (Path): data path

    Returns:
        Tuple[pd.DataFrame,pd.DataFrame]: dataframe with pitch origin w.r.t tracker and dataframe
        pitch frame w.r.t marker frame
    """

    def load_files(root: Path):
        dict_files = defaultdict(dict)  # Use step as keys. Each entry has a pitch and roll file.
        for f in (root / "pitch_roll_mov").glob("*"):
            step = int(re.findall("step[0-9]+", f.name)[0][4:])  # NOT very reliable
            if "pitch" in f.name:
                dict_files[step]["pitch"] = pd.read_csv(f)
            elif "roll" in f.name:
                dict_files[step]["roll"] = pd.read_csv(f)

        return dict_files

    # ------------------------------------------------------------
    # Iterate over data and calculate pitch origin in tracker coordinates
    # ------------------------------------------------------------
    # Final df
    cols = ["step", "area", "tpx", "tpy", "tpz"]
    cols_pitch_M = ["step", "mox", "moy", "moz", "mpx", "mpy", "mpz", "mrx", "mry", "mrz"]
    df_results = pd.DataFrame(columns=cols)
    df_pitch_axes = pd.DataFrame(columns=cols_pitch_M)

    dict_files = load_files(root)  # Use the step as the dictionary key
    keys = sorted(list(dict_files.keys()))

    prev_p_ax = None
    prev_r_ax = None
    for k in track(keys, "Computing pitch origin in tracker coordinates"):
        if len(list(dict_files[k].keys())) < 2:
            log.warning(f"files for step {k} are not available")
            continue
        step = k
        intermediate_values = {}
        m1, m2, m3 = calculate_midpoints(
            dict_files[k]["roll"], dict_files[k]["pitch"], other_vals_dict=intermediate_values
        )
        triangle = Triangle3D([m1, m2, m3])
        # Scale area to milimiters
        area = triangle.calculate_area(scale=1000)
        center = triangle.calculate_centroid()
        # Add new entry to df
        data = [step, area] + list(center)
        data = np.array(data).reshape((1, -1))
        new_df = pd.DataFrame(data, columns=cols)
        df_results = df_results.append(new_df)

        # Calculate pitch axis in Marker frame
        pitch_org_M, pitch_ax_M, roll_ax_M = calculate_axes_in_marker(
            center, intermediate_values, prev_p_ax, prev_r_ax
        )
        prev_p_ax = pitch_ax_M
        prev_r_ax = roll_ax_M
        data = [step] + list(pitch_org_M) + list(pitch_ax_M) + list(roll_ax_M)
        data = np.array(data).reshape((1, -1))
        new_pt = pd.DataFrame(data, columns=cols_pitch_M)
        df_pitch_axes = df_pitch_axes.append(new_pt)

    return df_results, df_pitch_axes


def calculate_axes_in_marker(pitch_ori_T, intermediate_values: dict, prev_p_ax, prev_r_ax):
    # Marker2Tracker
    T_TM1 = utils.pykdl2frame(intermediate_values["marker_frame_pitch1"])
    pitch_ori1 = (T_TM1.inv() @ pitch_ori_T).squeeze()
    pitch_ax1 = T_TM1.inv().r @ intermediate_values["pitch_axis1"]
    roll_ax1 = T_TM1.inv().r @ intermediate_values["roll_axis"]

    T_TM2 = utils.pykdl2frame(intermediate_values["marker_frame_pitch2"])
    pitch_ori2 = (T_TM2.inv() @ pitch_ori_T).squeeze()
    pitch_ax2 = T_TM2.inv().r @ intermediate_values["pitch_axis2"]
    roll_ax2 = T_TM2.inv().r @ intermediate_values["roll_axis"]

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
    if np.dot(roll_ax1, roll_ax2) < 0:
        roll_ax1 *= -1
    if prev_r_ax is None:
        prev_r_ax = roll_ax1
    # Then make sure that all the vectors from all the steps in the trajectory point in the same direction
    else:
        if np.dot(prev_r_ax, roll_ax1) < 0:
            roll_ax1 *= -1
            roll_ax2 *= -1
        prev_r_ax = roll_ax1

    return (pitch_ori1 + pitch_ori2) / 2, (pitch_ax1 + pitch_ax2) / 2, (roll_ax1 + roll_ax2) / 2


def obtain_registration_data(root: Path):
    # Robot points
    robot_jp = root / "robot_mov" / "robot_jp_temp.txt"
    robot_jp = pd.read_csv(robot_jp)
    pitch_robot = pitch_orig_in_robot(robot_jp)  # df with columns = [step,px,py,pz]

    # Tracker points
    # df with columns = [step area tx, ty,tz]
    pitch_tracker, pitch_marker = pitch_orig_in_tracker(root)

    # combine df using step as index
    pitch_df = pd.merge(pitch_tracker, pitch_marker, on="step")
    final_df = pd.merge(pitch_robot, pitch_df, on="step")

    # Save df
    dst_f = root / "registration_results"
    if not dst_f.exists():
        dst_f.mkdir(parents=True)

    dst_f = dst_f / "registration_data.txt"
    final_df.to_csv(dst_f, index=None)

    return final_df


def main():
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    # Important paths
    log_level = args.log
    log = Logger("pitch_exp_analize2", log_level=log_level).log
    root = Path(args.root)
    marker_file = Path("./share/custom_marker_id_112.json")
    regex = ""

    registration_data_path = root / "registration_results/registration_data.txt"
    if registration_data_path.exists():
        log.info("Loading registration data ...")
        registration_data = pd.read_csv(registration_data_path)
    else:
        log.info("Extracting registration data")
        registration_data = obtain_registration_data(root)

    # Calculate registration
    robot2tracker_t = calculate_registration(registration_data, root)
    # Calculate marker to pitch transformation
    axis_dict = {}
    pitch2marker_t = calculate_pitch_to_marker(registration_data, other_values_dict=axis_dict)

    # Save everything as a JSON file.
    json_data = {}
    json_data["robot2tracker_T"] = np.array(robot2tracker_t).tolist()
    json_data["pitch2marker_T"] = np.array(pitch2marker_t).tolist()
    json_data["pitchorigin_in_marker"] = axis_dict["pitchorigin"].tolist()
    json_data["pitchaxis_in_marker"] = axis_dict["pitchaxis"].tolist()
    json_data["rollaxis_in_marker"] = axis_dict["rollaxis"].tolist()

    new_name = registration_data_path.parent / "registration_values.json"
    with open(new_name, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    # Load json
    registration_dict = json.load(open(new_name, "r"))
    T_RT = Frame.init_from_matrix(np.array(registration_dict["robot2tracker_T"]))
    T_MP = Frame.init_from_matrix(np.array(registration_dict["pitch2marker_T"]))
    log.info(f"Robot to Tracker\n{T_RT}")
    log.info(f"Pitch to marker\n{T_MP}")


parser = argparse.ArgumentParser()
# fmt:off
parser.add_argument( "-r", "--root", type=str, default="./data/03_replay_trajectory/d04-rec-02", 
                help="root dir") 
parser.add_argument( "-l", "--log", type=str, default="DEBUG", 
                help="log level") #fmt:on
args = parser.parse_args()

if __name__ == "__main__":
    main()
