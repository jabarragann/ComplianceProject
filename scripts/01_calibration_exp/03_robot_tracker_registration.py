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
from kincalib.utils.ExperimentUtils import load_registration_data, calculate_midpoints
from kincalib.geometry import Line3D, Circle3D, Triangle3D
import kincalib.utils.CmnUtils as utils
from kincalib.Calibration.CalibrationUtils import CalibrationUtils as calib

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
    # calculate pitch orig in marker
    pitch_orig = registration_data[["mox", "moy", "moz"]].to_numpy()
    pitch_orig_mean = pitch_orig.mean(axis=0)
    pitch_orig_std = pitch_orig.std(axis=0)
    # calculate pitch axis in marker
    pitch_axis = registration_data[["mpx", "mpy", "mpz"]].to_numpy()
    pitch_axis_mean = pitch_axis.mean(axis=0)
    pitch_axis_std = pitch_axis.std(axis=0)
    # calculate roll axis in marker
    roll_axis = registration_data[["mrx", "mry", "mrz"]].to_numpy()
    roll_axis_mean = roll_axis.mean(axis=0)
    roll_axis_std = roll_axis.std(axis=0)
    # calculate fiducial in yaw
    fiducial_yaw = registration_data[["yfx", "yfy", "yfz"]].to_numpy()
    fiducial_yaw_mean = fiducial_yaw.mean(axis=0)
    fiducial_yaw_std = fiducial_yaw.std(axis=0)
    # Calculate pitch2yaw
    pitch2yaw = registration_data["pitch2yaw"].to_numpy()
    pitch2yaw_mean = pitch2yaw.mean()
    pitch2yaw_std = pitch2yaw.std()

    other_values_dict["pitchaxis"] = pitch_axis_mean
    other_values_dict["rollaxis"] = roll_axis_mean
    other_values_dict["pitchorigin"] = pitch_orig_mean
    other_values_dict["fiducial_yaw"] = fiducial_yaw_mean
    other_values_dict["pitch2yaw"] = pitch2yaw_mean

    # CRITICAL STEP OF THE IKIN CALCULATIONS
    # calculate transformation
    # ------------------------------------------------------------
    # Version2
    # pitch_axis aligned with y
    # roll_axis aligned with z
    # ------------------------------------------------------------
    pitch_x_axis = np.cross(pitch_axis_mean, roll_axis_mean)

    pitch2marker_T = np.identity(4)
    pitch2marker_T[:3, 0] = pitch_x_axis
    pitch2marker_T[:3, 1] = pitch_axis_mean
    pitch2marker_T[:3, 2] = roll_axis_mean
    pitch2marker_T[:3, 3] = pitch_orig_mean
    # ------------------------------------------------------------
    # Version1
    # pitch_axis aligned with z
    # roll_axis aligned with x
    # ------------------------------------------------------------
    # pitch_y_axis = np.cross(pitch_axis_mean, roll_axis_mean)

    # pitch2marker_T = np.identity(4)
    # pitch2marker_T[:3, 0] = roll_axis_mean
    # pitch2marker_T[:3, 1] = pitch_y_axis
    # pitch2marker_T[:3, 2] = pitch_axis_mean
    # pitch2marker_T[:3, 3] = pitch_orig_mean

    return Frame.init_from_matrix(pitch2marker_T)


def pitch_orig_in_robot(robot_df: pd.DataFrame, service_name: str = "/PSM1/local/query_cp") -> pd.DataFrame:
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

    # ------------------------------------------------------------
    # Iterate over data and calculate pitch origin in tracker coordinates
    # ------------------------------------------------------------
    # Final df
    cols = ["step", "area", "tpx", "tpy", "tpz"]  # pitch orig in tracker
    # mo -> pitch origin in marker
    # mp -> pitch axis in marker
    # mr -> roll axis in marker
    # yf -> wrist fiducial in yaw
    # fmt: off
    cols_pitch_M = [ "step", "mox", "moy", "moz", "mpx", "mpy", "mpz",
                        "mrx", "mry", "mrz", "yfx", "yfy", "yfz","tfx","tfy","tfz", "pitch2yaw" ]
    # fmt: on
    df_results = pd.DataFrame(columns=cols)
    df_pitch_axes = pd.DataFrame(columns=cols_pitch_M)

    dict_files = load_registration_data(root)  # Use the step as the dictionary key
    keys = sorted(list(dict_files.keys()))

    prev_p_ax = None
    prev_r_ax = None
    for step in track(keys, "Computing pitch origin in tracker coordinates"):
        if len(list(dict_files[step].keys())) < 2:
            log.warning(f"files for step {step} are not available")
            continue
        try:
            # Get roll circles
            roll_cir1, roll_cir2 = calib.create_roll_circles(dict_files[step]["roll"])
            # Get pitch and yaw circles
            pitch_yaw_circles = calib.create_yaw_pitch_circles(dict_files[step]["pitch"])
        except Exception as e:
            log.error(f"Error in circles creation on step {step}")
            log.error(e)
            continue

        # Calculate registration data
        m1, m2, m3 = calib.calculate_pitch_origin(roll_cir1, pitch_yaw_circles[0]["pitch"], pitch_yaw_circles[1]["pitch"])
        pitch_ori_T = (m1 + m2 + m3) / 3

        triangle = Triangle3D([m1, m2, m3])
        # Scale area to milimiters
        area = triangle.calculate_area(scale=1000)
        pitch_ori_T = triangle.calculate_centroid()
        # Add new entry to df
        data = [step, area] + list(pitch_ori_T)
        data = np.array(data).reshape((1, -1))
        new_df = pd.DataFrame(data, columns=cols)
        df_results = df_results.append(new_df)

        # Calculate pitch axis in Marker frame
        pitch_org_M, pitch_ax_M, roll_ax_M = calculate_axes_in_marker(
            pitch_ori_T, pitch_yaw_circles, roll_cir1, prev_p_ax, prev_r_ax
        )
        ## IMPORTANT NOTE
        # Pitch and roll axis are the normal vector of a 3D circle fitted to the tracker data.
        # Therefore, these normal vectors can have two possible directions. Previous pitch and roll axis
        # are used to ensure consistency in the selected direction for different points in the trajectory.
        prev_p_ax = pitch_ax_M
        prev_r_ax = roll_ax_M

        # Estimate wrist fiducial in yaw origin
        # todo get the fiducial measurement from robot_cp_temp.
        # todo fiducial_y and pitch2yaw1 have two values each. You are only using 1.
        fiducial_Y, fiducial_T, pitch2yaw1 = calib.calculate_fiducial_from_yaw(pitch_ori_T, pitch_yaw_circles, roll_cir2)

        # Add to dataframe
        # fmt: off
        data = [step] + list(pitch_org_M) + list(pitch_ax_M) + list(roll_ax_M) + \
                list(fiducial_Y[1].squeeze())+ list(fiducial_T[1].squeeze()) +  [pitch2yaw1[1]]
        # fmt: on

        data = np.array(data).reshape((1, -1))
        new_pt = pd.DataFrame(data, columns=cols_pitch_M)
        df_pitch_axes = df_pitch_axes.append(new_pt)

    return df_results, df_pitch_axes


def calculate_axes_in_marker(pitch_ori_T, pitch_yaw_circles: dict, roll_circle, prev_p_ax, prev_r_ax):
    # Marker2Tracker
    T_TM1 = utils.pykdl2frame(pitch_yaw_circles[0]["marker_pose"])
    pitch_ori1 = (T_TM1.inv() @ pitch_ori_T).squeeze()
    pitch_ax1 = T_TM1.inv().r @ pitch_yaw_circles[0]["pitch"].normal
    roll_ax1 = T_TM1.inv().r @ roll_circle.normal

    T_TM2 = utils.pykdl2frame(pitch_yaw_circles[1]["marker_pose"])
    pitch_ori2 = (T_TM2.inv() @ pitch_ori_T).squeeze()
    pitch_ax2 = T_TM2.inv().r @ pitch_yaw_circles[1]["pitch"].normal
    roll_ax2 = T_TM2.inv().r @ roll_circle.normal

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

    if final_df.shape[0] == 0:
        raise Exception("No data found for registration")

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

    # Obtain registration data
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
    json_data["fiducial_in_jaw"] = axis_dict["fiducial_yaw"].tolist()
    json_data["pitch2yaw"] = axis_dict["pitch2yaw"].tolist()

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
parser.add_argument( "-r", "--root", type=str, default="./data/03_replay_trajectory/d04-rec-06-traj01", 
                help="root dir") 
parser.add_argument( "-l", "--log", type=str, default="DEBUG", 
                help="log level") #fmt:on
args = parser.parse_args()

if __name__ == "__main__":
    main()
