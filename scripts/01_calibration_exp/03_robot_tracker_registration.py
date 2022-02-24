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


def pitch_orig_in_tracker(root: Path):
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
    # Itereate over data and calculate pitch origin in tracker coordinates
    # ------------------------------------------------------------
    # Final df
    cols = ["step", "area", "tpx", "tpy", "tpz"]
    df_results = pd.DataFrame(columns=cols)

    dict_files = load_files(root)  # Use the step as the dictionary key
    keys = sorted(list(dict_files.keys()))

    for k in track(keys, "Computing pitch origin in tracker coordinates"):
        if len(list(dict_files[k].keys())) < 2:
            log.warning(f"files for step {k} are not available")
            continue
        step = k
        m1, m2, m3 = calculate_midpoints(dict_files[k]["roll"], dict_files[k]["pitch"])
        triangle = Triangle3D([m1, m2, m3])
        # Scale sides and area to milimiters
        area = triangle.calculate_area(scale=1000)
        center = triangle.calculate_centroid()
        # Add new entry to df
        data = [step, area] + list(center)
        data = np.array(data).reshape((1, -1))
        new_df = pd.DataFrame(data, columns=cols)
        df_results = df_results.append(new_df)

    return df_results


def obtain_registration_data(root: Path):
    # Robot points
    robot_jp = root / "robot_mov" / "robot_jp_temp.txt"
    robot_jp = pd.read_csv(robot_jp)
    pitch_robot = pitch_orig_in_robot(robot_jp)  # df with columns = [step,px,py,pz]

    # Tracker points
    pitch_tracker = pitch_orig_in_tracker(root)  # df with columns = [step area tx, ty,tz]

    # combine df using step as index
    final_df = pd.merge(pitch_robot, pitch_tracker, on="step")

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


parser = argparse.ArgumentParser()
# fmt:off
parser.add_argument( "-r", "--root", type=str, default="./data/03_replay_trajectory/d04-rec-02", 
                help="root dir") 
parser.add_argument( "-l", "--log", type=str, default="DEBUG", 
                help="log level") #fmt:on
args = parser.parse_args()

if __name__ == "__main__":

    main()
