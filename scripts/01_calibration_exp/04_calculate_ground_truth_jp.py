# Python imports
from pathlib import Path
from re import I
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
from kincalib.utils.Frame import Frame
from kincalib.Calibration.CalibrationUtils import CalibrationUtils as calib
from kincalib.utils.ExperimentUtils import separate_markerandfiducial

np.set_printoptions(precision=4, suppress=True, sign=" ")


def joints123_calculation(px, py, pz):
    L1 = -0.4318
    Ltool = 0.4162

    tq1 = np.arctan2(px, -pz)
    tq2 = np.arctan2(-py, np.sqrt(px ** 2 + pz ** 2))
    tq3 = np.sqrt(px ** 2 + py ** 2 + pz ** 2) + L1 + Ltool
    return tq1, tq2, tq3


def joints456_calculation(px, py, pz):
    return px, py, pz


def obtain_true_joints(
    reg_data: pd.DataFrame, robot_jp: pd.DataFrame, robot_cp: pd.DataFrame, T_TR: Frame
) -> pd.DataFrame:

    T_RT = T_TR.inv()
    cols_robot = ["step", "rq1", "rq2", "rq3", "rq4", "rq5", "rq6"]
    cols_tracker = ["step", "tq1", "tq2", "tq3", "tq4", "tq5", "tq6"]
    df_robot = pd.DataFrame(columns=cols_robot)
    df_tracker = pd.DataFrame(columns=cols_tracker)

    # for idx in range(robot_jp.shape[0]):
    for idx in track(range(robot_jp.shape[0]), "ground truth calculation"):
        rq1, rq2, rq3, rq4, rq5, rq6 = (
            robot_jp.iloc[idx].loc[["q1", "q2", "q3", "q4", "q5", "q6"]].to_numpy()
        )
        step = robot_jp.iloc[idx].loc["step"]
        # Get pitch origin in tracker
        pitch_orig_T = reg_data.loc[reg_data["step"] == step][["tpx", "tpy", "tpz"]].to_numpy()
        pitch_orig_T = pitch_orig_T.squeeze()
        if pitch_orig_T.shape[0] < 3:
            log.warning(f"pitch_orig for step {step} not found")
            continue
        # Convert to R with registration T
        pitch_orig_R = T_RT @ pitch_orig_T

        # Calculate joints 1,2,3
        tq1, tq2, tq3 = joints123_calculation(*pitch_orig_R.squeeze())

        # Calculate joints 4,5,6
        s_time = time.time()
        df_temp = robot_cp[robot_cp["step"] == step]
        marker_file = Path("./share/custom_marker_id_112.json")
        pose_arr, wrist_fiducials = separate_markerandfiducial(None, marker_file, df=df_temp)
        e_time = time.time()
        log.info(f"Step {step} exec q5,q6 time {e_time-s_time:0.04f}")
        tq4, tq5, tq6 = joints456_calculation(*pitch_orig_R.squeeze())

        # Save data
        d = np.array([step, rq1, rq2, rq3, rq4, rq5, rq6]).reshape(1, -1)
        new_pt = pd.DataFrame(d, columns=cols_robot)
        df_robot = df_robot.append(new_pt)

        d = np.array([step, tq1, tq2, tq3, tq4, tq5, tq6]).reshape(1, -1)
        new_pt = pd.DataFrame(d, columns=cols_tracker)
        df_tracker = df_tracker.append(new_pt)

    return df_robot, df_tracker


def plot_joints(robot_df, tracker_df):
    fig, axes = plt.subplots(6, 1, sharex=True)
    robot_df.reset_index(inplace=True)
    tracker_df.reset_index(inplace=True)
    for i in range(6):
        # fmt:off
        axes[i].set_title(f"joint {i+1}")
        axes[i].plot(robot_df[f"rq{i+1}"], color="blue")
        axes[i].plot(robot_df[f"rq{i+1}"], marker="*", linestyle="None", color="blue", label="robot")
        axes[i].plot(tracker_df[f"tq{i+1}"], color="orange")
        axes[i].plot(tracker_df[f"tq{i+1}"], marker="*", linestyle="None", color="orange", label="tracker")
        # fmt:on
    axes[0].legend()
    plt.show()


def main():
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    # Important paths
    root = Path(args.root)
    marker_file = Path("./share/custom_marker_id_112.json")

    robot_jp = root / "robot_mov" / "robot_jp_temp.txt"
    robot_jp = pd.read_csv(robot_jp)
    robot_cp = root / "robot_mov" / "robot_cp_temp.txt"
    robot_cp = pd.read_csv(robot_cp)
    registration_data_path = root / "registration_results"

    if not (registration_data_path / "registration_data.txt").exists():
        log.error("Missing registration data file")
    if not (registration_data_path / "robot2tracker_t.npy").exists():
        log.error("Missing robot tracker transformation")

    reg_data = pd.read_csv(registration_data_path / "registration_data.txt")
    T_TR = np.load(registration_data_path / "robot2tracker_t.npy")
    T_TR = Frame(T_TR[:3, :3], T_TR[:3, 3])

    # Calculate ground truth joints
    robot_df, tracker_df = obtain_true_joints(reg_data, robot_jp, robot_cp, T_TR)

    # plot
    plot_joints(robot_df, tracker_df)

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # fmt:off
    parser.add_argument( "-r", "--root", type=str, default="./data/03_replay_trajectory/d04-rec-06-traj01", 
                    help="root dir") 
    parser.add_argument( "-l", "--log", type=str, default="DEBUG", 
                    help="log level") #fmt:on
    args = parser.parse_args()
    log_level = args.log
    log = Logger("pitch_exp_analize2", log_level=log_level).log

    main()
