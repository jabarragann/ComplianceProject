# Python imports
import json
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
import tf_conversions.posemath as pm

# kincalib module imports
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.Frame import Frame
from kincalib.Calibration.CalibrationUtils import CalibrationUtils, JointEstimator
from kincalib.utils.ExperimentUtils import separate_markerandfiducial
from kincalib.utils.CmnUtils import *

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
        rq1, rq2, rq3, rq4, rq5, rq6 = robot_jp.iloc[idx].loc[["q1", "q2", "q3", "q4", "q5", "q6"]].to_numpy()
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
        if wrist_fiducials.shape[0] == 3:
            log.info(f"Step {step} exec q5,q6 time {e_time-s_time:0.04f}")
            tq4, tq5, tq6 = joints456_calculation(*pitch_orig_R.squeeze())
        else:
            tq4, tq5, tq6 = None, None, None

        # Save data
        d = np.array([step, rq1, rq2, rq3, rq4, rq5, rq6]).reshape(1, -1)
        new_pt = pd.DataFrame(d, columns=cols_robot)
        df_robot = df_robot.append(new_pt)

        d = np.array([step, tq1, tq2, tq3, tq4, tq5, tq6]).reshape(1, -1)
        new_pt = pd.DataFrame(d, columns=cols_tracker)
        df_tracker = df_tracker.append(new_pt)

    return df_robot, df_tracker


def obtain_true_joints_v2(estimator: JointEstimator, robot_jp: pd.DataFrame, robot_cp: pd.DataFrame) -> pd.DataFrame:

    cols_robot = ["step", "rq1", "rq2", "rq3", "rq4", "rq5", "rq6"]
    cols_tracker = ["step", "tq1", "tq2", "tq3", "tq4", "tq5", "tq6"]
    cols_opt = ["step", "opt"]

    df_robot = pd.DataFrame(columns=cols_robot)
    df_tracker = pd.DataFrame(columns=cols_tracker)
    df_opt = pd.DataFrame(columns=cols_opt)

    opt_error = []
    # for idx in range(robot_jp.shape[0]):
    for idx in track(range(robot_jp.shape[0]), "ground truth calculation"):
        # Read robot joints
        rq1, rq2, rq3, rq4, rq5, rq6 = robot_jp.iloc[idx].loc[["q1", "q2", "q3", "q4", "q5", "q6"]].to_numpy()
        step = robot_jp.iloc[idx].loc["step"]

        # Read tracker data
        df_temp = robot_cp[robot_cp["step"] == step]
        marker_file = Path("./share/custom_marker_id_112.json")
        pose_arr, wrist_fiducials = separate_markerandfiducial(None, marker_file, df=df_temp)
        if len(pose_arr) == 0:
            log.warning(f"Marker pose in {step} not found")
            continue
        assert len(pose_arr) == 1, "There should only be one marker pose at each step."
        T_TM = Frame.init_from_matrix(pm.toMatrix(pose_arr[0]))

        # Calculate q1, q2 and q3
        tq1, tq2, tq3 = estimator.estimate_q123(T_TM)

        # Calculate joint 4
        tq4 = estimator.estimate_q4(tq1, tq2, tq3, T_TM.inv())

        # Calculate joints 5,6
        tq5, tq6, evaluation = estimator.estimate_q56(T_TM.inv(), wrist_fiducials.squeeze())
        d = np.array([step, evaluation]).reshape(1, -1)
        new_pt = pd.DataFrame(d, columns=cols_opt)
        df_opt = df_opt.append(new_pt)
        # opt_error.append(evaluation)

        # Save data
        d = np.array([step, rq1, rq2, rq3, rq4, rq5, rq6]).reshape(1, -1)
        new_pt = pd.DataFrame(d, columns=cols_robot)
        df_robot = df_robot.append(new_pt)

        d = np.array([step, tq1, tq2, tq3, tq4, tq5, tq6]).reshape(1, -1)
        new_pt = pd.DataFrame(d, columns=cols_tracker)
        df_tracker = df_tracker.append(new_pt)

        # if step > 400:
        #     break

    return df_robot, df_tracker, df_opt  # opt_error


def plot_joints(robot_df, tracker_df):
    fig, axes = plt.subplots(6, 1, sharex=True)
    robot_df.reset_index(inplace=True)
    tracker_df.reset_index(inplace=True)
    for i in range(6):
        # fmt:off
        axes[i].set_title(f"joint {i+1}")
        axes[i].plot(robot_df['step'].to_numpy(), robot_df[f"rq{i+1}"].to_numpy(), color="blue")
        axes[i].plot(robot_df['step'].to_numpy(), robot_df[f"rq{i+1}"].to_numpy(), marker="*", linestyle="None", color="blue", label="robot")
        axes[i].plot(robot_df['step'].to_numpy(),tracker_df[f"tq{i+1}"].to_numpy(), color="orange")
        axes[i].plot(robot_df['step'].to_numpy(),tracker_df[f"tq{i+1}"].to_numpy(), marker="*", linestyle="None", color="orange", label="tracker")
        # fmt:on
    axes[0].legend()


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
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    # Important paths
    root = Path(args.root)
    marker_file = Path("./share/custom_marker_id_112.json")

    if args.test:
        robot_jp_p = root / f"test_trajectories/{args.trajid:02d}" / "test_traj_jp.txt"
        robot_cp_p = root / f"test_trajectories/{args.trajid:02d}" / "test_traj_cp.txt"
    else:
        robot_jp_p = root / "robot_mov" / "robot_jp_temp.txt"
        robot_cp_p = root / "robot_mov" / "robot_cp_temp.txt"

    dst_p = robot_cp_p.parent / "result"
    if (robot_cp_p.parent / "result" / "robot_joints.txt").exists() and not args.reset:
        # ------------------------------------------------------------
        # Read calculated joint values
        # ------------------------------------------------------------
        robot_df = pd.read_csv(dst_p / "robot_joints.txt", index_col=None)
        tracker_df = pd.read_csv(dst_p / "tracker_joints.txt", index_col=None)
        opt_df = pd.read_csv(dst_p / "opt_error.txt", index_col=None)
    else:
        # ------------------------------------------------------------
        # Calculate joints
        # ------------------------------------------------------------
        # Read data
        robot_jp = pd.read_csv(robot_jp_p)
        robot_cp = pd.read_csv(robot_cp_p)
        registration_data_path = root / "registration_results"
        registration_dict = json.load(open(registration_data_path / "registration_values.json", "r"))
        # calculate joints
        T_TR = Frame.init_from_matrix(np.array(registration_dict["robot2tracker_T"]))
        T_RT = T_TR.inv()
        T_MP = Frame.init_from_matrix(np.array(registration_dict["pitch2marker_T"]))
        T_PM = T_MP.inv()
        wrist_fid_Y = np.array(registration_dict["fiducial_in_jaw"])

        if not (registration_data_path / "registration_data.txt").exists():
            log.error("Missing registration data file")
        if not (registration_data_path / "robot2tracker_t.npy").exists():
            log.error("Missing robot tracker transformation")

        reg_data = pd.read_csv(registration_data_path / "registration_data.txt")

        # Calculate ground truth joints
        # Version1
        # robot_df, tracker_df = obtain_true_joints(reg_data, robot_jp, robot_cp, T_TR)
        # Version2
        j_estimator = JointEstimator(T_RT, T_MP, wrist_fid_Y)
        robot_df, tracker_df, opt_df = obtain_true_joints_v2(j_estimator, robot_jp, robot_cp)

        # Save data
        if not dst_p.exists():
            dst_p.mkdir(parents=True)
        robot_df.to_csv(dst_p / "robot_joints.txt", index=False)
        tracker_df.to_csv(dst_p / "tracker_joints.txt", index=False)
        opt_df.to_csv(dst_p / "opt_error.txt", index=False)

    # Calculate stats
    valid_steps = opt_df.loc[opt_df["opt"] < 0.001]["step"]
    robot_valid = robot_df.loc[opt_df["step"].isin(valid_steps)].iloc[:, 1:].to_numpy()
    tracker_valid = tracker_df.loc[opt_df["step"].isin(valid_steps)].iloc[:, 1:].to_numpy()
    diff = robot_valid - tracker_valid
    diff_mean = diff.mean(axis=0)
    diff_std = diff.std(axis=0)

    log.info(f"Joint 1 mean difference (deg): {mean_std_str(diff_mean[0]*180/np.pi,diff_std[0]*180/np.pi)}")
    log.info(f"Joint 2 mean difference (deg): {mean_std_str(diff_mean[1]*180/np.pi,diff_std[1]*180/np.pi)}")
    log.info(f"Joint 3 mean difference (m):   {mean_std_str(diff_mean[2],diff_std[2])}")
    log.info(f"Joint 4 mean difference (deg): {mean_std_str(diff_mean[3]*180/np.pi,diff_std[3]*180/np.pi)}")
    log.info(f"Joint 5 mean difference (deg): {mean_std_str(diff_mean[4]*180/np.pi,diff_std[4]*180/np.pi)}")
    log.info(f"Joint 6 mean difference (deg): {mean_std_str(diff_mean[5]*180/np.pi,diff_std[5]*180/np.pi)}")

    # plot
    plot_joints(robot_df, tracker_df)
    fig, ax = plt.subplots(1, 1)
    create_histogram(
        opt_df["opt"], axes=ax, title=f"Optimization error", xlabel="objective function at optimal solution"
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # fmt:off
    parser.add_argument( "-r", "--root", type=str, default="./data/03_replay_trajectory/d04-rec-07-traj01", 
                    help="root dir") 
    parser.add_argument('-t','--test', action='store_true',help="Use test data")
    parser.add_argument("--trajid", type=int, help="The id of the test trajectory to use") 
    parser.add_argument( "-l", "--log", type=str, default="DEBUG", 
                    help="log level") #fmt:on
    parser.add_argument('--reset', action='store_true',help="Recalculate joint values")
    args = parser.parse_args()
    log_level = args.log
    log = Logger("pitch_exp_analize2", log_level=log_level).log

    main()
