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
from kincalib.Motion.DvrkKin import DvrkPsmKin

np.set_printoptions(precision=4, suppress=True, sign=" ")


def main(testid: int):
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    # Important paths
    root = Path(args.root)
    marker_file = Path("./share/custom_marker_id_112.json")

    joints = ["pitch", "roll", "yaw"]
    for j in joints:
        robot_jp_p = root / f"hysteresis_curves/{testid:02d}" / f"{j}_hysteresis_jp.txt"
        robot_cp_p = root / f"hysteresis_curves/{testid:02d}" / f"{j}_hysteresis_cp.txt"

        # ------------------------------------------------------------
        # Calculate tracker joint values
        # ------------------------------------------------------------
        dst_p = robot_cp_p.parent
        dst_p = dst_p / f"{j}_result"
        registration_data_path = root / "registration_results"

        log.info(f"Storing results in {dst_p}")
        log.info(f"Loading registration .json from {registration_data_path}")
        dst_p.mkdir(parents=True, exist_ok=True)

        if (dst_p / "robot_joints.txt").exists() and not args.reset:
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
            registration_dict = json.load(open(registration_data_path / "registration_values.json", "r"))
            # calculate joints
            T_TR = Frame.init_from_matrix(np.array(registration_dict["robot2tracker_T"]))
            T_RT = T_TR.inv()
            T_MP = Frame.init_from_matrix(np.array(registration_dict["pitch2marker_T"]))
            T_PM = T_MP.inv()
            wrist_fid_Y = np.array(registration_dict["fiducial_in_jaw"])

            # Calculate ground truth joints
            j_estimator = JointEstimator(T_RT, T_MP, wrist_fid_Y)
            robot_df, tracker_df, opt_df = CalibrationUtils.obtain_true_joints_v2(j_estimator, robot_jp, robot_cp)

            # Save data
            robot_df.to_csv(dst_p / "robot_joints.txt", index=False)
            tracker_df.to_csv(dst_p / "tracker_joints.txt", index=False)
            opt_df.to_csv(dst_p / "opt_error.txt", index=False)

        # ------------------------------------------------------------
        # Evaluate results
        # ------------------------------------------------------------

        # Calculate stats
        valid_steps = opt_df.loc[opt_df["q56res"] < 0.005]["step"]  # Use the residual to get valid steps.
        robot_valid = robot_df.loc[opt_df["step"].isin(valid_steps)].iloc[:, 1:].to_numpy()
        tracker_valid = tracker_df.loc[opt_df["step"].isin(valid_steps)].iloc[:, 1:].to_numpy()
        diff = robot_valid - tracker_valid
        diff_mean = diff.mean(axis=0)
        diff_std = diff.std(axis=0)

        # Calculate cartesian errors calculate cartesian positions from robot_valid and tracker_valid
        robot_cp = CalibrationUtils.calculate_cartesian(robot_valid)
        tracker_cp = CalibrationUtils.calculate_cartesian(tracker_valid)
        cp_error = tracker_cp - robot_cp
        mean_error = cp_error.apply(np.linalg.norm, 1)

        log.info(f"Number of valid samples: {diff.shape[0]}")
        log.info(f"Cartesian results")
        log.info(f"X mean error (mm):   {mean_std_str(cp_error['X'].mean()*1000, cp_error['X'].std()*1000)}")
        log.info(f"Y mean error (mm):   {mean_std_str(cp_error['Y'].mean()*1000, cp_error['Y'].std()*1000)}")
        log.info(f"Z mean error (mm):   {mean_std_str(cp_error['Z'].mean()*1000, cp_error['Z'].std()*1000)}")
        log.info(f"Mean error   (mm):   {mean_std_str(mean_error.mean()*1000, mean_error.std()*1000)}")

        log.info(f"Results in degrees")
        log.info(f"Joint 1 mean difference (deg): {mean_std_str(diff_mean[0]*180/np.pi,diff_std[0]*180/np.pi)}")
        log.info(f"Joint 2 mean difference (deg): {mean_std_str(diff_mean[1]*180/np.pi,diff_std[1]*180/np.pi)}")
        log.info(f"Joint 3 mean difference (m):   {mean_std_str(diff_mean[2],diff_std[2])}")
        log.info(f"Joint 4 mean difference (deg): {mean_std_str(diff_mean[3]*180/np.pi,diff_std[3]*180/np.pi)}")
        log.info(f"Joint 5 mean difference (deg): {mean_std_str(diff_mean[4]*180/np.pi,diff_std[4]*180/np.pi)}")
        log.info(f"Joint 6 mean difference (deg): {mean_std_str(diff_mean[5]*180/np.pi,diff_std[5]*180/np.pi)}")

        # log.info(f"Results in rad")
        # log.info(f"Joint 1 mean difference (rad): {mean_std_str(diff_mean[0],diff_std[0])}")
        # log.info(f"Joint 2 mean difference (rad): {mean_std_str(diff_mean[1],diff_std[1])}")
        # log.info(f"Joint 3 mean difference (m):   {mean_std_str(diff_mean[2],diff_std[2])}")
        # log.info(f"Joint 4 mean difference (rad): {mean_std_str(diff_mean[3],diff_std[3])}")
        # log.info(f"Joint 5 mean difference (rad): {mean_std_str(diff_mean[4],diff_std[4])}")
        # log.info(f"Joint 6 mean difference (rad): {mean_std_str(diff_mean[5],diff_std[5])}")

        # plot
        if args.plot:
            CalibrationUtils.plot_joints(robot_df, tracker_df)
            fig, ax = plt.subplots(2, 1)
            CalibrationUtils.create_histogram(
                opt_df["q4res"], axes=ax[0], title=f"Q4 error (Z3 dot Z4)", xlabel="Optimization residual error"
            )
            CalibrationUtils.create_histogram(
                opt_df["q56res"],
                axes=ax[1],
                title=f"Q56 Optimization error",
                xlabel="Optimization residual error",
                max_val=0.003,
            )
            fig.set_tight_layout(True)
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # fmt:off
    parser.add_argument( "-r", "--root", type=str, default="./data/03_replay_trajectory/d04-rec-11-traj01", 
                    help="root dir") 
    parser.add_argument('--testid', nargs='*', help='test trajectories to generate', required=True, type=int)
    parser.add_argument( '-l', '--log', type=str, default="DEBUG", help="log level") 
    parser.add_argument('--reset', action='store_true',help="Recalculate joint values")
    parser.add_argument('-p','--plot', action='store_true',default=False \
                        ,help="Plot optimization error and joints plots.")
    # fmt:on
    args = parser.parse_args()
    log_level = args.log
    log = Logger("hysteresis curves", log_level=log_level).log

    log.info(f"Calculating the following test trajectories: {args.testid}")

    for testid in args.testid:
        main(testid)
