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
import torch

# ROS and DVRK imports
from kincalib.Learning.Dataset2 import JointsDataset1, Normalizer
from kincalib.Learning.InferencePipeline import InferencePipeline
from kincalib.Learning.Models import BestMLP2
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
from pytorchcheckpoint.checkpoint import CheckpointHandler

from kincalib.utils.TableGenerator import ResultsTable

np.set_printoptions(precision=4, suppress=True, sign=" ")

def plot_joints(robot_df, tracker_df, pred_df):
        fig, axes = plt.subplots(6, 1, sharex=True)
        robot_df.reset_index(inplace=True)
        tracker_df.reset_index(inplace=True)
        for i in range(6):
            # fmt:off
            axes[i].set_title(f"joint {i+1}")
            axes[i].plot(robot_df['step'].to_numpy(), robot_df[f"q{i+1}"].to_numpy(), color="blue")
            axes[i].plot(robot_df['step'].to_numpy(), robot_df[f"q{i+1}"].to_numpy(), marker="*", linestyle="None", color="blue", label="robot")
            axes[i].plot(tracker_df['step'].to_numpy(),tracker_df[f"tq{i+1}"].to_numpy(), color="orange")
            axes[i].plot(tracker_df['step'].to_numpy(),tracker_df[f"tq{i+1}"].to_numpy(), marker="*", linestyle="None", color="orange", label="tracker")

            if i >2:
                axes[i].plot(pred_df['step'].to_numpy(),pred_df[f"q{i+1}"].to_numpy(), color="green")
                axes[i].plot(pred_df['step'].to_numpy(),pred_df[f"q{i+1}"].to_numpy(), marker="*", linestyle="None", color="green", label="predicted")

            # fmt:on

        axes[5].legend()

def main(testid: int):
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    # Important paths
    root = Path(args.root)

    if args.test:
        robot_cp_p = root / f"test_trajectories/{testid:02d}" / "robot_cp.txt"
    else:
        robot_cp_p = root / "robot_mov" / "robot_cp.txt"

    # ------------------------------------------------------------
    # Calculate tracker joint values
    # ------------------------------------------------------------
    dst_p = robot_cp_p.parent
    dst_p = dst_p / "result"
    registration_data_path = root / "registration_results"

    log.info(f"Storing results in {dst_p}")
    log.info(f"Loading registration .json from {registration_data_path}")
    dst_p.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Read calculated joint values
    # ------------------------------------------------------------
    if (dst_p / "robot_joints.txt").exists():
        robot_df = pd.read_csv(dst_p.parent / "robot_jp.txt", index_col=None)
        tracker_df = pd.read_csv(dst_p / "tracker_joints.txt", index_col=None)
        opt_df = pd.read_csv(dst_p / "opt_error.txt", index_col=None)
    else:
        log.info(f"robot_joints.txt file not found in {dst_p}")
        exit()

    # ------------------------------------------------------------
    # Calculate model predictions 
    # ------------------------------------------------------------

    # root = Path(f"data/ModelsCheckpoints/T{3:02d}/") / "final_checkpoint.pt"
    model_path = Path(f"data/deep_learning_data/Studies/TestStudy2")/args.modelname 
    inference_pipeline = InferencePipeline(model_path)

    cols= ["q1", "q2", "q3", "q4", "q5", "q6"] + ["t1", "t2", "t3", "t4", "t5", "t6"]
    valstopred = robot_df[cols].to_numpy().astype(np.float32)

    pred =  inference_pipeline.predict(valstopred)
    pred = np.hstack((robot_df["step"].to_numpy().reshape(-1,1),pred))
    pred_df = pd.DataFrame(pred,columns=["step", "q4", "q5", "q6"])

    # Calculate results 
    # Use the residual to get valid tracker points.
    valid_steps = opt_df.loc[opt_df["q56res"] < 0.005]["step"]  

    robot_valid = robot_df.loc[robot_df["step"].isin(valid_steps)].iloc[:, 1:7].to_numpy()
    tracker_valid = tracker_df.loc[tracker_df["step"].isin(valid_steps)].iloc[:, 1:].to_numpy()
    robot_error = robot_valid - tracker_valid
    robot_mean = robot_error.mean(axis=0)
    robot_std = robot_error.std(axis=0)

    # Network error
    network_valid = pred_df.loc[pred_df["step"].isin(valid_steps)][[ "q4", "q5", "q6"]].to_numpy()
    network_error = network_valid - tracker_valid[:,3:]
    network_mean = network_error.mean(axis=0)
    network_std = network_error.std(axis=0)

    # Calculate cartesian errors calculate cartesian positions from robot_valid and tracker_valid
    robot_cp = CalibrationUtils.calculate_cartesian(robot_valid)
    tracker_cp = CalibrationUtils.calculate_cartesian(tracker_valid)

    # Add the first three joints from the robot
    network_valid =np.hstack((robot_valid[:,:3],network_valid))
    network_cp = CalibrationUtils.calculate_cartesian(network_valid)

    robot_cp_error = tracker_cp - robot_cp
    robot_mean_error = robot_cp_error.apply(np.linalg.norm, 1)

    net_cp_error = tracker_cp - network_cp 
    net_mean_error = net_cp_error.apply(np.linalg.norm, 1)

    # Create results table
    table = ResultsTable()
    table.add_data(
        dict(
            type="robot", 
            q1=mean_std_str(robot_mean[0]*180/np.pi, robot_std[0]*180/np.pi),
            q2=mean_std_str(robot_mean[1]*180/np.pi, robot_std[1]*180/np.pi),
            q3=mean_std_str(robot_mean[2]*180/np.pi, robot_std[2]*180/np.pi),
            q4=mean_std_str(robot_mean[3]*180/np.pi, robot_std[3]*180/np.pi),
            q5=mean_std_str(robot_mean[4]*180/np.pi, robot_std[4]*180/np.pi),
            q6=mean_std_str(robot_mean[5]*180/np.pi, robot_std[5]*180/np.pi),
            cartesian=mean_std_str(robot_mean_error.mean()*1000, robot_mean_error.std()*1000),
        )
    )
    table.add_data(
        dict(
            type="network", 
            q4=mean_std_str(network_mean[0]*180/np.pi, network_std[0]*180/np.pi),
            q5=mean_std_str(network_mean[1]*180/np.pi, network_std[1]*180/np.pi),
            q6=mean_std_str(network_mean[2]*180/np.pi, network_std[2]*180/np.pi),
            cartesian=mean_std_str(net_mean_error.mean()*1000, net_mean_error.std()*1000),
        )
    )

    print("")
    print(f"**Evaluation report for test trajectory {testid} in {registration_data_path.parent.name}**")
    print(f"* Registration path: {registration_data_path}")
    print(f"* model path: {model_path}\n")
    print(f"Difference from ground truth (Tracker values) (N={robot_error.shape[0]})")
    print(f"\n{table.get_full_table()}")

    # plot
    if args.plot:
        plot_joints(robot_df, tracker_df, pred_df)
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
    ##TODO: Indicate that you need to use script 04 first to generate the tracker joints and then use this 
    ##TODO: to plot the corrected joints.

    parser = argparse.ArgumentParser()
    # fmt:off
    parser.add_argument( "-r", "--root", type=str, default="./data/03_replay_trajectory/d04-rec-11-traj01", 
                    help="This directory must a registration_results subdir contain a calibration .json file.") 
    parser.add_argument('-t','--test', action='store_true',help="Use test data")
    parser.add_argument('--testid', nargs='*', help='test trajectories to generate', required=True, type=int)
    parser.add_argument('-m','--modelname', type=str,default=False,required=True \
                        ,help="Name of deep learning model to use.")
    parser.add_argument( '-l', '--log', type=str, default="DEBUG", help="log level") 
    parser.add_argument('-p','--plot', action='store_true',default=False \
                        ,help="Plot optimization error and joints plots.")
    # fmt:on
    args = parser.parse_args()
    log_level = args.log
    log = Logger(__name__, log_level=log_level).log

    log.info(f"Calculating the following test trajectories: {args.testid}")

    for testid in args.testid:
        main(testid)
