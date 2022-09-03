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

# ROS and DVRK imports
import dvrk
from kincalib.Recording.DataRecorder import DataRecorder, OuterJointsCalibrationRecorder, WristJointsCalibrationRecorder
import rospy

# kincalib module imports
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.Motion.ReplayDevice import ReplayDevice
from kincalib.Motion.TrajectoryPlayer import (
    SoftRandomJointTrajectory,
    TrajectoryPlayer,
    Trajectory,
    RandomJointTrajectory,
)

log = Logger("collection").log


def main():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("-m","--mode", choices=["calib","test"], default="calib",
        help="Select 'calib' mode for calibration collection. Select 'test' to collect a test trajectory")
    parser.add_argument("-t","--testid",type=int, help="testid required for test mode")
    # parser.add_argument("-b", "--rosbag",type=str,
    #                     default="data/psm2_trajectories/pitch_exp_traj_01_test_cropped.bag",
    #                     help="rosbag trajectory to replay")
    parser.add_argument("-b", "--rosbag",type=str, default=None,
                        help="rosbag trajectory to replay")
    parser.add_argument( "-r", "--root", type=str, default="data/03_replay_trajectory/d04-rec-14-traj01", 
                            help="root dir to save the data.")                     
    parser.add_argument( "-s", "--save", action='store_true',  default=False, 
                            help="Use this flag to Save recorded data") 
    parser.add_argument("-d","--description", type=str, required=True, help="One line experiment description")
    #NUMBER OF SAMPLES ARGS
    args = parser.parse_args()
    # fmt: on

    # ------------------------------------------------------------
    # Script configuration
    # ------------------------------------------------------------
    root = Path(args.root)
    if not root.exists():
        root.mkdir(parents=True)
        log.info(f"creating root: {root}")

    expected_spheres = 4
    marker_name = "custom_marker_112"
    arm_namespace = "PSM2"
    testid = args.testid if args.mode == "test" else 0
    sampling_factor = 20 if args.mode == "test" else 60
    rospy.init_node("dvrk_bag_replay", anonymous=True)

    # ------------------------------------------------------------
    # Create trajectory from rosbag or random trajectory
    # ------------------------------------------------------------
    if args.rosbag is not None:
        rosbag_path = Path(args.rosbag)
        if not rosbag_path.exists():
            log.error("rosbag path does not exists")
            exit(0)
        rosbag_handle = RosbagUtils(rosbag_path)
        rosbag_handle.print_topics_info()
        rosbag_handle.print_number_msg()
        trajectory = Trajectory.from_ros_bag(rosbag_handle, sampling_factor=sampling_factor)
    else:
        log.info(f"Creating random trajectory")
        # trajectory = RandomJointTrajectory.generate_trajectory(1200)
        trajectory = SoftRandomJointTrajectory.generate_trajectory(500, samples_per_step=20)

    log.info(f"Trajectory size {len(trajectory)}")

    # ------------------------------------------------------------
    # Get robot ready
    # ------------------------------------------------------------
    arm = ReplayDevice(device_namespace=arm_namespace, expected_interval=0.01)

    # make sure the arm is powered
    print("-- Enabling arm")
    if not arm.enable(10):
        sys.exit("-- Failed to enable within 10 seconds")
    print("-- Homing arm")
    if not arm.home(10):
        sys.exit("-- Failed to home within 10 seconds")

    input('---> Make sure arm is ready to move. Press "Enter" to move to start position')
    arm.move_jp(np.array(trajectory[0].position)).wait()

    # ------------------------------------------------------------
    # Config trajectory player
    # ------------------------------------------------------------
    # Callbacks
    outer_js_calib_cb = OuterJointsCalibrationRecorder(
        replay_device=arm, save=args.save, expected_markers=expected_spheres, root=root, marker_name=marker_name
    )
    wrist_js_calib_cb = WristJointsCalibrationRecorder(
        replay_device=arm, save=args.save, expected_markers=expected_spheres, root=root, marker_name=marker_name
    )
    data_recorder_cb = DataRecorder(
        arm,
        expected_markers=expected_spheres,
        root=root,
        marker_name=marker_name,
        mode=args.mode,
        test_id=testid,
        description=args.description,
    )

    if args.mode == "test":
        trajectory_player = TrajectoryPlayer(
            arm,
            trajectory,
            before_motion_loop_cb=[],
            after_motion_cb=[data_recorder_cb],
        )
    elif args.mode == "calib":
        trajectory_player = TrajectoryPlayer(
            arm,
            trajectory,
            before_motion_loop_cb=[outer_js_calib_cb],  # [outer_js_calib_cb],
            after_motion_cb=[data_recorder_cb, wrist_js_calib_cb],
        )

    # Execute
    log.info("Collection information")
    log.info(f"Data root dir:     {args.root}")
    log.info(f"Trajectory name:   {args.rosbag}")
    log.info(f"Save data:         {args.save}")
    log.info(f"Trajectory length: {len(trajectory)}")
    log.info(f"Mode:              {args.mode} ")
    log.info(f"Test id:           {testid} ")
    log.info(f"Description:       {args.description}")

    ans = input('Press "y" to start data collection trajectory. Only replay trajectories that you know. ')
    if ans == "y":
        trajectory_player.replay_trajectory(execute_cb=True)
    else:
        exit(0)


if __name__ == "__main__":
    main()
