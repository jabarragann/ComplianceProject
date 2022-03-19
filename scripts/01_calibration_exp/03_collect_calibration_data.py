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
import rospy

# kincalib module imports
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.Motion.replay_rosbag import RosbagReplay
from kincalib.Motion.ReplayDevice import ReplayDevice
from kincalib.Motion.TrajectoryPlayer import TrajectoryPlayer, Trajectory, RandomJointTrajectory


log = Logger("collection").log


def main():
    # ------------------------------------------------------------
    # Script configuration
    # ------------------------------------------------------------
    root = Path(args.root)
    if not root.exists():
        root.mkdir(parents=True)
        print(f"creating root: {root}")

    expected_spheres = 4
    marker_name = "custom_marker_112"
    arm_namespace = "PSM2"
    rospy.init_node("dvrk_bag_replay", anonymous=True)

    # ------------------------------------------------------------
    # Create trajectory from rosbag or random trajectory
    # ------------------------------------------------------------

    rosbag_path = Path(args.rosbag)
    rosbag_handle = RosbagUtils(rosbag_path)
    rosbag_handle.print_topics_info()
    # replay = RosbagReplay(rosbag)
    # replay.rosbag_utils.print_topics_info()

    trajectory = Trajectory.from_ros_bag(rosbag_handle)

    # ------------------------------------------------------------
    # Get robot ready
    # ------------------------------------------------------------
    arm = ReplayDevice(device_namespace=arm_namespace, expected_interval=0.01)
    # arm = dvrk.arm(arm_name=arm_name, expected_interval=0.01)
    # setpoints = replay.setpoints

    # ------------------------------------------------------------
    # Extract trajectory and print summary
    # - Expected topic to use: /<arm>/setpoint_js
    # ------------------------------------------------------------
    # replay.collect_trajectory_setpoints()
    # replay.trajectory_report()

    trajectory_player = TrajectoryPlayer(
        arm,
        trajectory,
        expected_markers=expected_spheres,
        root=root,
        marker_name=marker_name,
        mode=args.mode,
        test_id=args.test_id,
    )

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
    # Execute trajectory
    # ------------------------------------------------------------
    log.info("Collection information")
    log.info(f"Data root dir:   {args.root}")
    log.info(f"Trajectory name: {args.rosbag}")
    log.info(f"Mode: {args.mode} ")
    input('-> Press "Enter" to start data collection trajectory')

    if args.mode == "test":
        # Collect a testing trajectory.
        log.info("Collecting test trajectory")
        log.info(f"Saving data to  {args.file}")
        filename = (root / "test_trajectories") / args.file
        # replay.execute_measure(arm, filename, marker_name, expected_spheres=expected_spheres)
        trajectory_player.replay_trajectory(measure=True, saving=True, calibration=False)
    elif args.mode == "calib":
        # Collect calibration data.
        log.info("Collecting calibration data")
        # replay.collect_calibration_data(arm, root, marker_name, expected_spheres=expected_spheres)
        trajectory_player.replay_trajectory(measure=True, saving=True, calibration=True)


parser = argparse.ArgumentParser()
# fmt: off
parser.add_argument("-m","--mode", choices=["calib","test"], default="calib",
help="Select 'calib' mode for calibration collection. Select 'test' to collect a test trajectory")

parser.add_argument("-b", "--rosbag",type=str,
                    default="data/psm2_trajectories/pitch_exp_traj_01_test_cropped.bag",
                    help="rosbag trajectory to replay")
parser.add_argument( "-r", "--root", type=str, default="data/03_replay_trajectory/d04-rec-07-traj01", 
                        help="root dir to save the data.")                     
parser.add_argument( "-f", "--file", type=str, default="test_traj01", 
                        help="filename where a test trajectory data will be saved. execute_measure() func") 
args = parser.parse_args()

# fmt: on
if __name__ == "__main__":
    main()
