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
from kincalib.Motion.replay_rosbag import RosbagReplay, replay_device


def main():
    # ------------------------------------------------------------
    # Script configuration 
    # ------------------------------------------------------------
    parser = argparse.ArgumentParser()
    #fmt:off
    parser.add_argument( "-f", "--file", type=str, default="data/03_replay_trajectory/d03-rec-03_traj-01.txt", 
                         help="filename to save the data") 
    parser.add_argument( "-r", "--root", type=str, default="data/03_replay_trajectory/d04-rec-00", 
                         help="root dir to save the data")                     
    args = parser.parse_args()

    filename = Path(args.file)
    root = Path(args.root)
    if not filename.parent.exists():
        print("filename root directory does not exists.")
        sys.exit(0)

    expected_spheres = 4
    marker_name = "custom_marker_112"
    rospy.init_node("dvrk_bag_replay", anonymous=True)

    # ------------------------------------------------------------
    # Create replay class and specify the rosbag path
    # ------------------------------------------------------------
    rosbag_root = Path("data/psm2_trajectories/")
    file_p = rosbag_root / "pitch_exp_traj_01_test_cropped.bag"
    replay = RosbagReplay(file_p)
    replay.rosbag_utils.print_topics_info()

    # ------------------------------------------------------------
    # Extract trajectory and print summary
    # - Expected topic to use: /<arm>/setpoint_js
    # ------------------------------------------------------------
    arm_name = "PSM2"
    replay.collect_trajectory_setpoints()
    replay.trajectory_report()

    # ------------------------------------------------------------
    # Get robot ready
    # ------------------------------------------------------------
    arm = replay_device(device_namespace=arm_name, expected_interval=0.01)
    arm = dvrk.arm(arm_name=arm_name, expected_interval=0.01)
    setpoints = replay.setpoints

    # make sure the arm is powered
    print("-- Enabling arm")
    if not arm.enable(10):
        sys.exit("-- Failed to enable within 10 seconds")

    print("-- Homing arm")
    if not arm.home(10):
        sys.exit("-- Failed to home within 10 seconds")

    input('---> Make sure arm is ready to move. Press "Enter" to move to start position')
    arm.move_jp(np.array(setpoints[0].position)).wait()
    input('-> Press "Enter" to start data collection trajectory')

    # ------------------------------------------------------------
    # Execute trajectory
    # ------------------------------------------------------------
    # replay.execute_measure(arm, filename,marker_name,expected_spheres=expected_spheres)
    replay.robot_registration(arm,root,marker_name,expected_spheres=expected_spheres,save=True)


if __name__ == "__main__":
    main()
