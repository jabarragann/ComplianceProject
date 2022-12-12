# Python imports
from pathlib import Path
import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ROS and DVRK imports
import dvrk
from kincalib.Calibration.CalibrationRoutines import DhParamCalibrationRoutine
from kincalib.Recording.DataRecorder import DataRecorder

import rospy
from kincalib.Recording.RecordCollections import ExperimentRecordCollection
from kincalib.Sensors.ftk_500_api import ftk_500

# kincalib module imports
from kincalib.utils.Logger import Logger
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
    parser.add_argument("-t","--testid",type=int, help="testid required for test mode")
    parser.add_argument( "-r", "--root", type=str, required=True, 
                            help="root dir to save the data.")                     
    parser.add_argument( "-s", "--save", action='store_true',  default=False, 
                            help="Use this flag to Save recorded data") 
    parser.add_argument("-d","--description", type=str, required=True, help="One line experiment description")
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
    # Get robot ready
    # ------------------------------------------------------------
    arm = ReplayDevice(device_namespace=arm_namespace, expected_interval=0.01)
    ftk_handler = ftk_500(marker_name, expected_spheres)

    print("-- Enabling arm")
    if not arm.enable(10):
        sys.exit("-- Failed to enable within 10 seconds")
    print("-- Homing arm")
    if not arm.home(10):
        sys.exit("-- Failed to home within 10 seconds")

    input('---> Make sure arm is ready to move. Press "Enter" to move to start position')

    # arm.move_jp(np.array(trajectory[0].position)).wait()

    # ------------------------------------------------------------
    # Config recording loops
    # ------------------------------------------------------------
    # Setup calibration callbacks
    dh_param_recorder = DataRecorder(arm, None, ftk_handler, expected_spheres, marker_name)
    dh_param_routine_cb = DhParamCalibrationRoutine(
        arm, ftk_handler, dh_param_recorder, save=True, root=root
    )

    # Execute
    log.info("Collection information")
    log.info(f"Data root dir:     {args.root}")
    log.info(f"Trajectory name:   {args.rosbag}")
    log.info(f"Save data:         {args.save}")
    if args.rosbag is None:
        log.info(f"Random Trajectory: {args.traj_type}")
    log.info(f"Mode:              {args.mode} ")
    log.info(f"Test id:           {testid} ")
    log.info(f"Description:       {args.description}")

    ans = input(
        'Press "y" to start data collection trajectory. Ensure gripper is outsite the cannula. '
    )
    if ans == "y":
        dh_param_routine_cb(args.test_id)
    else:
        exit(0)


if __name__ == "__main__":
    main()
