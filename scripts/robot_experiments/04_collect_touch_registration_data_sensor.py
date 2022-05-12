# Python imports
from pathlib import Path
from re import I
import time
import argparse
import sys
from gpg import Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Set
from rich.logging import RichHandler
from rich.progress import track

# Robotic toolbox
from spatialmath.base import r2q
from torch import cartesian_prod

# ROS and DVRK imports
import dvrk
from kincalib.Recording.DataRecord import Record
from kincalib.Recording.DataRecorder import DataRecorder
import rospy
import tf_conversions.posemath as pm

# kincalib module imports
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.Logger import Logger
from kincalib.Motion.DvrkKin import DvrkPsmKin
from kincalib.Motion.ReplayDevice import ReplayDevice
from kincalib.Motion.PsmIO import PsmIO

log = Logger("template").log
np.set_printoptions(precision=4, suppress=True, sign=" ")


class PsmInfo:
    def __init__(self, replay_device, psm_io):
        self.psm_handler = replay_device
        self.psm_io: PsmIO = psm_io
        self.counter = 0

    def __call__(self, **kwargs):
        self.counter += 1
        cartesian_pose = pm.toMatrix(self.psm_handler.measured_cp())
        cartesian_pose = cartesian_pose[:3, 3]
        log.info(f"Pose {self.counter:02d}: {cartesian_pose}")
        # change callback parameters
        self.psm_io.kwargs = {"index": self.counter}


def main():
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    rospy.init_node("dvrk_bag_replay", anonymous=True)
    replay_device = ReplayDevice("PSM2", expected_interval=0.01)
    replay_device.home_device()

    expected_markers = 4
    root = Path(args.root)
    log.info(f"Saving data to: {root}")
    marker_name = "custom_marker_112"
    mode = "test"
    test_id: int = args.testid
    psm_io2 = PsmIO("PSM2")
    data_recorder_cb = DataRecorder(replay_device, expected_markers, root, marker_name, mode, test_id)
    psm_info_cb = PsmInfo(replay_device, psm_io2)
    psm_io2.action_list = [psm_info_cb, data_recorder_cb]
    time.sleep(0.2)

    print("Recording the kinematic information with IO ...")
    # Loop until you receive a keyboard interruption
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("-t","--testid",type=int, help="testid required for test mode", default=0)
    parser.add_argument( "-r", "--root", type=str, default="data/04_robot_measuring_experiments/registration_sensor_exp", 
                            help="root dir to save the data.")                     
    args = parser.parse_args()
    # fmt: on
    main()
