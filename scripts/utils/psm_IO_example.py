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

# Robotic toolbox
from spatialmath.base import r2q
from torch import cartesian_prod

# ROS and DVRK imports
import dvrk
from kincalib.Recording.DataRecord import Record
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
    def __init__(self, replay_device):
       self.psm_handler = replay_device 
    
    def __call__(self):
        cartesian_pose = pm.toMatrix(self.psm_handler.measured_cp()) 
        cartesian_pose = cartesian_pose[:3,3] 
        log.info(cartesian_pose)

def main():
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    rospy.init_node("dvrk_bag_replay", anonymous=True)
    replay_device = ReplayDevice("PSM2",expected_interval=0.01)
    replay_device.home_device()
    psm_info_cb = PsmInfo(replay_device=replay_device)
    psm_io2 = PsmIO("PSM2", action=psm_info_cb)
    time.sleep(0.2)

    print("Recording the kinematic information with IO ...")
    # Loop until you receive a keyboard interruption
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)


if __name__ == "__main__":

    main()