"""
rosbag record /PSM2/setpoint_js /PSM2/measured_js /force_sensor/measured_cf   
"""
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
from kincalib.utils.Logger import Logger
from kincalib.Motion.ReplayDevice import ReplayDevice
from kincalib.Sensors.ForceSensor import AtiForceSensor

log = Logger("template").log
np.set_printoptions(precision=4, suppress=True, sign=" ")


def main():
    # ------------------------------------------------------------
    #   Setup
    # ------------------------------------------------------------
    ati_force_handler = AtiForceSensor()
    psm_handler = ReplayDevice(device_namespace="PSM2", expected_interval=0.01)

    print("-- Enabling arm")
    if not psm_handler.enable(10):
        sys.exit("-- Failed to enable within 10 seconds")
    print("-- Homing arm")
    if not psm_handler.home(10):
        sys.exit("-- Failed to home within 10 seconds")

    # ------------------------------------------------------------
    #   Move robot to a predefined base position
    # ------------------------------------------------------------
    init_jp = np.array([-0.1661, -0.4076, 0.2025, 0.0738, -0.0588, -0.7197])
    psm_handler.move_jp(init_jp)

    # ------------------------------------------------------------
    #   Do incremental motions until the force exceeds
    #   a threshold.
    # ------------------------------------------------------------
    force_thresh = 1.0  # 1Newton
    sensed_force = None
    # force_thresh < sensed_force
    while True:
        answer = input("continue (c) or quit (q) ")
        if answer == "c":
            init_jp[2] += 0.004
            psm_handler.move_jp(init_jp).wait()
            time.sleep(0.2)
            force = ati_force_handler.measured_cf()[:3]

            log.info(f"force measurement: {force}")
            log.info(f"mag: {np.linalg.norm(force):0.05f}")
        elif answer == "q":
            log.info("Quiting")
            break


if __name__ == "__main__":

    main()
