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
import tf_conversions.posemath as pm

# kincalib module imports
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.Logger import Logger
from kincalib.Motion.DvrkKin import DvrkPsmKin
from kincalib.Motion.ReplayDevice import ReplayDevice

log = Logger("template").log
np.set_printoptions(precision=4, suppress=True, sign=" ")


# fmt:off
#Coordinates from aluminum phantom
phantom_coord = {
"1":np.array([  -12.7  ,  12.7 ,    0]),
"2":np.array([  -12.7  ,  31.75,    0]),
"3":np.array([  -58.42 ,  31.75,    0]),
"4":np.array([  -53.34 ,  63.5 ,    0]),
"5":np.array([  -12.7  ,  63.5 ,    0]),
"6":np.array([  -12.7  , 114.3 ,    0]),
"7":np.array([ -139.7  , 114.3 ,    0]),
"8":np.array([ -139.7  ,  63.5 ,    0]),
"9":np.array([ -139.7  ,  12.7 ,    0]),
"A":np.array([ -193.675,  19.05,   25.4]),
"B":np.array([ -193.675,  44.45,   50.8]),
"C":np.array([ -193.675,  69.85,   76.2]),
"D":np.array([ -193.675,  95.25,  101.6])}

# Base transforms based on DVRK console configuration file
tool_offset = np.array([ [ 0.0, -1.0,  0.0,  0.0],
                         [ 0.0,  0.0,  1.0,  0.0],
                         [-1.0,  0.0,  0.0,  0.0],
                         [ 0.0,  0.0,  0.0,  1.0]])

base_transform =np.array([[  1.0,  0.0,          0.0,          0.20],
                          [  0.0, -0.866025404,  0.5,          0.0 ],
                          [  0.0, -0.5,         -0.866025404,  0.0 ],
                          [  0.0,  0.0,          0.0,          1.0 ]])

# Base transforms for measuring experiments
tool_offset_exp = np.array([[ 0.0, -1.0,  0.0,  0.0],
                            [ 0.0,  0.0,  1.0,  0.019],
                            [-1.0,  0.0,  0.0,  0.0],
                            [ 0.0,  0.0,  0.0,  1.0]])

base_transform_exp =np.array([[  1.0,  0.0,          0.0,          0.20],
                              [  0.0, -0.866025404,  0.5,          0.0 ],
                              [  0.0, -0.5,         -0.866025404,  0.0 ],
                              [  0.0,  0.0,          0.0,          1.0 ]])

# fmt:on
def main():
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    psm_handle = ReplayDevice("PSM2", expected_interval=0.01)
    psm_handle.home_device()
    psm_fkin = DvrkPsmKin(tool_offset=tool_offset, base_transform=base_transform)
    time.sleep(0.2)

    # ------------------------------------------------------------
    # Check that kinematic model matches measured_cp
    # ------------------------------------------------------------
    measured_jp = psm_handle.measured_jp()
    # CRTK cartesian pose
    measured_cp = psm_handle.measured_cp()
    measured_cp = pm.toMatrix(measured_cp)
    # My model cartesian pose
    model_cp = psm_fkin.fkine(measured_jp).data[0]
    log.info(f"Error between dvrk pose and my model pose\n{measured_cp-model_cp}")

    # ------------------------------------------------------------
    # Measuring experiment
    # ------------------------------------------------------------

    # Checking
    psm_fkin_correct_tool = DvrkPsmKin(tool_offset=tool_offset_exp, base_transform=np.identity(4))
    # log.info(psm_fkin_correct_tool.fkine(np.zeros(6)).data[0]) # just to debug the tool offset

    # loc: str = input("Input phantom locations separed by a space: ")
    loc = "A C"
    loc = loc.strip().split(" ")
    log.info(
        f"Distance between {loc[0]} and {loc[1]} (True): {np.linalg.norm(phantom_coord[loc[0]]-phantom_coord[loc[1]]):0.4f}"
    )

    input(f"collect dvrk joints on {loc[0]} and press enter")
    cp1 = psm_fkin_correct_tool.fkine(psm_handle.measured_jp()).data[0]

    input(f"collect dvrk joints on {loc[1]} and press enter")
    cp2 = psm_fkin_correct_tool.fkine(psm_handle.measured_jp()).data[0]

    log.info(f"Distance between {loc[0]} and {loc[1]} (Measured): {np.linalg.norm(cp1[:3,3]- cp2[:3,3]*1000):0.04f}")


if __name__ == "__main__":

    main()
