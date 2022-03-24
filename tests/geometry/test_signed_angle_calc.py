"""
Script to calculate the signed angle between two vectors.
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

log = Logger("template").log
np.set_printoptions(precision=4, suppress=True, sign=" ")

from numpy import cos, sin, arctan2

c = cos
s = sin


def main():
    # ------------------------------------------------------------
    # Sample comment
    # ------------------------------------------------------------

    # fmt:off
    rot_z = lambda x: np.array([[c(x),-s(x),0],
                                [s(x), c(x),0],
                                [0,    0,   1]])
    # fmt:on
    vn = np.array([0, 0, 1])
    theta = -45 * np.pi / 180
    va = np.array([1, 0, 0])
    vb = rot_z(theta) @ va
    vn = vn / np.linalg.norm(vn)
    log.info(va)
    log.info(vb)
    theta_est = arctan2(np.cross(va, vb).dot(vn), np.dot(va, vb))
    log.info(f"theta est {theta_est*180/np.pi:0.04}")


if __name__ == "__main__":

    main()
