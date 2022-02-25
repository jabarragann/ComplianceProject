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
import PyKDL
import tf_conversions.posemath as pm

# kincalib module imports
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.Frame import Frame

np.set_printoptions(precision=4, suppress=True, sign=" ")


def pykdl2frame(frame: PyKDL.Frame):
    mat = pm.toMatrix(frame)
    return Frame(mat[:3, :3], mat[:3, 3])


def calculate_mean_frame(
    frame_arr: List[PyKDL.Frame],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[summary]

    Args:
        frame_arr (List[PyKDL.Frame]): [description]

    Returns:
        Tuple[np.ndarray,np.ndarray,np.ndarray]: mean_frame, position_std, orientation_std
    """

    position = []
    orientation = []
    for k in range(len(frame_arr)):
        position.append(np.array(list(frame_arr[k].p)))
        orientation.append(np.array(list(frame_arr[k].M.GetQuaternion())))

    position_mean = np.array(position).mean(axis=0)
    orientation_mean = np.array(orientation).mean(axis=0)
    position_std = np.array(position).std(axis=0)
    orientation_std = np.array(orientation).std(axis=0)
    orientation_mean = orientation_mean / np.linalg.norm(orientation_mean)

    mean_frame = PyKDL.Frame(
        PyKDL.Rotation.Quaternion(*orientation_mean), PyKDL.Vector(*position_mean)
    )
    return mean_frame, position_std, orientation_std


# if __name__ == "__main__":

#     main()
