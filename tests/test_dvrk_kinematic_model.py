""" Script to test if robotic toolbox dvrk model produces the same values as the cisst-saw software"""

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

# Robotic toolbox
from spatialmath.base import r2q
from pyquaternion import Quaternion

# ROS and DVRK imports
import dvrk
import rospy

# kincalib module imports
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.Motion.DvrkKin import DvrkPsmKin

np.set_printoptions(precision=4, suppress=True, sign=" ")
log = Logger("main").log


def obtain_pos(cartesian_t: np.ndarray):
    position_list = []
    for i in range(len(cartesian_t)):
        position_list.append(cartesian_t[i][:3, 3])

    return np.array(position_list)


def obtain_quaternions(cartesian_t: np.ndarray):
    quaternion_list = []
    for i in range(len(cartesian_t)):
        # method 1
        q = Quaternion(matrix=cartesian_t[i][:3, :3])
        q = [q.real.tolist()] + q.imaginary.tolist()
        quaternion_list.append(q)
        # method 2
        # quaternion_list.append(r2q(cartesian_t[i][:3, :3]))

    return np.array(quaternion_list)


def main():
    # ------------------------------------------------------------
    # Read data
    # Transformation between rotation and quaternions will produce
    # different results depending on the transformation
    # ------------------------------------------------------------

    # Test case 1 - will only work with quaternions method 2 in obtain_quaternion function
    # root = Path("data/03_replay_trajectory/d04-rec-07-traj01/robot_mov")
    # # fmt:off
    # tool_offset = np.array([[ 0.0, -1.0,  0.0,  0.0  ],
    #                         [ 0.0,  0.0,  1.0,  0.0  ],
    #                         [-1.0,  0.0,  0.0,  0.0  ],
    #                         [ 0.0,  0.0,  0.0,  1.0  ]])
    # base_transform =np.array([[  1.0,  0.0,          0.0,          0.20],
    #                           [  0.0, -0.866025404,  0.5,          0.0 ],
    #                           [  0.0, -0.5,         -0.866025404,  0.0 ],
    #                           [  0.0,  0.0,          0.0,          1.0 ]])
    # # fmt:on
    # psm_kin = DvrkPsmKin(tool_offset=tool_offset, base_transform=base_transform)

    # Test case 2 - will only work with quaternions method 1 in obtain_quaternion function
    root = Path("data/03_replay_trajectory/d04-rec-12-traj01/test_trajectories/03")
    psm_kin = DvrkPsmKin()

    robot_cp_df = pd.read_csv(root / "robot_cp.txt")
    robot_cp_df = robot_cp_df.loc[robot_cp_df["m_t"] == "r"]
    robot_jp_df = pd.read_csv(root / "robot_jp.txt")

    # ------------------------------------------------------------
    # Calculate cartesian using custom kinematic model
    # ------------------------------------------------------------
    cartesian_custom = psm_kin.fkine(robot_jp_df[["q1", "q2", "q3", "q4", "q5", "q6"]].to_numpy())
    position_custom = obtain_pos(cartesian_custom.data)
    orientation_custom = obtain_quaternions(cartesian_custom.data)

    log.info(position_custom.shape)
    log.info(robot_cp_df.shape)

    r = position_custom - robot_cp_df[["px", "py", "pz"]].to_numpy()
    log.info(f"error {r.mean(axis=0)}")
    log.info(f"std   {r.std(axis=0)}")

    # ------------------------------------------------------------
    # Plot data
    # ------------------------------------------------------------
    fig, axes = plt.subplots(4, 2)
    axes[0, 0].plot(robot_cp_df["px"].to_numpy(), label="cisst-saw")
    axes[0, 0].plot(position_custom[:, 0], label="custom")
    axes[0, 0].set_title("px")

    axes[1, 0].plot(robot_cp_df["py"].to_numpy())
    axes[1, 0].plot(position_custom[:, 1], label="custom")
    axes[1, 0].set_title("py")

    axes[2, 0].plot(robot_cp_df["pz"].to_numpy())
    axes[2, 0].plot(position_custom[:, 2], label="custom")
    axes[2, 0].set_title("pz")
    axes[0, 0].legend()

    axes[0, 1].plot(robot_cp_df["qx"].to_numpy(), label="cisst-saw")
    axes[0, 1].plot(orientation_custom[:, 1], label="custom")
    axes[0, 1].set_title("qx")

    axes[1, 1].plot(robot_cp_df["qy"].to_numpy(), label="cisst-saw")
    axes[1, 1].plot(orientation_custom[:, 2], label="custom")
    axes[1, 1].set_title("qy")

    axes[2, 1].plot(robot_cp_df["qz"].to_numpy(), label="cisst-saw")
    axes[2, 1].plot(orientation_custom[:, 3], label="custom")
    axes[2, 1].set_title("qz")

    axes[3, 1].plot(robot_cp_df["qw"].to_numpy(), label="cisst-saw")
    axes[3, 1].plot(orientation_custom[:, 0], label="custom")
    axes[3, 1].set_title("qw")

    plt.show()


if __name__ == "__main__":

    main()
