"""
Script to crop a bag file based on linear velocity of end effector
Ensure that the bag has the following topics
- "/PSM2/measured_js"
- "/PSM2/setpoint_js"
- "/PSM2/measured_cv"

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

# ROS and DVRK imports
import dvrk
import rospy

# kincalib module imports
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.RosbagUtils import RosbagUtils


def plot_cv(linear_vel):
    fig, axes = plt.subplots(3, 1)
    axes[0].set_title("Cartesian linear velocity")
    axes[0].plot(linear_vel[:, 0])
    axes[0].set_ylabel("x vel")
    axes[1].plot(linear_vel[:, 1])
    axes[1].set_ylabel("y vel")
    axes[2].plot(linear_vel[:, 2])
    axes[2].set_ylabel("z vel")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    #fmt:off
    parser.add_argument( "-f", "--file", type=str, default="./data/psm2_trajectories/test_trajectory1.bag",
                         help="path where the rosbag is located.") #fmt:on
    args = parser.parse_args()

    # ------------------------------------------------------------
    # Open rosbag
    # ------------------------------------------------------------

    file_p = Path(args.file)
    root = file_p.parent 

    log = Logger("rosbag_utils").log

    rb = RosbagUtils(file_p, log=log)
    rb.print_topics_info()

    # ------------------------------------------------------------
    # Extract data
    # ------------------------------------------------------------
    topics = ["/PSM2/measured_js", "/PSM2/setpoint_js", "/PSM2/measured_cv"]
    msg_dict, msg_dict_ts = rb.read_messages(topics)

    log.info(f"Messages summary in {file_p.name}")
    for i in range(len(topics)):
        log.info(f"Number of messages in {topics[i]}: {len(msg_dict[topics[i]])} ")

    jp, jv = RosbagUtils.extract_joint_state_data(msg_dict[topics[0]])
    linear_vel = RosbagUtils.extract_twist_data(msg_dict[topics[2]])

    log.info("use linear velocity plot to identify max and min indexes")
    plot_cv(linear_vel)

    # ------------------------------------------------------------
    # Get max and min index to crop rosbag.
    # This indicates the max and min ts that you want to include
    # ------------------------------------------------------------
    max_idx = int(input("max_idx: "))
    min_idx = int(input("min_idx: "))

    max_ts = msg_dict_ts["/PSM2/measured_cv"][max_idx]
    min_ts = msg_dict_ts["/PSM2/measured_cv"][min_idx]
    rb.save_crop_bag(min_ts, max_ts)

    # ------------------------------------------------------------
    # See crop data
    # ------------------------------------------------------------
    file_crop = file_p.parent / (file_p.with_suffix("").name + "_cropped.bag")
    rb_crop = RosbagUtils(file_crop, log=log)
    rb_crop.print_topics_info()

    topics = ["/PSM2/measured_js", "/PSM2/setpoint_js", "/PSM2/measured_cv"]
    msg_dict, msg_dict_ts = rb_crop.read_messages(topics)

    log.info(f"Messages summary in {file_crop.name}")
    for i in range(len(topics)):
        log.info(f"Number of messages in {topics[i]}: {len(msg_dict[topics[i]])} ")
    linear_vel = RosbagUtils.extract_twist_data(msg_dict[topics[2]])

    plot_cv(linear_vel)


if __name__ == "__main__":
    main()
