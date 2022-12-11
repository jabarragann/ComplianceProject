# Python imports
from itertools import product
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

# ROS and DVRK imports
import dvrk
from kincalib.Recording.DataRecord import AtracsysCartesianRecord, JointRecord, RobotSensorCartesianRecord
from kincalib.Sensors.ftk_500_api import ftk_500
import rospy

# kincalib module imports
from kincalib.Recording.DataRecorder import DataRecorder, OuterJointsCalibrationRecorder, WristJointsCalibrationRecorder
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.Motion.ReplayDevice import ReplayDevice
from kincalib.Motion.TrajectoryPlayer import TrajectoryPlayer, Trajectory, RandomJointTrajectory
from kincalib.Motion.DvrkMotions import DvrkMotions


log = Logger("collection").log

expected_spheres = 4
marker_name = "custom_marker_112"


def go_and_come_back(joint_values):
    joint_values = DvrkMotions.generate_pitch_motion()
    joint_values_rev = joint_values.copy()
    joint_values_rev = np.flip(joint_values_rev, axis=0)
    joint_values = np.concatenate((joint_values, joint_values_rev[1:]))
    return joint_values


def roll_hysteresis(arm_handler, root, save, init_jp):
    log.info(init_jp)
    arm_handler.move_jp(init_jp).wait()
    # init_roll = init_jp[3]
    init_pitch = init_jp[4]
    init_yaw = init_jp[5]

    # Create hysteresis motion
    roll_traj = DvrkMotions.generate_roll_motion()
    roll_traj = go_and_come_back(roll_traj)
    roll_traj = list(product(roll_traj, [init_pitch], [init_yaw]))

    hysteresis_collection("roll", init_jp.copy(), arm_handler, expected_spheres, save, root=root, trajectory=roll_traj)


def pitch_hysteresis(arm_handler, root, save, init_jp):
    log.info(init_jp)
    arm_handler.move_jp(init_jp).wait()
    init_roll = init_jp[3]
    # init_pitch = init_jp[4]
    init_yaw = init_jp[5]

    # Create hysteresis motion
    pitch_traj = DvrkMotions.generate_pitch_motion()
    pitch_traj = go_and_come_back(pitch_traj)
    pitch_traj = list(product([init_roll], pitch_traj, [init_yaw]))

    hysteresis_collection(
        "pitch", init_jp.copy(), arm_handler, expected_spheres, save, root=root, trajectory=pitch_traj
    )


def yaw_hysteresis(arm_handler, root, save, init_jp):

    log.info(init_jp)
    arm_handler.move_jp(init_jp).wait()
    init_roll = init_jp[3]
    init_pitch = init_jp[4]
    # init_yaw = init_jp[5]

    # Create hysteresis motion
    yaw_traj = DvrkMotions.generate_yaw_motion()
    yaw_traj = go_and_come_back(yaw_traj)
    yaw_traj = list(product([init_roll], [init_pitch], yaw_traj))

    hysteresis_collection("yaw", init_jp.copy(), arm_handler, expected_spheres, save, root=root, trajectory=yaw_traj)


def hysteresis_collection(
    joint_name,
    init_jp,
    psm_handler,
    expected_markers=4,
    save: bool = False,
    root: Path = None,
    trajectory: List = None,
):
    """Perform a trajectory using the last three joints only. This function can be used to
        collect the data to identify the pitch axis w.r.t. the marker. The motions to identify the pitch
        axis are a outer roll (shaft movement) swing and  a wrist pitch swing.

    Args:
        init_jp ([type]): [description]
        psm_handler ([type]): [description]
        log ([type]): [description]
        expected_markers (int, optional): [description]. Defaults to 4.
        save (bool, optional): [description]. Defaults to True.
        filename (Path, optional): [description]. Defaults to None.
        trajectory (List): List of joints values that the robot will follow. Each point in the trajectory
        needs to specify q4, q5 and q6.
    """

    pitch_trajectory = DvrkMotions.generate_pitch_motion()
    roll_trajectory = [0.2, -0.3]
    total = len(roll_trajectory) * len(pitch_trajectory)

    ftk_handler = ftk_500("custom_marker_112")

    # Create record to store the measured data
    cp_filename = root / f"{joint_name}_hysteresis_cp.txt"
    jp_filename = root / f"{joint_name}_hysteresis_jp.txt"
    # sensor_record = AtracsysCartesianRecord(ftk_handler, expected_markers=expected_markers, filename=filename)
    cp_record = RobotSensorCartesianRecord(psm_handler, ftk_handler, expected_markers, cp_filename)
    jp_record = JointRecord(psm_handler, jp_filename)

    # Move to initial position
    psm_handler.move_jp(init_jp).wait()
    time.sleep(1)
    # Start trajectory
    counter = 0
    for q4, q5, q6 in track(trajectory, description="-- Trajectory Progress -- "):
        counter += 1
        # Move only wrist joints
        log.info(f"q4 {q4:+.4f} q5 {q5:+.4f} q6 {q6:+.4f}")
        init_jp[3] = q4
        init_jp[4] = q5
        init_jp[5] = q6
        q7 = psm_handler.jaw_jp()
        psm_handler.move_jp(init_jp).wait()
        time.sleep(0.15)

        cp_record.create_new_entry(counter, init_jp, q7)
        jp_record.create_new_entry(counter)

    if save:
        cp_record.to_csv(safe_save=False)
        jp_record.to_csv(safe_save=False)


def main():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("-t","--testid",type=int, default=7, help="testid required for test mode")
    parser.add_argument( "-r", "--root", type=str, default="data/03_replay_trajectory/d04-rec-13-traj01", 
                            help="root dir to save the data.")                     
    parser.add_argument( "-s", "--save", action='store_true',  default=False, 
                            help="Use this flag to Save recorded data") 
    args = parser.parse_args()
    # fmt: on

    # ------------------------------------------------------------
    # Script configuration
    # ------------------------------------------------------------
    root = Path(args.root)
    if not root.exists():
        log.info(f"root folder ({root}) does not exists")
        exit(0)
    root = root / "hysteresis_curves" / f"{args.testid:02d}"
    root.mkdir(parents=True, exist_ok=True)
    arm_namespace = "PSM2"
    rospy.init_node("dvrk_bag_replay", anonymous=True)

    # ------------------------------------------------------------
    # Get robot ready
    # ------------------------------------------------------------
    arm = ReplayDevice(device_namespace=arm_namespace, expected_interval=0.01)
    time.sleep(0.2)
    arm.home_device()
    # init_jp = arm.measured_jp()
    init_jp = np.array(RandomJointTrajectory.generate_random_joint())
    # init_jp = np.array([-0.1612, -0.2601, 0.1834, 0.1650, -0.2677, -0.4013])
    arm.move_jp(init_jp).wait()

    # Execute
    # save = args.save
    save = True
    log.info("Collection information")
    log.info(f"Data root dir:     {root}")
    log.info(f"Save data:         {save}")
    log.info(f"Test id:           {args.testid} ")

    ans = input('Press "y" to start data collection trajectory. Only replay trajectories that you know. ')
    if ans == "y":
        roll_hysteresis(arm, root, save, init_jp)
        pitch_hysteresis(arm, root, save, init_jp)
        yaw_hysteresis(arm, root, save, init_jp)
    else:
        exit(0)

    arm.move_jp(init_jp).wait()


if __name__ == "__main__":
    main()
