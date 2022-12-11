# Python imports
from itertools import product
from pathlib import Path
import argparse
import sys
import numpy as np
from pathlib import Path
from kincalib.Motion.DvrkMotions import DvrkMotions
from kincalib.Motion.ReplayDevice import ReplayDevice

# ROS and DVRK imports
from kincalib.Recording.DataRecorder import DataRecorder
from kincalib.Sensors.ftk_500_api import ftk_500
import rospy

# kincalib module imports
from kincalib.utils.Logger import Logger
from kincalib.Motion.TrajectoryPlayer import Trajectory, TrajectoryPlayer

log = Logger("collection").log


def outer_joint_calibration(q3, q4, q5, q6, reverse: bool = True) -> Trajectory:
    outer_yaw_trajectory = DvrkMotions.generate_outer_yaw(steps=25)
    outer_pitch_trajectory = DvrkMotions.generate_outer_pitch(steps=25)

    if reverse:
        outer_yaw_trajectory = outer_yaw_trajectory[::-1]
        outer_pitch_trajectory = outer_pitch_trajectory[::-1]

    outer_yaw_trajectory = list(product(outer_yaw_trajectory, [0.0], [q3], [q4], [q5], [q6]))
    # Generate outer pitch trajectory
    outer_pitch_trajectory = list(product([0.0], outer_pitch_trajectory, [q3], [q4], [q5], [q6]))

    # traj_setpoints = outer_yaw_trajectory + outer_pitch_trajectory
    traj_setpoints = outer_pitch_trajectory + outer_yaw_trajectory
    traj_setpoints = np.array(traj_setpoints)

    trajectory = Trajectory.from_numpy(joint_array=traj_setpoints)

    return trajectory


def main():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument( "-r", "--root", type=str, default="data/05_calib_exp/rec00", 
                            help="root dir to save the data.")                     
    parser.add_argument( "-s", "--save", action='store_true',  default=False, 
                            help="Use this flag to Save recorded data") 
    parser.add_argument("-t","--testid",type=int, help="testid")
    args = parser.parse_args()
    # fmt: on

    # ------------------------------------------------------------
    # Script configuration
    # ------------------------------------------------------------
    root = Path(args.root)
    if not root.exists():
        root.mkdir()
        log.info(f"creating root: {root}")

    expected_spheres = 4
    marker_name = "custom_marker_113"
    arm_namespace = "PSM2"
    testid = 7
    rospy.init_node("dvrk_bag_replay", anonymous=True)

    # ------------------------------------------------------------
    # Get robot and sensor ready
    # ------------------------------------------------------------
    ftk_handler = ftk_500(marker_name)
    arm = ReplayDevice(device_namespace=arm_namespace, expected_interval=0.01)

    # make sure the arm is powered
    print("-- Enabling arm")
    if not arm.enable(10):
        sys.exit("-- Failed to enable within 10 seconds")
    print("-- Homing arm")
    if not arm.home(10):
        sys.exit("-- Failed to home within 10 seconds")

    input('---> Make sure arm is ready to move. Press "Enter" to move to start position')

    init_jp = arm.measured_jp()
    init_cp = arm.measured_cp()

    print(init_jp)

    # ------------------------------------------------------------
    # Create trajectory
    # ------------------------------------------------------------
    trajectory = outer_joint_calibration(q3=init_jp[2], q4=init_jp[3], q5=init_jp[4], q6=init_jp[5])
    # trajectory = outer_joint_calibration(q3=0.0, q4=0.0, q5=0.0, q6=0.0)
    log.info(f"Trajectory size {len(trajectory)}")

    # ------------------------------------------------------------
    # Config trajectory player
    # ------------------------------------------------------------
    # Callbacks
    data_recorder_cb = DataRecorder(
        arm,
        ftk_handler,
        expected_markers=expected_spheres,
        root=root,
        marker_name=marker_name,
        mode="test",
        test_id=testid,
        description="with delay of 100ms - reversed",
    )

    trajectory_player = TrajectoryPlayer(arm, trajectory, before_motion_loop_cb=[], after_motion_cb=[data_recorder_cb])
    # trajectory_player = TrajectoryPlayer(arm, trajectory, before_motion_loop_cb=[], after_motion_cb=[])

    # Execute
    log.info("Collection information")
    log.info(f"Data root dir:     {args.root}")
    log.info(f"Save data:         {args.save}")
    log.info(f"Trajectory length: {len(trajectory)}")

    ans = input('Press "y" to start data collection trajectory. Only replay trajectories that you know. ')
    if ans == "y":
        trajectory_player.replay_trajectory(execute_cb=True, delay=0.1)
    else:
        exit(0)


if __name__ == "__main__":
    main()
