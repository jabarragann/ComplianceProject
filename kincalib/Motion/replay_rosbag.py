"""
Rosbag replay script for PSM arm
OLD FILE THIS CAN BE REMOVED. IT WAS DEPRECATED FOR TrajectoryPlayer.py
"""
from __future__ import annotations
from random import sample
import dvrk
import crtk
import sys
import time
import rospy
import rosbag
import numpy as np
import numpy
import PyKDL
import argparse
from pathlib import Path
import tf_conversions.posemath as pm
import pandas as pd
from rich.logging import RichHandler
from rich.progress import track

from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.Logger import Logger
from kincalib.Sensors.ftk_500_api import ftk_500
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.Motion.CalibrationMotions import CalibrationMotions
from kincalib.Motion.ReplayDevice import ReplayDevice


class RosbagReplay:
    """
    Class to replay trajectories with the dvrk.

    For now only replay a rosbag from setpoint_js
    - /PSM2/setpoint_js

    Adapted from https://github.com/jhu-dvrk/dvrk-ros/blob/devel/dvrk_python/scripts/dvrk_bag_replay.py
    """

    def __init__(self, filename: Path, log=None) -> None:

        self.filename = filename

        if log is None:
            self.log = Logger("ros_bag_log").log
        else:
            self.log = log
        self.rosbag_utils = RosbagUtils(filename, log=self.log)
        self.rosbag = self.rosbag_utils.rosbag_handler

        # parse bag and create list of points
        self.bbmin = np.zeros(3)
        self.bbmax = np.zeros(3)
        self.last_message_time = 0.0
        self.out_of_order_counter = 0
        self.setpoints = []

        self.setpoint_js_t = "/PSM2/setpoint_js"
        self.setpoint_cp_t = "/PSM2/setpoint_cp"

    def collect_trajectory_setpoints(self):

        self.log.info("-- Parsing bag %s" % (self.rosbag_utils.name))
        for bag_topic, bag_message, t in self.rosbag.read_messages():
            # Collect setpoint_cp only to keep track of workspace
            if bag_topic == self.setpoint_cp_t:
                # check order of timestamps, drop if out of order
                transform_time = bag_message.header.stamp.to_sec()
                if transform_time <= self.last_message_time:
                    self.out_of_order_counter = self.out_of_order_counter + 1
                else:
                    # keep track of workspace
                    position = numpy.array(
                        [
                            bag_message.pose.position.x,
                            bag_message.pose.position.y,
                            bag_message.pose.position.z,
                        ]
                    )
                    if len(self.setpoints) == 1:
                        self.bbmin = position
                        self.bmax = position
                    else:
                        self.bbmin = numpy.minimum(self.bbmin, position)
                        self.bbmax = numpy.maximum(self.bbmax, position)
            elif bag_topic == self.setpoint_js_t:
                # check order of timestamps, drop if out of order
                transform_time = bag_message.header.stamp.to_sec()
                if transform_time <= self.last_message_time:
                    self.out_of_order_counter = self.out_of_order_counter + 1
                else:
                    self.setpoints.append(bag_message)

    def trajectory_report(self):
        self.log.info("Trajectory report:")
        # report out of order setpoints
        if self.out_of_order_counter > 0:
            self.log.info("-- Found and removed %i out of order setpoints" % (self.out_of_order_counter))

        # convert to mm
        bbmin = self.bbmin * 1000.0
        bbmax = self.bbmax * 1000.0
        self.log.info(
            "-- Range of motion in mm:\n   X:[%f, %f]\n   Y:[%f, %f]\n   Z:[%f, %f]"
            % (bbmin[0], bbmax[0], bbmin[1], bbmax[1], bbmin[2], bbmax[2])
        )

        # compute duration
        duration = self.setpoints[-1].header.stamp.to_sec() - self.setpoints[0].header.stamp.to_sec()
        self.log.info("-- Duration of trajectory: %f seconds" % (duration))

        # Number of poses
        self.log.info("-- Found %i setpoints using topic %s" % (len(self.setpoints), self.setpoint_js_t))
        if len(self.setpoints) == 0:
            self.log.error("-- No trajectory found!")

    def add_measurement(self, df_vals_cp, df_vals_jp, replay_device, ftk_handler, expected_spheres, index):
        # Measure
        jp = replay_device.measured_jp()
        jaw_jp = replay_device.jaw.measured_jp()[0]
        # -------------------------------------
        # Add sensor cp measurements
        # -------------------------------------
        new_pt = CalibrationMotions.create_df_with_measurements(ftk_handler, expected_spheres, index, jp, jaw_jp)
        df_vals_cp = df_vals_cp.append(new_pt)
        # -------------------------------------
        # Add robot cp measurements
        # -------------------------------------
        new_pt = CalibrationMotions.create_df_with_robot_cp(replay_device, index, jp, jaw_jp)
        df_vals_cp = df_vals_cp.append(new_pt)
        # -------------------------------------
        # Add robot cp and jp measurements
        # -------------------------------------
        new_pt = CalibrationMotions.create_df_with_robot_jp(replay_device, index)
        df_vals_jp = df_vals_jp.append(new_pt)

        return df_vals_cp, df_vals_jp

    def execute_trajectory(self, replay_device: RosbagReplay):
        """Normal trajectory execution based on Anton example.

        Args:
            replay_device (RosbagReplay): [description]
        """
        last_bag_time = self.setpoints[0].header.stamp.to_sec()
        counter = 0
        total = len(self.setpoints)

        start_time = time.time()
        # for the replay, use the jp/cp setpoint for the arm to control the
        # execution time.  Jaw positions are picked in order without checking
        # time.  There might be better ways to synchronized the two
        # sequences...
        for index in range(total):
            # record start time
            loop_start_time = time.time()
            # compute expected dt
            new_bag_time = self.setpoints[index].header.stamp.to_sec()
            delta_bag_time = new_bag_time - last_bag_time
            last_bag_time = new_bag_time
            # replay
            replay_device.servo_jp(numpy.array(self.setpoints[index].position))
            # update progress
            counter = counter + 1
            sys.stdout.write("\r-- Progress %02.1f%%" % (float(counter) / float(total) * 100.0))
            sys.stdout.flush()
            # try to keep motion synchronized
            loop_end_time = time.time()
            sleep_time = delta_bag_time - (loop_end_time - loop_start_time)
            # if process takes time larger than console rate, don't sleep
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("\n--> Time to replay trajectory: %f seconds" % (time.time() - start_time))
        print("--> Done!")

    def execute_measure(
        self,
        replay_device: RosbagReplay,
        filename: Path,
        marker_name: str,
        expected_spheres: int,
        save=False,
    ):
        """Stop every `n` number of setpoints to take a measurement while replaying the trajectory.
        Measurements will be saved in two different dataframes: a pose (cp) df (markers,fiducials, and robot
        end-effector pose) and a joint position (jp) df (joint position of the robot.)

        Args:
            replay_device (RosbagReplay): replay device handler.
            filename (Path): filename for the the pose_df and joint_df. The files will be saved as 'filename_pose.txt' and
            'filename_jp.txt'.
            marker_name (str): Marker name.
            expected_spheres (int, optional): Expected number of fiducials. Defaults to 3.
            save (bool, optional): [description]. Defaults to False.
        """
        last_bag_time = self.setpoints[0].header.stamp.to_sec()
        counter = 0
        total = len(self.setpoints)
        start_time = time.time()

        if not filename.exists:
            filename.mkdir(parents=True)

        # Create ftk handler
        ftk_handler = ftk_500(marker_name=marker_name)
        # Create data frames
        df_vals_cp = pd.DataFrame(columns=CalibrationMotions.df_cols_cp)
        df_vals_jp = pd.DataFrame(columns=CalibrationMotions.df_cols_jp)

        for index in track(range(total), "-- Trajectory Progress -- "):
            # record start time
            loop_start_time = time.time()
            # compute expected dt
            new_bag_time = self.setpoints[index].header.stamp.to_sec()
            delta_bag_time = new_bag_time - last_bag_time
            last_bag_time = new_bag_time
            if index % 20 == 0:
                # Move
                replay_device.move_jp(numpy.array(self.setpoints[index].position)).wait()

                marker_pose, fiducials_pose = ftk_handler.obtain_processed_measurement(
                    expected_spheres, t=200, sample_time=15
                )
                if marker_pose is not None:
                    df_vals_cp, df_vals_jp = self.add_measurement(
                        df_vals_cp, df_vals_jp, replay_device, ftk_handler, expected_spheres, index
                    )
                else:
                    self.log.warning(f"No marker detected at step {index}")

            # try to keep motion synchronized
            loop_end_time = time.time()
            sleep_time = delta_bag_time - (loop_end_time - loop_start_time)
            # if process takes time larger than console rate, don't sleep
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Save data frames
        cp_filename = filename.parent / (filename.with_suffix("").name + "_cp.txt")
        jp_filename = filename.parent / (filename.with_suffix("").name + "_jp.txt")
        save_without_overwritting(df_vals_cp, cp_filename)
        save_without_overwritting(df_vals_jp, jp_filename)

        self.log.info("Time to replay trajectory: %f seconds" % (time.time() - start_time))

    def collect_calibration_data(
        self,
        replay_device: RosbagReplay,
        root_dir: Path,
        marker_name: str,
        expected_spheres: int,
    ):
        """Stop every `n` number of setpoints to take a measurement while replaying the trajectory.
        Measurements will be saved in two different dataframes: a pose (cp) df (markers,fiducials, and robot
        end-effector pose) and a joint position (jp) df (joint position of the robot.)

        Args:
            replay_device (RosbagReplay): replay device handler.
            filename (Path): filename for the the pose_df and joint_df. The files will be saved as 'filename_pose.txt' and
            'filename_jp.txt'.
            marker_name (str): Marker name.
            expected_spheres (int, optional): Expected number of fiducials. Defaults to 3.
            save (bool, optional): [description]. Defaults to False.
        """
        outer_js_files = root_dir / "outer_mov"
        wrist_files = root_dir / "pitch_roll_mov"
        robot_files = root_dir / "robot_mov"
        cp_filename = robot_files / ("robot_cp.txt")
        jp_filename = robot_files / ("robot_jp.txt")
        if not root_dir.exists():
            root_dir.mkdir(parents=True)
        if not outer_js_files.exists():
            outer_js_files.mkdir(parents=True)
        if not wrist_files.exists():
            wrist_files.mkdir(parents=True)
        if not robot_files.exists():
            robot_files.mkdir(parents=True)

        last_bag_time = self.setpoints[0].header.stamp.to_sec()
        total = len(self.setpoints)
        start_time = time.time()

        # Create ftk handler
        ftk_handler = ftk_500(marker_name=marker_name)

        # Create data frames
        df_vals_cp = pd.DataFrame(columns=CalibrationMotions.df_cols_cp)
        df_vals_jp = pd.DataFrame(columns=CalibrationMotions.df_cols_jp)

        # ------------------------------------------------------------
        # Collect outer joints calibration data
        # ------------------------------------------------------------
        CalibrationMotions.outer_pitch_yaw_motion(
            replay_device.measured_jp(), psm_handler=replay_device, expected_markers=4, save=True, root=outer_js_files
        )

        # ------------------------------------------------------------
        # Collect wrist calibration data
        # ------------------------------------------------------------
        # for index in track(range(total), "-- Trajectory Progress -- "):
        for index in range(total):
            # record start time
            loop_start_time = time.time()
            # compute expected dt
            new_bag_time = self.setpoints[index].header.stamp.to_sec()
            delta_bag_time = new_bag_time - last_bag_time
            last_bag_time = new_bag_time
            # replay
            if index % 40 == 0:
                self.log.info(f"-- Trajectory Progress --> {100*index/total:0.02f} %")
                # Move
                replay_device.move_jp(numpy.array(self.setpoints[index].position)).wait()

                marker_pose, fiducials_pose = ftk_handler.obtain_processed_measurement(
                    expected_spheres, t=200, sample_time=15
                )
                if marker_pose is not None:
                    df_vals_cp, df_vals_jp = self.add_measurement(
                        df_vals_cp, df_vals_jp, replay_device, ftk_handler, expected_spheres, index
                    )
                    # Save values
                    df_vals_cp.to_csv(robot_files / ("robot_cp_temp.txt"), index=None)
                    df_vals_jp.to_csv(robot_files / ("robot_jp_temp.txt"), index=None)

                    # ------------------------------------------------------------
                    # Wrist motions
                    # ------------------------------------------------------------
                    init_jp = replay_device.measured_jp()
                    CalibrationMotions.pitch_yaw_roll_independent_motion(
                        init_jp,
                        psm_handler=replay_device,
                        expected_markers=4,
                        save=True,
                        filename=wrist_files / f"step{index:03d}_wrist_motion.txt",
                    )
                else:
                    self.log.warning("No marker pose detected.")

            # try to keep motion synchronized
            loop_end_time = time.time()
            sleep_time = delta_bag_time - (loop_end_time - loop_start_time)
            # if process takes time larger than console rate, don't sleep
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Save data frames
        save_without_overwritting(df_vals_cp, cp_filename)
        save_without_overwritting(df_vals_jp, jp_filename)

        self.log.info("Time to replay trajectory: %f seconds" % (time.time() - start_time))


if __name__ == "__main__":
    # ------------------------------------------------------------
    # ROS setup
    # ------------------------------------------------------------
    rospy.init_node("dvrk_bag_replay", anonymous=True)
    argv = rospy.myargv(argv=sys.argv)

    # ------------------------------------------------------------
    # Create replay class
    # ------------------------------------------------------------
    root = Path("data/psm2_trajectories/")
    file_p = root / "pitch_exp_traj_03_test_cropped.bag"
    replay = RosbagReplay(file_p)
    replay.rosbag_utils.print_topics_info()

    # ------------------------------------------------------------
    # Extract trajectory and print summary
    # - Expected topic to use: /<arm>/setpoint_js
    # ------------------------------------------------------------
    arm_name = "PSM2"
    replay.collect_trajectory_setpoints()
    replay.trajectory_report()

    # ------------------------------------------------------------
    # Get robot ready
    # ------------------------------------------------------------

    arm = ReplayDevice(device_namespace=arm_name, expected_interval=0.01)
    arm.measured_cp()
    setpoints = replay.setpoints

    # make sure the arm is powered
    print("-- Enabling arm")
    if not arm.enable(10):
        sys.exit("-- Failed to enable within 10 seconds")

    print("-- Homing arm")
    if not arm.home(10):
        sys.exit("-- Failed to home within 10 seconds")

    input('---> Make sure arm is ready to move. Press "Enter" to move to start position')

    # Create frame using first pose and use blocking move
    arm.move_jp(numpy.array(setpoints[0].position)).wait()
    input('-> Press "Enter" to replay trajectory')

    # ------------------------------------------------------------
    # Execute trajectory
    # ------------------------------------------------------------
    replay.execute_trajectory(arm)
