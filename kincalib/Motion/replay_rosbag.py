"""
Rosbag replay script for PSM arm
"""
from __future__ import annotations
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
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.Logger import Logger
import tf_conversions.posemath as pm


# simplified arm class to replay motion, better performance than
# dvrk.arm since we're only subscribing to topics we need
class replay_device:

    # simplified jaw class to control the jaws, will not be used without the -j option
    class __jaw_device:
        def __init__(self, jaw_namespace, expected_interval, operating_state_instance):
            self.__crtk_utils = crtk.utils(
                self, jaw_namespace, expected_interval, operating_state_instance
            )
            self.__crtk_utils.add_move_jp()
            self.__crtk_utils.add_servo_jp()

    def __init__(self, device_namespace, expected_interval):
        # populate this class with all the ROS topics we need
        self.crtk_utils = crtk.utils(self, device_namespace, expected_interval)
        self.crtk_utils.add_operating_state()
        self.crtk_utils.add_servo_jp()
        self.crtk_utils.add_move_jp()
        self.crtk_utils.add_servo_cp()
        self.crtk_utils.add_move_cp()
        self.jaw = self.__jaw_device(
            device_namespace + "/jaw", expected_interval, operating_state_instance=self
        )


class RosbagReplay:
    """
    Rosbag replay class.

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
            self.log.info(
                "-- Found and removed %i out of order setpoints" % (self.out_of_order_counter)
            )

        # convert to mm
        bbmin = self.bbmin * 1000.0
        bbmax = self.bbmax * 1000.0
        self.log.info(
            "-- Range of motion in mm:\n   X:[%f, %f]\n   Y:[%f, %f]\n   Z:[%f, %f]"
            % (bbmin[0], bbmax[0], bbmin[1], bbmax[1], bbmin[2], bbmax[2])
        )

        # compute duration
        duration = (
            self.setpoints[-1].header.stamp.to_sec() - self.setpoints[0].header.stamp.to_sec()
        )
        self.log.info("-- Duration of trajectory: %f seconds" % (duration))

        # Number of poses
        self.log.info(
            "-- Found %i setpoints using topic %s" % (len(self.setpoints), self.setpoint_js_t)
        )
        if len(self.setpoints) == 0:
            self.log.error("-- No trajectory found!")

    def execute_trajectory(self, replay_device: RosbagReplay):
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

    def execute_measure(self, replay_device: RosbagReplay):
        """Stop every `n` number of setpoints to take a measurement while replaying the trajectory.

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
            if index % 20 == 0:
                replay_device.move_jp(numpy.array(self.setpoints[index].position)).wait()
                time.sleep(0.5)
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


if __name__ == "__main__":
    # ------------------------------------------------------------
    # ROS setup
    # ------------------------------------------------------------
    rospy.init_node("dvrk_bag_replay", anonymous=True)
    argv = rospy.myargv(argv=sys.argv)

    # ------------------------------------------------------------
    # Create replay class
    # ------------------------------------------------------------
    root = Path("temp/bags/")
    file_p = root / "pitch_exp_traj_01_test_cropped.bag"
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

    arm = replay_device(device_namespace=arm_name, expected_interval=0.01)
    arm = dvrk.arm(arm_name=arm_name, expected_interval=0.01)
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
    # replay.execute_trajectory(arm)
    replay.execute_measure(arm)
