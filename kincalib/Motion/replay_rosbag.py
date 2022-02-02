"""
Rosbag replay script for PSM arm
"""

import dvrk
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


class RosbagReplay:
    """
    Rosbag replay class.

    Adapted from https://github.com/jhu-dvrk/dvrk-ros/blob/master/dvrk_python/scripts/dvrk_bag_replay.py
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
        self.poses = []

        self.topic = None

    def create_list_points(self, topic_name):
        self.topic = topic_name
        self.log.info("-- Parsing bag %s" % (self.rosbag_utils.name))
        for bag_topic, bag_message, t in self.rosbag.read_messages():
            if bag_topic == topic_name:
                # check order of timestamps, drop if out of order
                transform_time = bag_message.header.stamp.to_sec()
                if transform_time <= self.last_message_time:
                    self.out_of_order_counter = self.out_of_order_counter + 1
                else:
                    # append message
                    self.poses.append(bag_message)
                    # keep track of workspace
                    position = numpy.array(
                        [
                            bag_message.pose.position.x,
                            bag_message.pose.position.y,
                            bag_message.pose.position.z,
                        ]
                    )
                    if len(self.poses) == 1:
                        self.bbmin = position
                        self.bmax = position
                    else:
                        self.bbmin = numpy.minimum(self.bbmin, position)
                        self.bbmax = numpy.maximum(self.bbmax, position)

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
        duration = self.poses[-1].header.stamp.to_sec() - self.poses[0].header.stamp.to_sec()
        self.log.info("-- Duration of trajectory: %f seconds" % (duration))

        # Number of poses
        self.log.info("-- Found %i setpoints using topic %s" % (len(self.poses), self.topic))
        if len(self.poses) == 0:
            self.log.error("-- No trajectory found!")


if __name__ == "__main__":
    # ------------------------------------------------------------
    # ROS setup
    # ------------------------------------------------------------
    rospy.init_node("dvrk_bag_replay", anonymous=True)
    argv = rospy.myargv(argv=sys.argv)

    # ------------------------------------------------------------
    # Create replay class
    # ------------------------------------------------------------
    root = Path("./data/psm2_trajectories/")
    file_p = root / "test_trajectory1.bag"
    replay = RosbagReplay(file_p)
    replay.rosbag_utils.print_topics_info()

    # ------------------------------------------------------------
    # Extract trajectory and print summary
    # - Expected topic to use: /<arm>/setpoint_cp
    # ------------------------------------------------------------
    arm_name = "PSM2"
    topic_name = "/setpoint_cp"
    topic_name = "/" + arm_name + topic_name
    replay.create_list_points(topic_name)
    replay.trajectory_report()

    # ------------------------------------------------------------
    # Execute trajectory
    # ------------------------------------------------------------
    arm = dvrk.arm(arm_name=arm_name, expected_interval=0.01)
    poses = replay.poses

    # make sure the arm is powered
    print("-- Enabling arm")
    if not arm.enable(10):
        sys.exit("-- Failed to enable within 10 seconds")

    print("-- Homing arm")
    if not arm.home(10):
        sys.exit("-- Failed to home within 10 seconds")

    input(
        "---> Make sure the arm is ready to move using cartesian positions.  "
        "For a PSM or ECM, you need to have a tool in place and the tool tip needs "
        'to be outside the cannula.  You might have to manually adjust your arm.  Press "\Enter" when the arm is ready.'
    )

    input('---> Press "Enter" to move to start position')

    # Create frame using first pose and use blocking move
    cp = PyKDL.Frame()
    cp.p = PyKDL.Vector(
        poses[0].pose.position.x,
        poses[0].pose.position.y,
        poses[0].pose.position.z,
    )
    cp.M = PyKDL.Rotation.Quaternion(
        poses[0].pose.orientation.x,
        poses[0].pose.orientation.y,
        poses[0].pose.orientation.z,
        poses[0].pose.orientation.w,
    )
    arm.move_cp(cp).wait()

# # Replay
# input('---> Press "Enter" to replay trajectory')

# last_time = poses[0].header.stamp.to_sec()

# counter = 0
# total = len(poses)
# start_time = time.time()

# for pose in poses:
#     # crude sleep to emulate period.  This doesn't take into account
#     # time spend to compute cp from pose and publish so this will
#     # definitely be slower than recorded trajectory
#     new_time = pose.header.stamp.to_sec()
#     time.sleep(new_time - last_time)
#     last_time = new_time
#     cp.p = PyKDL.Vector(
#         pose.transform.translation.x, pose.transform.translation.y, pose.transform.translation.z
#     )
#     cp.M = PyKDL.Rotation.Quaternion(
#         pose.transform.rotation.x,
#         pose.transform.rotation.y,
#         pose.transform.rotation.z,
#         pose.transform.rotation.w,
#     )
#     arm.servo_cp(cp)
#     counter = counter + 1
#     sys.stdout.write("\r-- Progress %02.1f%%" % (float(counter) / float(total) * 100.0))
#     sys.stdout.flush()

# print("\n--> Time to replay trajectory: %f seconds" % (time.time() - start_time))
# print("--> Done!")
