"""

Recording rosbag record with regex to capture the a subset of topics
- rosbag record -O test -e "/PSM2/(measured|setpoint).*"

Rosbag record regex do not use escape characters. The following command will not work as intended.
- rosbag record -O test -e "/PSM2/\(measured\|setpoint\).*

Util webpage to use rosbag package
http://wiki.ros.org/rosbag/Cookbook

Python rosbag module documentation
http://docs.ros.org/en/diamondback/api/rosbag/html/python/rosbag.bag.Bag-class.html

Interesting python package for rosbag manipulation
https://jmscslgroup.github.io/bagpy/installation.html

medium tutorial: https://rahulbhadani.medium.com/reading-ros-messages-from-a-bagfile-in-python-b006538bb520
"""
import rosbag
import rospy
from pathlib import Path
from kincalib.utils.Logger import Logger
from typing import List
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from kincalib.utils.Logger import Logger

log = Logger(__name__).log


class RosbagUtils:
    def __init__(self, path: Path) -> None:

        self.path = path
        self.name = path.name
        self.rosbag_handler = rosbag.Bag(path)

    def print_topics_info(self):
        n = len(self.rosbag_handler.get_type_and_topic_info()[1].values())
        types = []
        topics = list(self.rosbag_handler.get_type_and_topic_info()[1].keys())
        topics_headers = list(self.rosbag_handler.get_type_and_topic_info()[1].values())

        for i in range(0, n):
            types.append(topics_headers[i][0])

        log.info(f"Topics available in {self.name}")
        for i in range(n):
            log.info(f"Name: {topics[i]:40s} type: {types[i]}")

    def read_messages(self, topics: List[str]):
        msg_dict = defaultdict(list)
        msg_dict_ts = defaultdict(list)
        setpoint_time_previous = 0.0
        setpoints_out_of_order = 0

        for bag_topic, bag_message, t in self.rosbag_handler.read_messages():
            if bag_topic in topics:
                # check order of timestamps, drop if out of order
                setpoint_time = bag_message.header.stamp.to_sec()
                if setpoint_time <= setpoint_time_previous:
                    setpoints_out_of_order += 1

                msg_dict[bag_topic].append(bag_message)
                msg_dict_ts[bag_topic].append(t)

        if setpoints_out_of_order > 0:
            log.error(f"Messages out of order: {setpoints_out_of_order}")

        return msg_dict, msg_dict_ts

    def print_number_msg(self):
        topics = ["/PSM2/measured_js"]
        msg_dict, msg_dict_ts = self.read_messages(topics)

        for i in range(len(topics)):
            log.info(f"Number of messages in {topics[i]}: {len(msg_dict[topics[i]])} ")

    def save_crop_bag(self, min_ts: rospy.Time, max_ts, filename: Path = None):
        """Create a new rosbag containing only msg between min_ts and max_ts.
        If the filename is not specified, the following file name will be use:
        <path_to_current_rosbag>/<current_rosbag_name>_cropped.bag

        Args:
            min_ts ([type]): [description]
            max_ts ([type]): [description]
            filename ([Path], optional): [description]. Defaults to None.
        """
        if filename is None:
            filename = self.path.parent / (self.path.with_suffix("").name + "_cropped.bag")

        with rosbag.Bag(filename, "w") as outbag:
            for topic, msg, t in self.rosbag_handler.read_messages(start_time=min_ts, end_time=max_ts):
                outbag.write(topic, msg, t)

    @staticmethod
    def extract_joint_state_data(msg_list: List):
        pos = []
        vel = []
        for idx in range(len(msg_list)):
            msg = msg_list[idx]
            pos.append(msg.position)
            vel.append(msg.velocity)

        pos = np.array(pos)
        vel = np.array(vel)
        return pos, vel

    @staticmethod
    def extract_twist_data(msg_list: List):
        linear_vel = []
        for idx in range(len(msg_list)):
            msg = msg_list[idx]
            d = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]
            linear_vel.append(d)

        linear_vel = np.array(linear_vel)
        return linear_vel


if __name__ == "__main__":

    # ------------------------------------------------------------
    # Test rosbag utils class
    # ------------------------------------------------------------
    log = Logger("rosbag_utils").log
    root = Path("data/psm2_trajectories")
    file_p = root / "pitch_exp_traj_03_test_cropped.bag"

    rb = RosbagUtils(file_p)
    rb.print_topics_info()

    # ------------------------------------------------------------
    # Extract data
    # ------------------------------------------------------------
    topics = ["/PSM2/measured_js", "/PSM2/setpoint_js", "/PSM2/measured_cv"]
    msg_dict, msg_dict_ts = rb.read_messages(topics)

    for i in range(len(topics)):
        log.info(f"Number of messages in {topics[i]}: {len(msg_dict[topics[i]])} ")

    jp, jv = RosbagUtils.extract_joint_state_data(msg_dict[topics[0]])
    linear_vel = RosbagUtils.extract_twist_data(msg_dict[topics[2]])
