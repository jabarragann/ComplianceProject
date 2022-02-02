"""


Recording rosbag record with regex to capture the a subset of topics
- rosbag record -O test -e "/PSM2/(measured|setpoint).*"

Rosbag record regex do not use escape characters. The following command will not work as intended.
- rosbag record -O test -a  -e "/PSM2/\(measured\|setpoint\).*

Util webpage to use rosbag package
http://wiki.ros.org/rosbag/Cookbook

"""
import rosbag
from pathlib import Path
from kincalib.utils.Logger import Logger


class RosbagUtils:
    def __init__(self, path: Path, log=None) -> None:

        if log is None:
            self.log = Logger("ros_bag_log").log
        else:
            self.log = log

        self.name = path.name
        self.rosbag_handler = rosbag.Bag(path)

    def print_topics_info(self):
        n = len(self.rosbag_handler.get_type_and_topic_info()[1].values())
        types = []
        topics = list(self.rosbag_handler.get_type_and_topic_info()[1].keys())
        topics_headers = list(self.rosbag_handler.get_type_and_topic_info()[1].values())

        for i in range(0, n):
            types.append(topics_headers[i][0])

        self.log.info("Topics available")
        for i in range(n):
            self.log.info(f"Name: {topics[i]:40s} type: {types[i]}")

    def read_messages(self):
        pass


if __name__ == "__main__":

    # ------------------------------------------------------------
    # Test rosbag utils class
    # ------------------------------------------------------------
    root = Path("./data/psm2_trajectories/")
    file_p = root / "test.bag"

    rb = RosbagUtils(file_p)
    rb.print_topics_info()

    # ------------------------------------------------------------
