"""
Does the atracsys tells me what balls/detected pose correspond to which marker?
"""

from re import I
import rospy
import PyKDL
import std_msgs
import tf_conversions.posemath as pm
from sensor_msgs.msg import JointState
import geometry_msgs.msg
import time
import numpy as np
from typing import List
import scipy
from scipy.spatial import distance
from pathlib import Path

np.set_printoptions(precision=3, suppress=True)


class ftk_500:
    def __init__(self) -> None:
        # create node
        if not rospy.get_node_uri():
            rospy.init_node("ftk_500", anonymous=True, log_level=rospy.WARN)
        else:
            rospy.logdebug(rospy.get_caller_id() + " -> ROS already initialized")

        # suscriber
        self.__trajectory_j_ratio_sub = rospy.Subscriber(
            "/atracsys/Controller/measured_cp_array",
            geometry_msgs.msg.PoseArray,
            self.pose_array_callback,
        )

        self.poses_arr = []

    def pose_array_callback(self, msg):
        record = []
        for marker in range(len(msg.poses)):
            record.append(
                [
                    msg.poses[marker].position.x,
                    msg.poses[marker].position.y,
                    msg.poses[marker].position.z,
                ]
            )
        self.poses_arr = record

    def collect_measurements(
        self, m: int, t: float = 1000, sample_time: float = 50
    ) -> List[List[float]]:
        """collectect measurements from `m` markers for a specific amount of time.

        Args:
            m (int):  Expected number of markers
            t (float, optional): Number of mseconds collecting data. (ms)
            sample_time (float): Sample time in (ms)

        Returns:
            -List: List of measurements
            -markers_mismatch (int): number of times the detected markers did not match the expected number
        """

        init_time = time.time()
        markers_mismatch = 0
        recorded = []
        while time.time() - init_time < t / 1000:
            # Check number of markers
            if len(self.poses_arr) == m:
                recorded.append(self.poses_arr)
            else:
                markers_mismatch += 1
                # print("marker mismatch")
            time.sleep(sample_time / 1000)

        return recorded, markers_mismatch

    def sort_measurements(self, measurement_list: List[List[float]]) -> np.ndarray:
        if len(measurement_list) < 10:
            print("Not enough records (10 minimum)")  # this is totally arbitrary
            return None

        # Each record has n poses but we don't know if they are sorted by markers
        # create n lists to store the pose of each marker based on distance, using the last record as reference
        reference = measurement_list.pop()
        # create a records with markers sorted by proximity to reference order
        sorted_records = []
        sorted_records.append(reference)
        # iterate through rest of list
        for record in measurement_list:
            correspondence = []  # index of closest reference
            # find correspondence
            for marker in record:
                # find closest reference
                min = 100000.0  # arbitrary high
                closest_to_reference = -1
                for index_reference in range(0, len(reference)):
                    distance = scipy.spatial.distance.euclidean(marker, reference[index_reference])
                    if distance < min:
                        min = distance
                        closest_to_reference = index_reference
                correspondence.append(closest_to_reference)
            # create sorted record
            sorted_record = record  # just to make sure we have the correct size
            for index in correspondence:
                sorted_record[correspondence[index]] = record[index]
            sorted_records.append(sorted_record)

        return np.array(sorted_records)


if __name__ == "__main__":

    ftk_handler = ftk_500()

    print(__name__)

    # Obtain and sort measurements
    # measurements, miss_match = ftk_handler.collect_measurements(2, t=10000, sample_time=20)
    # sorted_cp = ftk_handler.sort_measurements(measurements)

    # print(sorted_cp.shape)
    # print(f"collected samples {sorted_cp.shape[0]}")
    # print(f"mismatches {miss_match}")
    # print(sorted_cp[:3, :, :])

    # filename = Path("./../atracsys_recordings/single_ball.npy")
    # np.save(filename, sorted_cp)

    # Plot data
    p = Path("./data/atracsys_recordings/single_ball.npy")
    data = np.load(p)
    print(data[0])
