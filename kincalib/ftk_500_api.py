"""
Does the atracsys tells me what balls/detected pose correspond to which marker?
"""

from re import I

from sympy import O
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
import matplotlib.pyplot as plt
from kincalib.utils.Logger import Logger
import sys

np.set_printoptions(precision=4, suppress=True)


class ftk_500:
    def __init__(self, marker_name: str = None) -> None:
        # Init variables
        self.marker_name = None

        # Create node
        if not rospy.get_node_uri():
            rospy.init_node("ftk_500", anonymous=True, log_level=rospy.WARN)
        else:
            rospy.logdebug(rospy.get_caller_id() + " -> ROS already initialized")

        # subscribers
        self.__trajectory_j_ratio_sub = rospy.Subscriber(
            "/atracsys/Controller/measured_cp_array",
            geometry_msgs.msg.PoseArray,
            self.pose_array_callback,
        )
        if marker_name is not None:
            self.marker_name = marker_name
            self.marker_pose = None
            self.__marker_subs_list = rospy.Subscriber(
                "/atracsys/" + marker_name + "/measured_cp",
                geometry_msgs.msg.PoseStamped,
                self.marker_pose_callback,
            )

        self.poses_arr = []

    def marker_pose_callback(self, msg):
        self.marker_pose = pm.fromMsg(msg.pose)

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
        """Collectect measurements from `m` fiducials for a specific amount of time.
        Marker pose will also be collected if it is provided when initializing the class.

        Output dict keys:
        * "fiducials": recorded,
        * "fiducials_dropped": records_removed,
        * "marker": marker_pose_arr,
        * "markers_dropped": markers_dropped,

        Args:
            m (int):  Expected number of markers
            t (float, optional): Number of mseconds collecting data. (ms)
            sample_time (float): Sample time in (ms)

        Returns:
            measurement_dict (dict): Dictionary with measurements
        """

        init_time = time.time()
        records_removed = 0
        recorded = []
        marker_pose_arr = []
        markers_dropped = 0
        self.marker_pose = None
        while time.time() - init_time < t / 1000:
            # Collect fiducials pose if they match the expected number
            if len(self.poses_arr) == m:
                recorded.append(self.poses_arr)
            else:
                records_removed += 1
                # print("marker mismatch")
            # Collect marker pose if available
            if self.marker_name is not None:
                if self.marker_pose is not None:
                    marker_pose_arr.append(self.marker_pose)
                else:
                    markers_dropped += 1

            time.sleep(sample_time / 1000)

        return {
            "fiducials": recorded,
            "fiducials_dropped": records_removed,
            "markers": marker_pose_arr,
            "markers_dropped": markers_dropped,
        }

    @staticmethod
    def sort_measurements(measurement_list: List[List[float]]) -> np.ndarray:
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

    @staticmethod
    def average_marker_pose(pose_arr):
        if len(pose_arr) < 10:
            print("Not enough records (10 minimum)")  # this is totally arbitrary
            return None, None, None
        position = []
        orientation = []
        for k in range(len(pose_arr)):
            position.append(np.array(list(pose_arr[k].p)))
            orientation.append(np.array(list(pose_arr[k].M.GetQuaternion())))

        position_mean = np.array(position).mean(axis=0)
        orientation_mean = np.array(orientation).mean(axis=0)
        position_std = np.array(position).std(axis=0)
        orientation_std = np.array(orientation).std(axis=0)
        orientation_mean = orientation_mean / np.linalg.norm(orientation_mean)

        mean_frame = PyKDL.Frame(
            PyKDL.Rotation.Quaternion(*orientation_mean), PyKDL.Vector(*position_mean)
        )
        return mean_frame, position_std, orientation_std


def clean_avg_measurements(measurements, expected_markers=1):
    if len(measurements) > 0:
        if expected_markers == 1:
            measurements = np.array(measurements)
            mean_value = measurements.squeeze().mean(axis=0)
            std_value = measurements.squeeze().std(axis=0)
            log.debug(f"sample values \n {measurements.squeeze()[:1, :]}")
            log.debug(f"mean value: {mean_value}")
            log.debug(f"std value:  {std_value}")
        elif expected_markers > 1:
            sorted = ftk_500.sort_measurements(measurements)
            if sorted is not None:
                mean_value = sorted.mean(axis=0)
                std_value = sorted.std(axis=0)
                log.debug(f"sample values \n {sorted.squeeze()[:1, :]}")
                log.debug(f"mean value:\n{mean_value}")
                log.debug(f"std value:\n{std_value}")
        else:
            print("expected markers needs to be a positive number")
            sys.exit(0)


if __name__ == "__main__":
    log = Logger("utils_log").log

    print(__name__)

    # ftk_handler = ftk_500()
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
    # p = Path("./data/atracsys_recordings/single_ball.npy")
    # data = np.load(p)
    # print(data.shape)
    # print(data[0:3])

    # fig, axis = plt.subplots(1, 3, sharey=True, tight_layout=True)
    # axis[0].hist(data[:, 0, 0], bins=15)
    # axis[1].hist(data[:, 0, 1], bins=15)
    # axis[2].hist(data[:, 0, 2], bins=15)
    # plt.show()

    # ------------------------------------------------------------
    # Obtain measurements - example
    # ------------------------------------------------------------
    input("Enter to collect data ")
    marker_name = "custom_marker_112"
    expected_markers = 4
    ftk_handler = ftk_500(marker_name=marker_name)

    measurement_dict = ftk_handler.collect_measurements(expected_markers, t=1000, sample_time=20)

    measure_list = measurement_dict["fiducials"]
    miss_match = measurement_dict["fiducials_dropped"]
    marker_list = measurement_dict["markers"]
    marker_dropped = measurement_dict["markers_dropped"]

    measurements = np.array(measure_list)
    log.debug(f"collected fiducials samples: {measurements.shape[0]}")
    log.debug(f"drop fiducials samples:      {miss_match}")
    log.debug(f"Measurement shape: {measurements.shape}")
    clean_avg_measurements(measure_list, expected_markers=expected_markers)

    log.debug(f"collected marker:  {len(marker_list)}")
    log.debug(f"Dropped markers:   {marker_dropped}")
    mean_frame, p_std, r_std = ftk_500.average_marker_pose(marker_list)

    if mean_frame is not None:
        log.debug(f"mean frame: \n {pm.toMatrix(mean_frame)}")
        log.debug(f"position std:\n{p_std}")
        log.debug(f"orientation std:\n{r_std}")
