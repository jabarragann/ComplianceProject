from __future__ import annotations

# Python
from pathlib import Path
from random import Random
import time
from typing import Any, Union
from dataclasses import dataclass, field
from black import out
import numpy as np
import numpy
from typing import List

from psutil import cpu_count

# ros
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

# Custom
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.Logger import Logger
from kincalib.Recording.DataRecord import CalibrationRecord
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.Logger import Logger
from kincalib.Sensors.ftk_500_api import ftk_500
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.Motion.CalibrationMotions import CalibrationMotions
from kincalib.Motion.ReplayDevice import ReplayDevice
from collections import namedtuple

log = Logger(__name__).log


class DataRecorder:
    """
    Callable data class to record data from robot and sensor.
    """

    def __init__(self, replay_device, expected_markers: int, root: Path, marker_name: str, mode: str, test_id: int):

        self.replay_device = replay_device
        self.expected_markers = expected_markers
        self.ftk_handler = ftk_500(marker_name)
        self.calibration_record = CalibrationRecord(
            ftk_handler=self.ftk_handler,
            robot_handler=self.replay_device,
            expected_markers=expected_markers,
            root_dir=root,
            mode=mode,
            test_id=test_id,
        )

    def __call__(self, **kwargs):
        self.collect_data(**kwargs)

    def collect_data(self, index=None):
        if index is None:
            log.error("specify index WristJointsCalibrationRecorder")
            exit(0)
        log.info("record data")
        marker_pose, fiducials_pose = self.ftk_handler.obtain_processed_measurement(
            self.expected_markers, t=200, sample_time=15
        )
        # Measure
        jp = self.replay_device.measured_jp()
        jaw_jp = self.replay_device.jaw.measured_jp()[0]
        # Check if marker is visible add sensor data
        if marker_pose is not None:
            self.calibration_record.create_new_sensor_entry(index, jp, jaw_jp)
        # Always add robot data
        self.calibration_record.create_new_robot_entry(index, jp, jaw_jp)

        self.calibration_record.to_csv(safe_save=False)


class OuterJointsCalibrationRecorder:
    """Class to perfom the calibration motion of the outer joints while recording the data from
    robot and sensor
    """

    def __init__(self, replay_device, save: bool, expected_markers: int, root: Path, marker_name: str):
        self.save = save
        self.replay_device = replay_device
        self.expected_markers = expected_markers

        self.outer_js_files = root / "outer_mov"
        if not self.outer_js_files.exists():
            self.outer_js_files.mkdir(parents=True)

    def __call__(self) -> Any:
        CalibrationMotions.outer_pitch_yaw_motion(
            self.replay_device.measured_jp(),
            psm_handler=self.replay_device,
            expected_markers=self.expected_markers,
            save=self.save,
            root=self.outer_js_files,
        )


class WristJointsCalibrationRecorder:
    """Class to perfom the calibration motion of the wrist joints while recording the data from
    robot and sensor
    """

    def __init__(self, replay_device, save: bool, expected_markers: int, root: Path, marker_name: str):
        self.save = save
        self.replay_device = replay_device
        self.expected_markers = expected_markers

        self.wrist_files = root / "pitch_roll_mov"
        if not self.wrist_files.exists():
            self.wrist_files.mkdir(parents=True)

    def __call__(self, index=None) -> Any:
        if index is None:
            log.error("specify index WristJointsCalibrationRecorder")
            exit(0)
        log.info("calibration")
        CalibrationMotions.pitch_yaw_roll_independent_motion(
            self.replay_device.measured_jp(),
            psm_handler=self.replay_device,
            expected_markers=self.expected_markers,
            save=self.save,
            filename=self.wrist_files / f"step{index:03d}_wrist_motion.txt",
        )


if __name__ == "__main__":

    from kincalib.Motion.TrajectoryPlayer import Trajectory, TrajectoryPlayer

    # Test calibration motions
    rosbag_path = Path("data/psm2_trajectories/pitch_exp_traj_01_test_cropped.bag")
    rosbag_handle = RosbagUtils(rosbag_path)
    trajectory = Trajectory.from_ros_bag(rosbag_handle, sampling_factor=60)

    log.info(f"Initial pt {trajectory.setpoints[0].position}")
    log.info(f"Starting ts {trajectory.setpoints[0].header.stamp.to_sec()}")
    log.info(f"number of points {len(trajectory)}")

    arm_namespace = "PSM2"
    arm = ReplayDevice(device_namespace=arm_namespace, expected_interval=0.01)
    arm.home_device()

    # Setup calibration callbacks
    outer_js_calib_cb = OuterJointsCalibrationRecorder(
        replay_device=arm, save=False, expected_markers=4, root=Path("."), marker_name="none"
    )
    wrist_js_calib_cb = WristJointsCalibrationRecorder(
        replay_device=arm, save=False, expected_markers=4, root=Path("."), marker_name="none"
    )
    data_recorder_cb = DataRecorder(arm, 4, root=Path("~/temp"), marker_name="test", mode="calib", test_id=11)

    # Create trajectory player with cb
    # trajectory_player = TrajectoryPlayer(
    #     arm, trajectory, before_motion_cb=[outer_js_calib_cb], after_motion_cb=[wrist_js_calib_cb]
    # )
    trajectory_player = TrajectoryPlayer(
        arm, trajectory, before_motion_loop_cb=[], after_motion_cb=[data_recorder_cb, wrist_js_calib_cb]
    )

    ans = input('Press "y" to start data collection trajectory. Only replay trajectories that you know. ')
    if ans == "y":
        trajectory_player.replay_trajectory(execute_cb=True)
    else:
        exit(0)
