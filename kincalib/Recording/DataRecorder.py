from __future__ import annotations

# Python
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List
from kincalib.Entities.Msgs import MyJointState, MyPoseStamped

# ros
from kincalib.Motion.TrajectoryPlayer import SoftRandomJointTrajectory

# Custom
from kincalib.Recording.RecordCollections import RecordCollectionTemplate
from kincalib.utils.Logger import Logger
from kincalib.Sensors.ftk_500_api import FTKDummy, ftk_500
from kincalib.Motion.ReplayDevice import ReplayDevice
from kincalib.Entities.RosConversions import RosConversion

log = Logger(__name__).log


@dataclass
class DataRecorder:
    replay_device: ReplayDevice
    record_collection: RecordCollectionTemplate
    ftk_handler: ftk_500
    expected_markers: int
    marker_name: str

    def __call__(self, **kwargs):
        self.record(**kwargs)

    def record(self, index=None):
        if index is None:
            log.error("index not specified")
            exit(0)
        self.collect_data(index)
        self.save_data()

    def collect_data(self, index):
        log.info("record data")

        # quickly check the sensor to see if tool and fiducial are observable
        marker_pose, fiducials_pose = self.ftk_handler.obtain_processed_measurement(
            self.expected_markers, t=100, sample_time=15
        )

        # Get robot data
        setpoint_js = MyJointState.from_crtk_js(self.replay_device.setpoint_js())
        measured_js = MyJointState.from_crtk_js(self.replay_device.measured_js())
        measured_cp = RosConversion.pykdl_to_myposestamped("psm2", self.replay_device.measured_cp())
        self.record_collection.add_new_robot_data(index, measured_js, setpoint_js, measured_cp)

        if marker_pose is not None:
            mean_tool_frame, fiducial_positions = self.ftk_handler.obtain_processed_measurement(
                self.expected_markers, t=500, sample_time=15
            )
            tool_cp = RosConversion.pykdl_to_myposestamped("tool", mean_tool_frame)
            fiducial_cp = MyPoseStamped.from_array_of_positions("fiducial", fiducial_positions)

            self.record_collection.add_new_sensor_data(
                index, setpoint_js, fiducial_cp, [tool_cp], [self.ftk_handler.marker_name]
            )

    def save_data(self):
        self.record_collection.save_data()


if __name__ == "__main__":

    from kincalib.Motion.TrajectoryPlayer import Trajectory, TrajectoryPlayer
    from kincalib.Calibration.CalibrationRoutines import OuterJointsCalibrationRoutine
    from kincalib.Calibration.CalibrationRoutines import WristCalibrationRoutine
    from kincalib.Recording.RecordCollections import ExperimentRecordCollection

    import os
    import time

    # Test calibration motions

    # rosbag_path = Path("data/psm2_trajectories/pitch_exp_traj_01_test_cropped.bag")
    # rosbag_handle = RosbagUtils(rosbag_path)
    # trajectory = Trajectory.from_ros_bag(rosbag_handle, sampling_factor=60)

    trajectory = SoftRandomJointTrajectory.generate_trajectory(50, samples_per_step=20)

    log.info(f"Initial pt {trajectory.setpoints[0].position}")
    log.info(f"Starting ts {trajectory.setpoints[0].header.stamp.to_sec()}")
    log.info(f"number of points {len(trajectory)}")

    arm_namespace = "PSM2"
    arm = ReplayDevice(device_namespace=arm_namespace, expected_interval=0.01)
    arm.home_device()
    time.sleep(0.1)

    home = os.path.expanduser("~")

    root = Path(f"{home}/temp/test_rec/rec06")
    # Sensors
    # ftk_handler = ftk_500("marker_12")
    ftk_handler = FTKDummy("marker_12")

    # Main recording loop
    experiment_record_collection = ExperimentRecordCollection(
        root, mode="calib", description="test calib"
    )
    data_recorder_cb = DataRecorder(
        arm, experiment_record_collection, ftk_handler, 4, marker_name="test"
    )

    # Setup calibration callbacks
    outer_joints_recorder = DataRecorder(arm, None, ftk_handler, 4, "none")
    outer_js_calib_cb = OuterJointsCalibrationRoutine(
        arm, ftk_handler, outer_joints_recorder, save=True, root=root
    )

    wrist_joints_recorder = DataRecorder(arm, None, ftk_handler, 4, "none")
    wrist_js_calib_cb = WristCalibrationRoutine(
        arm, ftk_handler, wrist_joints_recorder, save=True, root=root
    )

    trajectory_player = TrajectoryPlayer(
        arm,
        trajectory,
        before_motion_loop_cb=[],
        after_motion_cb=[data_recorder_cb, wrist_js_calib_cb],
    )

    ans = input(
        'Press "y" to start data collection trajectory. Only replay trajectories that you know. '
    )
    if ans == "y":
        trajectory_player.replay_trajectory(delay=0.005)

    log.info(f"Initial pt {trajectory.setpoints[0].position}")
    log.info(f"Starting ts {trajectory.setpoints[0].header.stamp.to_sec()}")
    log.info(f"number of points {len(trajectory)}")
