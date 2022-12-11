from __future__ import annotations
from dataclasses import dataclass

# Python
from pathlib import Path
from typing import Any, List
from kincalib.Entities.Msgs import MyJointState, MyPoseStamped

# ros
from kincalib.Motion.TrajectoryPlayer import SoftRandomJointTrajectory

# Custom
from kincalib.Recording.DataRecord import CartesianRecord, JointRecord
from kincalib.utils.Logger import Logger
from kincalib.Sensors.ftk_500_api import FTKDummy, ftk_500
from kincalib.Motion.DvrkMotions import DvrkMotions
from kincalib.Motion.ReplayDevice import ReplayDevice
from kincalib.Entities.RosConversions import RosConversion

log = Logger(__name__).log


@dataclass
class DataRecorder:
    replay_device: ReplayDevice
    record_collection: ExperimentRecordCollection
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


class RecordCollectionTemplate:
    def __init__(self, jp_filename: Path, cp_filename: Path) -> None:
        self.joint_record = JointRecord(jp_filename)
        self.pose_record = CartesianRecord(cp_filename)

    def add_new_robot_data(self, step, measured_js: MyJointState, setpoint_js: MyJointState, robot_cp: MyPoseStamped):
        self.joint_record.create_new_entry(step, measured_js, setpoint_js)
        self.pose_record.create_new_entry(step, "robot", "-1", robot_cp, setpoint_js)

    def add_new_sensor_data(
        self,
        step,
        setpoint_js: MyJointState,
        fiducials_cp: List[MyPoseStamped],
        tools_cp: List[MyPoseStamped],
        tools_id_list: List[str],
    ):
        for fid_cp in fiducials_cp:
            self.pose_record.create_new_entry(step, "fiducial", "-1", fid_cp, setpoint_js)
        for tool_cp, id in zip(tools_cp, tools_id_list):
            self.pose_record.create_new_entry(step, "tool", id, tool_cp, setpoint_js)

    def save_data(self, safe_save=False):
        self.joint_record.to_csv(safe_save=safe_save)
        self.pose_record.to_csv(safe_save=safe_save)


class ExperimentRecordCollection(RecordCollectionTemplate):
    def __init__(
        self,
        root_dir: Path,
        mode: str = "calib",
        test_id: int = None,
        description: str = "",
    ) -> None:
        """_summary_

        Parameters
        ----------
        root_dir : _type_
            _description_
        mode : str, optional
            Operation mode, by default "calib". Needs to be either 'calib' or 'test'
        test_id : int, optional
            _description_, by default None
        description: str, optional
            one line experiment description
        """

        assert mode in ["calib", "test"], "mode needs to be calib or test"
        self.description = description
        self.root_dir = root_dir
        self.mode = mode
        self.test_id = test_id

        self.create_paths()
        self.cp_filename, self.jp_filename = self.obtain_filenames_for_records()

        super().__init__(self.jp_filename, self.cp_filename)

    def obtain_filenames_for_records(self):
        if self.mode == "calib":
            cp_filename = self.robot_files / ("robot_cp.csv")
            jp_filename = self.robot_files / ("robot_jp.csv")
        elif self.mode == "test":
            cp_filename = self.test_files / ("robot_cp.csv")
            jp_filename = self.test_files / ("robot_jp.csv")

        if cp_filename.exists() or jp_filename.exists():
            n = input(f"Data was found in directory {cp_filename.parent}. Press (y/Y) to overwrite. ")
            if not (n == "y" or n == "Y"):
                log.info("exiting the script")
                exit(0)
        return cp_filename, jp_filename

    def create_paths(self):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if self.mode == "calib":
            self.robot_files = self.root_dir / "robot_mov"
            self.robot_files.mkdir(parents=True, exist_ok=True)

        if self.mode == "test":
            assert self.test_id is not None, "undefined test id"
            self.test_files = self.root_dir / f"test_trajectories/{self.test_id:02d}"
            self.test_files.mkdir(parents=True, exist_ok=True)

        description_path = self.robot_files if self.mode == "calib" else self.test_files
        with open(description_path / "description.txt", "w") as f:
            f.write(self.description)


####################OLD

# class DataRecorder:
#     """
#     Callable data class to record data from robot and sensor.
#     """

#     def __init__(
#         self,
#         replay_device,
#         ftk_handler: ftk_500,
#         expected_markers: int,
#         root: Path,
#         marker_name: str,
#         mode: str,
#         test_id: int,
#         description: str = None,
#     ):

#         self.replay_device = replay_device
#         self.expected_markers = expected_markers
#         self.ftk_handler = ftk_handler
#         self.calibration_record = CalibrationRecord(
#             ftk_handler=self.ftk_handler,
#             robot_handler=self.replay_device,
#             expected_markers=expected_markers,
#             root_dir=root,
#             mode=mode,
#             test_id=test_id,
#             description=description,
#         )

#     def __call__(self, **kwargs):
#         self.collect_data(**kwargs)

#     def collect_data(self, index=None):
#         if index is None:
#             log.error("specify index WristJointsCalibrationRecorder")
#             exit(0)
#         log.info("record data")
#         marker_pose, fiducials_pose = self.ftk_handler.obtain_processed_measurement(
#             self.expected_markers, t=200, sample_time=15
#         )
#         # Measure
#         jp = self.replay_device.measured_jp()
#         jaw_jp = self.replay_device.jaw.measured_jp()[0]
#         # Check if marker is visible add sensor data
#         if marker_pose is not None:
#             self.calibration_record.create_new_sensor_entry(index, jp, jaw_jp)
#         # Always add robot data
#         self.calibration_record.create_new_robot_entry(index, jp, jaw_jp)

#         self.calibration_record.to_csv(safe_save=False)


class OuterJointsCalibrationRecorder:
    """Class to perfom the calibration motion of the outer joints while recording the data from
    robot and sensor
    """

    def __init__(
        self,
        replay_device,
        ftk_handler: ftk_500,
        save: bool,
        expected_markers: int,
        root: Path,
        marker_name: str,
    ):
        self.save = save
        self.replay_device = replay_device
        self.ftk_handler = ftk_handler
        self.expected_markers = expected_markers

        self.outer_js_files = root / "outer_mov"
        if not self.outer_js_files.exists():
            self.outer_js_files.mkdir(parents=True)

    def __call__(self) -> Any:
        DvrkMotions.outer_pitch_yaw_motion(
            self.replay_device.measured_jp(),
            psm_handler=self.replay_device,
            ftk_handler=self.ftk_handler,
            expected_markers=self.expected_markers,
            save=self.save,
            root=self.outer_js_files,
        )


class WristJointsCalibrationRecorder:
    """Class to perfom the calibration motion of the wrist joints while recording the data from
    robot and sensor
    """

    def __init__(
        self,
        replay_device,
        ftk_handler: ftk_500,
        save: bool,
        expected_markers: int,
        root: Path,
        marker_name: str,
    ):
        self.save = save
        self.replay_device = replay_device
        self.expected_markers = expected_markers
        self.ftk_handler = ftk_handler

        self.wrist_files = root / "pitch_roll_mov"
        if not self.wrist_files.exists():
            self.wrist_files.mkdir(parents=True)

    def __call__(self, index=None) -> Any:
        if index is None:
            log.error("specify index WristJointsCalibrationRecorder")
            exit(0)
        log.info("calibration")
        DvrkMotions.pitch_yaw_roll_independent_motion(
            self.replay_device.measured_jp(),
            psm_handler=self.replay_device,
            ftk_handler=self.ftk_handler,
            expected_markers=self.expected_markers,
            save=self.save,
            filename=self.wrist_files / f"step{index:03d}_wrist_motion.txt",
        )


if __name__ == "__main__":

    from kincalib.Motion.TrajectoryPlayer import Trajectory, TrajectoryPlayer
    from kincalib.Calibration.CalibrationRoutines import OuterJointsCalibrationRoutine
    from kincalib.Calibration.CalibrationRoutines import WristCalibrationRoutine

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
    root = Path(f"{home}/temp/test_rec/rec04")
    # Sensors
    # ftk_handler = ftk_500("marker_12")
    ftk_handler = FTKDummy("marker_12")

    # Main recording loop
    experiment_record_collection = ExperimentRecordCollection(root, mode="calib", description="test calib")
    data_recorder_cb = DataRecorder(arm, experiment_record_collection, ftk_handler, 4, marker_name="test")

    # Setup calibration callbacks
    outer_joints_recorder = DataRecorder(arm, None, ftk_handler, 4, "none")
    outer_js_calib_cb = OuterJointsCalibrationRoutine(arm, ftk_handler, outer_joints_recorder, save=True, root=root)

    wrist_joints_recorder = DataRecorder(arm, None, ftk_handler, 4, "none")
    wrist_js_calib_cb = WristCalibrationRoutine(arm, ftk_handler, wrist_joints_recorder, save=True, root=root)

    # outer_js_calib_cb = OuterJointsCalibrationRecorder(
    #     replay_device=arm,
    #     ftk_handler=ftk_handler,
    #     save=False,
    #     expected_markers=4,
    #     root=Path("."),
    #     marker_name="none",
    # )

    # wrist_js_calib_cb = WristJointsCalibrationRecorder(
    #     replay_device=arm,
    #     ftk_handler=ftk_handler,
    #     save=False,
    #     expected_markers=4,
    #     root=Path("."),
    #     marker_name="none",
    # )

    # Create trajectory player with cb
    # trajectory_player = TrajectoryPlayer(
    #     arm, trajectory, before_motion_cb=[outer_js_calib_cb], after_motion_cb=[wrist_js_calib_cb]
    # )
    trajectory_player = TrajectoryPlayer(
        arm,
        trajectory,
        before_motion_loop_cb=[],
        after_motion_cb=[data_recorder_cb, wrist_js_calib_cb],
    )

    ans = input('Press "y" to start data collection trajectory. Only replay trajectories that you know. ')
    if ans == "y":
        trajectory_player.replay_trajectory(delay=0.005)

    log.info(f"Initial pt {trajectory.setpoints[0].position}")
    log.info(f"Starting ts {trajectory.setpoints[0].header.stamp.to_sec()}")
    log.info(f"number of points {len(trajectory)}")

    # arm_namespace = "PSM2"
    # arm = ReplayDevice(device_namespace=arm_namespace, expected_interval=0.01)
    # arm.home_device()

    # # Sensors
    # # ftk_handler = ftk_500("marker_12")
    # ftk_handler = FTKDummy()

    # # Setup calibration callbacks
    # outer_js_calib_cb = OuterJointsCalibrationRecorder(
    #     replay_device=arm,
    #     ftk_handler=ftk_handler,
    #     save=False,
    #     expected_markers=4,
    #     root=Path("."),
    #     marker_name="none",
    # )
