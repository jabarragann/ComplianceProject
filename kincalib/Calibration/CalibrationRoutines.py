from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

# Python
from pathlib import Path
from re import I
from typing import Any, Callable, List

# ros
from kincalib.Motion.TrajectoryPlayer import SoftRandomJointTrajectory, Trajectory, TrajectoryPlayer

# Custom
from kincalib.utils.Logger import Logger
from kincalib.Sensors.ftk_500_api import FTKDummy, ftk_500
from kincalib.Motion.DvrkMotions import DvrkMotions
from kincalib.Motion.ReplayDevice import ReplayDevice
from kincalib.Entities.RosConversions import RosConversion
from kincalib.Recording.DataRecorder import DataRecorder, RecordCollectionTemplate


log = Logger(__name__).log


@dataclass
class OuterJointsCalibrationRoutine:
    replay_device: ReplayDevice
    ftk_handler: ftk_500
    data_recorder_cb: DataRecorder
    save: bool
    root: Path

    class RecordCollection(RecordCollectionTemplate):
        def __init__(self, root: Path, file_preffix: str, save: bool):
            if save:
                dir = root / "outer_mov"
                dir.mkdir(exist_ok=True)
                jp_filename = dir / (file_preffix + "_jp_record.csv")
                cp_filename = dir / (file_preffix + "_cp_record.csv")
                super().__init__(jp_filename, cp_filename)

    def __post_init__(self):
        self.delay = 0.15
        self.pitch_record_collection = self.RecordCollection(self.root, "outer_pitch", self.save)
        self.yaw_record_collection = self.RecordCollection(self.root, "yaw_pitch", self.save)

        init_jp = self.replay_device.measured_jp()
        self.outer_pitch_traj = Trajectory.from_numpy(DvrkMotions.outer_pitch_trajectory(init_jp))
        self.outer_yaw_traj = Trajectory.from_numpy(DvrkMotions.outer_yaw_trajectory(init_jp))

    def outer_yaw_calib(self):
        if self.save:
            self.data_recorder_cb.record_collection = self.yaw_record_collection
            after_motion_cb = [self.data_recorder_cb]

        trajectory_player = TrajectoryPlayer(
            self.replay_device,
            self.outer_yaw_traj,
            before_motion_loop_cb=[],
            after_motion_cb=after_motion_cb,
        )
        trajectory_player.replay_trajectory(execute_cb=True, delay=self.delay)

    def outer_pitch_calib(self):
        if self.save:
            self.data_recorder_cb.record_collection = self.pitch_record_collection
            after_motion_cb = [self.data_recorder_cb]

        trajectory_player = TrajectoryPlayer(
            self.replay_device,
            self.outer_pitch_traj,
            before_motion_loop_cb=[],
            after_motion_cb=after_motion_cb,
        )
        trajectory_player.replay_trajectory(execute_cb=True, delay=self.delay)

    def __call__(self):
        self.outer_yaw_calib()
        self.outer_pitch_calib()


@dataclass
class WristCalibrationRoutine:
    replay_device: ReplayDevice
    ftk_handler: ftk_500
    data_recorder_cb: DataRecorder
    save: bool
    root: Path

    class RecordCollection(RecordCollectionTemplate):
        def __init__(self, root: Path, file_preffix: str, save: bool):
            if save:
                dir = root / "pitch_roll_mov"
                dir.mkdir(exist_ok=True)
                jp_filename = dir / (file_preffix + "_jp_record.csv")
                cp_filename = dir / (file_preffix + "_cp_record.csv")
                super().__init__(jp_filename, cp_filename)

    def create_records(self, step_in_traj):
        self.delay = 0.15
        prefix = f"step{step_in_traj:03d}_wrist_motion_"
        self.pitch_yaw_record_collection = self.RecordCollection(self.root, prefix + "pitch_yaw", self.save)
        self.roll_record_collection = self.RecordCollection(self.root, prefix + "roll", self.save)

        init_jp = self.replay_device.measured_jp()
        self.pitch_yaw_traj = Trajectory.from_numpy(DvrkMotions.pitch_yaw_trajectory(init_jp))
        self.roll_traj = Trajectory.from_numpy(DvrkMotions.roll_trajectory(init_jp))

    def pitch_yaw_calib(self):
        if self.save:
            self.data_recorder_cb.record_collection = self.pitch_yaw_record_collection
            after_motion_cb = [self.data_recorder_cb]

        trajectory_player = TrajectoryPlayer(
            self.replay_device,
            self.pitch_yaw_traj,
            before_motion_loop_cb=[],
            after_motion_cb=after_motion_cb,
        )
        trajectory_player.replay_trajectory(execute_cb=True, delay=self.delay)

    def roll_calib(self):
        if self.save:
            self.data_recorder_cb.record_collection = self.roll_record_collection
            after_motion_cb = [self.data_recorder_cb]

        trajectory_player = TrajectoryPlayer(
            self.replay_device,
            self.roll_traj,
            before_motion_loop_cb=[],
            after_motion_cb=after_motion_cb,
        )
        trajectory_player.replay_trajectory(execute_cb=True, delay=self.delay)

    def __call__(self, index):
        self.create_records(index)
        self.roll_calib()
        self.pitch_yaw_calib()

        self.pitch_yaw_record_collection = None
        self.roll_record_collection = None


@dataclass
class DhParamCalibrationRoutine:
    replay_device: ReplayDevice
    ftk_handler: ftk_500
    data_recorder_cb: DataRecorder
    save: bool
    root: Path
    delay: float = 0.15

    class FilePrefixes(Enum):
        J1 = "j1_outer_yaw"
        J2 = "j2_outer_pitch"
        J3 = "j3_insertion"
        J4 = "j2_roll"

    class RecordCollection(RecordCollectionTemplate):
        def __init__(self, step: int, root: Path, file_preffix: str, save: bool):
            if save:
                dir = root / f"dh_calibration/{step:03d}"
                dir.mkdir(parents=True, exist_ok=True)
                jp_filename = dir / (file_preffix + "_jp_record.csv")
                cp_filename = dir / (file_preffix + "_cp_record.csv")
                super().__init__(jp_filename, cp_filename)

    def __post_init__(self):
        self.motions_list = [
            (self.FilePrefixes.J1, DvrkMotions.outer_yaw_trajectory),
            (self.FilePrefixes.J2, DvrkMotions.outer_pitch_trajectory),
            (self.FilePrefixes.J3, DvrkMotions.insertion_trajectory),
            (self.FilePrefixes.J4, DvrkMotions.roll_trajectory),
        ]

    def __call__(self, index):
        prefix: self.FilePrefixes
        traj_generator: Callable
        for prefix, traj_generator in self.motions_list:
            init_jp = self.replay_device.measured_jp()
            collection = self.RecordCollection(index, self.root, prefix.value, self.save)
            self.data_recorder_cb.record_collection = collection

            traj = Trajectory.from_numpy(traj_generator(init_jp, steps=22))
            TrajectoryPlayer(
                self.replay_device,
                traj,
                before_motion_loop_cb=[],
                after_motion_cb=[self.data_recorder_cb],
            ).replay_trajectory(execute_cb=True, delay=self.delay)
