from __future__ import annotations
from dataclasses import dataclass

# Python
from pathlib import Path
from typing import Any, List

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
        def __init__(self, root: Path, file_preffix: str):
            dir = root / "outer_mov"
            dir.mkdir(exist_ok=True)
            jp_filename = dir / (file_preffix + "_jp_record.csv")
            cp_filename = dir / (file_preffix + "_cp_record.csv")
            super().__init__(jp_filename, cp_filename)

    def __post_init__(self):
        self.delay = 0.15
        self.pitch_record_collection = self.RecordCollection(self.root, "outer_pitch")
        self.yaw_record_collection = self.RecordCollection(self.root, "yaw_pitch")

        init_jp = self.replay_device.measured_jp()
        self.outer_pitch_traj = Trajectory.from_numpy(DvrkMotions.outer_pitch_trajectory(init_jp))
        self.outer_yaw_traj = Trajectory.from_numpy(DvrkMotions.outer_yaw_trajectory(init_jp))

    def outer_yaw_calib(self):
        self.data_recorder_cb.record_collection = self.yaw_record_collection
        trajectory_player = TrajectoryPlayer(
            self.replay_device, self.outer_yaw_traj, before_motion_loop_cb=[], after_motion_cb=[self.data_recorder_cb]
        )
        trajectory_player.replay_trajectory(execute_cb=True, delay=self.delay)

    def outer_pitch_calib(self):
        self.data_recorder_cb.record_collection = self.pitch_record_collection
        trajectory_player = TrajectoryPlayer(
            self.replay_device, self.outer_pitch_traj, before_motion_loop_cb=[], after_motion_cb=[self.data_recorder_cb]
        )
        trajectory_player.replay_trajectory(execute_cb=True, delay=self.delay)

    def __call__(self):
        self.outer_yaw_calib()
        self.outer_pitch_calib()


# class OuterJointsCalibrationRecorder:
#     """Class to perfom the calibration motion of the outer joints while recording the data from
#     robot and sensor
#     """

#     def __init__(
#         self,
#         replay_device,
#         ftk_handler: ftk_500,
#         save: bool,
#         expected_markers: int,
#         root: Path,
#         marker_name: str,
#     ):
#         self.save = save
#         self.replay_device = replay_device
#         self.ftk_handler = ftk_handler
#         self.expected_markers = expected_markers

#         self.outer_js_files = root / "outer_mov"
#         if not self.outer_js_files.exists():
#             self.outer_js_files.mkdir(parents=True)

#     def __call__(self) -> Any:
#         DvrkMotions.outer_pitch_yaw_motion(
#             self.replay_device.measured_jp(),
#             psm_handler=self.replay_device,
#             ftk_handler=self.ftk_handler,
#             expected_markers=self.expected_markers,
#             save=self.save,
#             root=self.outer_js_files,
#         )

#     def outer_pitch_yaw_motion(
#         init_jp,
#         psm_handler,
#         ftk_handler: ftk_500,
#         root: Path = None,
#         expected_markers: int = 4,
#         save: bool = True,
#     ):
#         # Init outer roll(Shaft)
#         init_jp[3] = -0.1589
#         # Generate outer yaw trajectory
#         outer_yaw_trajectory = DvrkMotions.generate_outer_yaw()
#         outer_yaw_trajectory = list(product(outer_yaw_trajectory, [0.0], [init_jp[2]]))
#         # Generate outer pitch trajectory
#         outer_pitch_trajectory = DvrkMotions.generate_outer_pitch()
#         outer_pitch_trajectory = list(product([0.0], outer_pitch_trajectory, [init_jp[2]]))

#         outer_yaw_file = root / "outer_yaw.txt"
#         outer_pitch_file = root / "outer_pitch.txt"

#         # Outer yaw trajectory (q1)
#         init_jp[0] = 0.0
#         init_jp[1] = 0.0
#         DvrkMotions.outer_joints_motion(
#             init_jp,
#             psm_handler,
#             expected_markers,
#             save,
#             filename=outer_yaw_file,
#             trajectory=outer_yaw_trajectory,
#         )
#         # Outer pitch trajectory (q2)
#         init_jp[0] = 0.0
#         init_jp[1] = 0.0
#         DvrkMotions.outer_joints_motion(
#             init_jp,
#             psm_handler,
#             expected_markers,
#             save,
#             filename=outer_pitch_file,
#             trajectory=outer_pitch_trajectory,
#         )

#     @staticmethod
#     def outer_joints_motion(
#         init_jp,
#         psm_handler,
#         ftk_handler: ftk_500,
#         expected_markers=4,
#         save: bool = False,
#         filename: Path = None,
#         trajectory: List = None,
#     ):
#         """Perform a trajectory using only the first three joints.

#         Args:
#             init_jp ([type]): [description]
#             psm_handler ([type]): [description]
#             log ([type]): [description]
#             expected_markers (int, optional): [description]. Defaults to 4.
#             save (bool, optional): [description]. Defaults to True.
#             filename (Path, optional): [description]. Defaults to None.
#             trajectory (List): List of joints values that the robot will follow. Each point in the trajectory
#             needs to specify q1, q2 and q3.
#         """
#         if trajectory is None:
#             raise Exception("Trajectory not given")
#         if filename is None:
#             raise Exception("filename not give")

#         # ftk_handler = ftk_500("custom_marker_112")

#         # Create record to store the measured data
#         sensor_record = AtracsysCartesianRecord(ftk_handler, expected_markers=expected_markers, filename=filename)
#         # Move to initial position
#         psm_handler.move_jp(init_jp).wait()
#         time.sleep(1)
#         # Start trajectory
#         counter = 0
#         for q1, q2, q3 in track(trajectory, description="-- Trajectory Progress -- "):
#             counter += 1
#             # Move only wrist joints
#             log.info(f"q1 {q1:+.4f} q2 {q2:+.4f} q3 {q3:+.4f}")
#             init_jp[0] = q1
#             init_jp[1] = q2
#             init_jp[2] = q3
#             q7 = psm_handler.jaw_jp()
#             psm_handler.move_jp(init_jp).wait()
#             time.sleep(0.15)

#             sensor_record.create_new_entry(counter, init_jp, q7)

#         if save:
#             sensor_record.to_csv(safe_save=False)
