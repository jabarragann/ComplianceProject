from __future__ import annotations

# Python
from pathlib import Path
from re import I
import time
from typing import Union
from dataclasses import dataclass, field
from black import out
import numpy as np
import numpy

# Custom
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.Logger import Logger
from kincalib.Motion.DataRecorder import CalibrationRecord
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.Logger import Logger
from kincalib.Atracsys.ftk_500_api import ftk_500
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.Motion.CalibrationMotions import CalibrationMotions
from kincalib.Motion.ReplayDevice import ReplayDevice

log = Logger(__name__).log


class TrajectoryPlayer:
    def __init__(
        self, replay_device, trajectory, expected_markers, root: Path, marker_name: str, mode: str, test_id: int
    ):
        assert mode in ["calib", "test"], "mode needs to be calib or test"
        self.mode = mode
        self.expected_markers = expected_markers
        self.trajectory = trajectory
        self.ftk_handler = ftk_500(marker_name)
        self.replay_device = replay_device
        self.calibration_record = CalibrationRecord(
            ftk_handler=self.ftk_handler,
            robot_handler=self.replay_device,
            expected_markers=expected_markers,
            root_dir=root,
            mode=mode,
            test_id=test_id,
        )

        self.outer_js_files = root / "outer_mov"
        self.wrist_files = root / "pitch_roll_mov"
        if not self.outer_js_files.exists():
            self.outer_js_files.mkdir(parents=True)
        if not self.wrist_files.exists():
            self.wrist_files.mkdir(parents=True)

    def replay_trajectory(self, measure=True, saving=True, calibration=False):

        if self.mode == "calib" and measure:
            # Outer joints calibration
            CalibrationMotions.outer_pitch_yaw_motion(
                self.replay_device.measured_jp(),
                psm_handler=self.replay_device,
                expected_markers=4,
                save=True,
                root=self.outer_js_files,
            )

        start_time = time.time()
        last_bag_time = self.trajectory[0].header.stamp.to_sec()

        for index, new_js in enumerate(self.trajectory):
            # record start time
            loop_start_time = time.time()
            # compute expected dt
            new_bag_time = self.trajectory[index].header.stamp.to_sec()
            delta_bag_time = new_bag_time - last_bag_time
            last_bag_time = new_bag_time

            # replay
            if index % 40 == 0:
                log.info(f"-- Trajectory Progress --> {100*index/len(self.trajectory):0.02f} %")
                # Move
                self.replay_device.move_jp(numpy.array(new_js.position)).wait()

                if measure:
                    marker_pose, fiducials_pose = self.ftk_handler.obtain_processed_measurement(
                        self.expected_markers, t=200, sample_time=15
                    )
                    if marker_pose is not None:
                        # Measure
                        jp = self.replay_device.measured_jp()
                        jaw_jp = self.replay_device.jaw.measured_jp()[0]
                        self.calibration_record.create_new_entry(index, jp, jaw_jp)

                        if calibration and measure:
                            # wrist joints calibration
                            init_jp = self.replay_device.measured_jp()
                            CalibrationMotions.pitch_yaw_roll_independent_motion(
                                init_jp,
                                psm_handler=self.replay_device,
                                expected_markers=4,
                                save=True,
                                filename=self.wrist_files / f"step{index:03d}_wrist_motion.txt",
                            )

                        self.calibration_record.to_csv(safe_save=False)
                    else:
                        log.warning("No marker pose detected.")

            # try to keep motion synchronized
            loop_end_time = time.time()
            sleep_time = delta_bag_time - (loop_end_time - loop_start_time)
            # if process takes time larger than console rate, don't sleep
            if sleep_time > 0:
                time.sleep(sleep_time)

        log.info("Time to replay trajectory: %f seconds" % (time.time() - start_time))


@dataclass
class Trajectory:
    bbmin: np.ndarray = np.zeros(3)
    bbmax: np.ndarray = np.zeros(3)
    last_message_time: float = 0.0
    out_of_order_counter: int = 0
    setpoints: list = field(default_factory=lambda: [])
    setpoint_js_t: str = ""
    setpoint_cp_t: str = ""

    def __post_init__(self) -> None:
        # parse bag and create list of points
        pass

    def trajectory_report(self):
        log.info("Trajectory report:")
        # report out of order setpoints
        if self.out_of_order_counter > 0:
            self.log.info("-- Found and removed %i out of order setpoints" % (self.out_of_order_counter))

        # convert to mm
        bbmin = self.bbmin * 1000.0
        bbmax = self.bbmax * 1000.0
        self.log.info(
            "-- Range of motion in mm:\n   X:[%f, %f]\n   Y:[%f, %f]\n   Z:[%f, %f]"
            % (bbmin[0], bbmax[0], bbmin[1], bbmax[1], bbmin[2], bbmax[2])
        )

        # compute duration
        duration = self.setpoints[-1].header.stamp.to_sec() - self.setpoints[0].header.stamp.to_sec()
        self.log.info("-- Duration of trajectory: %f seconds" % (duration))

        # Number of poses
        self.log.info("-- Found %i setpoints using topic %s" % (len(self.setpoints), self.setpoint_js_t))
        if len(self.setpoints) == 0:
            self.log.error("-- No trajectory found!")

    def __iter__(self):
        self.iteration_idx = 0
        return self

    def __next__(self):
        if self.iteration_idx < len(self.setpoints):
            result = self.setpoints[self.iteration_idx]
            self.iteration_idx += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, i):
        return self.setpoints[i]

    def __len__(self):
        return len(self.setpoints)

    @classmethod
    def from_ros_bag(cls, rosbag_handle: RosbagUtils, namespace="PSM2") -> Trajectory:
        bbmin = np.zeros(3)
        bbmax = np.zeros(3)
        last_message_time = 0.0
        out_of_order_counter = 0
        setpoints = []
        setpoint_js_t = f"/{namespace}/setpoint_js"
        setpoint_cp_t = f"/{namespace}/setpoint_cp"

        # rosbag_handle = RosbagUtils(filename)
        log.info("-- Parsing bag %s" % (rosbag_handle.name))
        for bag_topic, bag_message, t in rosbag_handle.rosbag_handler.read_messages():
            # Collect setpoint_cp only to keep track of workspace
            if bag_topic == setpoint_cp_t:
                # check order of timestamps, drop if out of order
                transform_time = bag_message.header.stamp.to_sec()
                if transform_time <= last_message_time:
                    out_of_order_counter = out_of_order_counter + 1
                else:
                    # keep track of workspace
                    position = numpy.array(
                        [
                            bag_message.pose.position.x,
                            bag_message.pose.position.y,
                            bag_message.pose.position.z,
                        ]
                    )
                    if len(setpoints) == 1:
                        bbmin = position
                        bmax = position
                    else:
                        bbmin = numpy.minimum(bbmin, position)
                        bbmax = numpy.maximum(bbmax, position)
            elif bag_topic == setpoint_js_t:
                # check order of timestamps, drop if out of order
                transform_time = bag_message.header.stamp.to_sec()
                if transform_time <= last_message_time:
                    out_of_order_counter = out_of_order_counter + 1
                else:
                    setpoints.append(bag_message)

        return Trajectory(
            bbmin=bbmin,
            bbmax=bbmax,
            last_message_time=last_message_time,
            out_of_order_counter=out_of_order_counter,
            setpoints=setpoints,
            setpoint_js_t=setpoint_js_t,
            setpoint_cp_t=setpoint_cp_t,
        )


@dataclass
class RandomJointTrajectory(Trajectory):
    pass

    @classmethod
    def generate_trajectory(samples):
        pass


if __name__ == "__main__":
    traj = Trajectory()
