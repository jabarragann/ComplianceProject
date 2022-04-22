from __future__ import annotations

# Python
from pathlib import Path
from random import Random
import time
from typing import Union
from dataclasses import dataclass, field
from black import out
import numpy as np
import numpy
from typing import List

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


class TrajectoryPlayer:
    def __init__(self, replay_device, trajectory, before_motion_cb: List = [], after_motion_cb: List = []):
        self.trajectory = trajectory
        self.replay_device = replay_device
        self.after_motion_cb = after_motion_cb
        self.before_motion_cb = before_motion_cb

    def replay_trajectory(self, execute_cb: bool = True):

        start_time = time.time()
        last_bag_time = self.trajectory[0].header.stamp.to_sec()

        # Before motion callbacks
        [cb() for cb in self.before_motion_cb]

        for index, new_js in enumerate(self.trajectory):
            # record start time
            loop_start_time = time.time()
            # compute expected dt
            new_bag_time = self.trajectory[index].header.stamp.to_sec()
            delta_bag_time = new_bag_time - last_bag_time
            last_bag_time = new_bag_time

            # Move
            log.info(f"-- Trajectory Progress --> {100*index/len(self.trajectory):0.02f} %")
            self.replay_device.move_jp(numpy.array(new_js.position)).wait()  # wait until motion is finished

            # After motion callbacks
            [cb(**{"wrist_calib_index": index}) for cb in self.after_motion_cb]

            # try to keep motion synchronized
            loop_end_time = time.time()
            sleep_time = delta_bag_time - (loop_end_time - loop_start_time)
            # if process takes time larger than console rate, don't sleep
            if sleep_time > 0:
                time.sleep(sleep_time)
            # Safety sleep
            if loop_end_time - loop_start_time < 0.1:
                time.sleep(0.1)

        log.info("Time to replay trajectory: %f seconds" % (time.time() - start_time))


@dataclass
class Trajectory:
    sampling_factor: int = 1
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
        if self.iteration_idx * self.sampling_factor < len(self.setpoints):
            result = self.setpoints[self.iteration_idx * self.sampling_factor]
            self.iteration_idx += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, i):
        return self.setpoints[i]

    def __len__(self):
        return int(len(self.setpoints) / self.sampling_factor)

    @classmethod
    def from_ros_bag(cls, rosbag_handle: RosbagUtils, namespace="PSM2", sampling_factor: int = 1) -> Trajectory:
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
            sampling_factor=sampling_factor,
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
    class PsmJointLimits:
        # Specified in rad
        q1_range = np.array([-0.60, 0.70])
        q2_range = np.array([-0.49, 0.47])
        q3_range = np.array([0.13, 0.22])
        q4_range = np.array([-0.35, 1.0])
        q5_range = np.array([-1.34, 1.34])
        q6_range = np.array([-1.34, 1.34])

    @staticmethod
    def generate_random_joint():
        limits = RandomJointTrajectory.PsmJointLimits
        q1 = np.random.uniform(limits.q1_range[0], limits.q1_range[1])
        q2 = np.random.uniform(limits.q2_range[0], limits.q2_range[1])
        q3 = np.random.uniform(limits.q3_range[0], limits.q3_range[1])
        q4 = np.random.uniform(limits.q4_range[0], limits.q4_range[1])
        q5 = np.random.uniform(limits.q5_range[0], limits.q5_range[1])
        q6 = np.random.uniform(limits.q6_range[0], limits.q6_range[1])
        return [q1, q2, q3, q4, q5, q6]

    @classmethod
    def generate_trajectory(cls, samples: int):
        setpoints = []
        for i in range(samples):
            setpoint = JointState()
            setpoint.name = ["q1", "q2", "q3", "q4", "q5", "q6"]
            setpoint.position = RandomJointTrajectory.generate_random_joint()
            setpoints.append(setpoint)

        return RandomJointTrajectory(sampling_factor=1, setpoints=setpoints)


if __name__ == "__main__":
    rosbag_path = Path("data/psm2_trajectories/pitch_exp_traj_01_test_cropped.bag")
    rosbag_handle = RosbagUtils(rosbag_path)
    trajectory = Trajectory.from_ros_bag(rosbag_handle, sampling_factor=60)
    # trajectory = RandomJointTrajectory.generate_trajectory(2000)

    log.info(f"Initial pt {trajectory.setpoints[0].position}")
    log.info(f"Starting ts {trajectory.setpoints[0].header.stamp.to_sec()}")
    log.info(f"number of points {len(trajectory)}")

    arm_namespace = "PSM2"
    arm = ReplayDevice(device_namespace=arm_namespace, expected_interval=0.01)
    arm.home_device()

    trajectory_player = TrajectoryPlayer(arm, trajectory)

    ans = input('Press "y" to start data collection trajectory. Only replay trajectories that you know. ')
    if ans == "y":
        trajectory_player.replay_trajectory(execute_cb=True)
    else:
        exit(0)
