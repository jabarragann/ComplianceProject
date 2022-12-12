from __future__ import annotations

# Python
from pathlib import Path
import time
from dataclasses import dataclass, field
import numpy as np
import numpy
from typing import List
from kincalib.Calibration.CalibrationUtils import CalibrationUtils

# ros
from sensor_msgs.msg import JointState
from kincalib.Motion.DvrkMotions import DvrkMotions

# Custom
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.Logger import Logger
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.Logger import Logger
from kincalib.Motion.ReplayDevice import ReplayDevice

log = Logger(__name__).log


class TrajectoryPlayer:
    def __init__(
        self,
        replay_device: ReplayDevice,
        trajectory: Trajectory,
        before_motion_loop_cb: List = [],
        after_motion_cb: List = [],
    ):
        """Class to playback a `Trajectory` using a `ReplayDevice`. This class only uses the joint
        states to replay the trajectory, no cartesian setpoints are used. An additional callback system is used
        to add extra functionalities before the motion loop and after each motion.

        After motion callbacks will always receive a index parameter input. To use this argument
        use the sample signature below.

        ```python
        def after_motion_cb(index=None):
            pass
        ```

        Parameters
        ----------
        replay_device : ReplayDevice
           Device that will be executing the trajectory
        trajectory : Trajectory
            Object that contains a collection of setpoints
        before_motion_loop_cb : List, optional
            List of callback functions that will be executed before the motion loop
        after_motion_cb : List, optional
            List of callback functions that will be executed after each motion in the motion loop
        """

        self.trajectory = trajectory
        self.replay_device = replay_device
        self.after_motion_cb = after_motion_cb
        self.before_motion_loop_cb = before_motion_loop_cb

    def replay_trajectory(self, execute_cb: bool = True, delay=0.01):

        start_time = time.time()

        # Before motion callbacks
        if execute_cb:
            [cb() for cb in self.before_motion_loop_cb]

        for index, new_js in enumerate(self.trajectory):
            log.info(f"Executed step {index}")
            log.info(f"-- Trajectory Progress --> {100*index/len(self.trajectory):0.02f} %")
            self.replay_device.move_jp(numpy.array(new_js.position)).wait()
            time.sleep(delay)

            # After motion callbacks
            if execute_cb:
                [cb(**{"index": index}) for cb in self.after_motion_cb]

        log.info("Time to replay trajectory: %f seconds" % (time.time() - start_time))


@dataclass
class Trajectory:
    """Class storing a collection of joint setpoints to reproduce a trajectory. This class is iterable and should be use
    with the `TrajectoryPlayer`.


    Parameters
    ----------
    sampling_factor: int
       Sample the setpoints every sampling_factor.

    """

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
            self.log.info(
                "-- Found and removed %i out of order setpoints" % (self.out_of_order_counter)
            )

        # convert to mm
        bbmin = self.bbmin * 1000.0
        bbmax = self.bbmax * 1000.0
        self.log.info(
            "-- Range of motion in mm:\n   X:[%f, %f]\n   Y:[%f, %f]\n   Z:[%f, %f]"
            % (bbmin[0], bbmax[0], bbmin[1], bbmax[1], bbmin[2], bbmax[2])
        )

        # compute duration
        duration = (
            self.setpoints[-1].header.stamp.to_sec() - self.setpoints[0].header.stamp.to_sec()
        )
        self.log.info("-- Duration of trajectory: %f seconds" % (duration))

        # Number of poses
        self.log.info(
            "-- Found %i setpoints using topic %s" % (len(self.setpoints), self.setpoint_js_t)
        )
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
    def from_ros_bag(
        cls, rosbag_handle: RosbagUtils, namespace="PSM2", sampling_factor: int = 1
    ) -> Trajectory:
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

    @classmethod
    def from_numpy(cls, joint_array: np.ndarray, sampling_factor: int = 1) -> Trajectory:
        """Create a trajectory from a numpy array of joint values

        Parameters
        ----------
        joint_array : np.ndarray
            Numpy array of shape (N,6) where N is the number of points in the trajectory
        sampling_factor : int, optional
            Sample points in joint_array every `sampling_factor`, by default 1

        Returns
        -------
        Trajectory
        """

        if joint_array.shape[1] != 6:
            raise ValueError("Joint array must have shape (N,6)")

        setpoints = []
        for i in range(joint_array.shape[0]):
            setpoint = JointState()
            setpoint.name = ["q1", "q2", "q3", "q4", "q5", "q6"]
            setpoint.position = joint_array[i, :]
            setpoints.append(setpoint)

        return Trajectory(sampling_factor=sampling_factor, setpoints=setpoints)


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


@dataclass
class SoftRandomJointTrajectory(RandomJointTrajectory):
    """Sample two random joints positions, then create setpoint in betweeen the start and end goal.
    The number of samples is proportional to the cartesian distance between start and end.
    """

    max_dist = 0.2632

    def __post_init__(self) -> None:
        return super().__post_init__()

    @classmethod
    def generate_trajectory(cls, samples: int, samples_per_step=18):
        setpoints = []

        init_jp = RandomJointTrajectory.generate_random_joint()
        count = 0

        while count < samples:
            new_jp = RandomJointTrajectory.generate_random_joint()

            # Calculate distance between start and end point
            joints = np.vstack((np.array(init_jp).reshape(1, 6), np.array(new_jp).reshape(1, 6)))
            cp_positions = CalibrationUtils.calculate_cartesian(joints)[["X", "Y", "Z"]].to_numpy()
            dist = np.linalg.norm(cp_positions[0, :] - cp_positions[1, :])

            # Set number of samples proportional to the distance between start and end. At least use 2 samples
            num = int((dist / SoftRandomJointTrajectory.max_dist) * samples_per_step)
            num = max(2, num)
            all_setpoints = np.linspace(init_jp, new_jp, num=num)

            for idx in range(all_setpoints.shape[0]):
                setpoint = JointState()
                setpoint.name = ["q1", "q2", "q3", "q4", "q5", "q6"]
                setpoint.position = all_setpoints[idx, :].tolist()
                setpoints.append(setpoint)
                count += 1

            init_jp = setpoint.position

        return SoftRandomJointTrajectory(sampling_factor=1, setpoints=setpoints[:samples])


if __name__ == "__main__":

    arm_namespace = "PSM2"
    arm = ReplayDevice(device_namespace=arm_namespace, expected_interval=0.01)
    arm.home_device()

    init_jp = arm.measured_jp()

    # rosbag_path = Path("data/psm2_trajectories/pitch_exp_traj_03_test_cropped.bag")
    # rosbag_handle = RosbagUtils(rosbag_path)
    # trajectory = Trajectory.from_ros_bag(rosbag_handle, sampling_factor=80)
    # trajectory = RandomJointTrajectory.generate_trajectory(50)
    # trajectory = SoftRandomJointTrajectory.generate_trajectory(100, samples_per_step=28)
    trajectory = DvrkMotions.insertion_trajectory(init_jp, steps=20)

    log.info(f"Initial pt {np.array(trajectory.setpoints[0].position)}")
    log.info(f"Starting ts {trajectory.setpoints[0].header.stamp.to_sec()}")
    log.info(f"number of points {len(trajectory)}")

    trajectory_player = TrajectoryPlayer(arm, trajectory, before_motion_loop_cb=[])

    ans = input(
        'Press "y" to start data collection trajectory. Only replay trajectories that you know. '
    )
    if ans == "y":
        trajectory_player.replay_trajectory(execute_cb=True)
    else:
        exit(0)
