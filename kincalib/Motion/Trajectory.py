from __future__ import annotations

# Python
from dataclasses import dataclass, field
import numpy as np

# ros
from sensor_msgs.msg import JointState

# Custom
from kincalib.Calibration.CalibrationUtils import CalibrationUtils
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.utils.Logger import Logger

log = Logger(__name__).log


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
