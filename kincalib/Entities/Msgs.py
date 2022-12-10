from __future__ import annotations
from dataclasses import dataclass
from typing import List
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import numpy as np


@dataclass
class MyJointState:
    ts: float = None
    name: List[str] = None
    position: np.ndarray = None
    velocity: np.ndarray = None
    effort: np.ndarray = None

    def __post_init__(self):
        pass

    def __str__(self):
        msg = (
            f"ts: {self.ts}\n"
            + f"joints name: {self.name}\n"
            + f"position: {self.position}\n"
            + f"velocity: {self.velocity}\n"
            + f"effort: {self.effort}\n"
        )
        return msg

    @classmethod
    def from_ros_msg(cls, msg: JointState):
        joint_state = MyJointState()
        joint_state.ts = msg.header.stamp.secs + msg.header.stamp.nsecs / 10e9
        joint_state.position = np.array(msg.position)
        joint_state.velocity = np.array(msg.velocity)
        joint_state.effort = np.array(msg.effort)

        return joint_state


@dataclass
class MyPoseStamped:
    name: str = None
    ts: float = None
    position: np.ndarray = None
    orientation: np.ndarray = None

    def __post_init__(self):
        pass

    def __str__(self):
        msg = (
            f"ts: {self.ts}\n"
            + f"object name: {self.name}\n"
            + f"Position {self.position}\n"
            + f"orientation: {self.orientation}\n"
        )
        return msg

    @classmethod
    def from_ros_msg(cls, name, msg: PoseStamped):
        pose_stamped = MyPoseStamped()
        pose_stamped.name = name
        pose_stamped.position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        )
        pose_stamped.orientation = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
        )

        return pose_stamped

    @classmethod
    def from_array_of_positions(cls, name, arr: np.ndarray) -> List[MyPoseStamped]:
        """Generate a list[MyPoseStamped] from a numpy array of positions

        Parameters
        ----------
        name : str
        arr : np.ndarray
            array of shape (N,3) where N is the total number of points

        Returns
        -------
        List[MyPoseStamped]
        """
        assert arr.shape[1] == 3, "Incorrect shape for `arr`"
        result = []
        for k in range(arr.shape[0]):
            pose = MyPoseStamped(name, position=arr[k, :].squeeze())
            result.append(pose)
        return result


if __name__ == "__main__":

    js_msg = JointState()
    cp_msg = PoseStamped()

    my_js = MyJointState.from_ros_msg(js_msg)
    my_cp = MyPoseStamped.from_ros_msg("test_object", cp_msg)

    print(f"print joint_state \n{my_js}")
    print(f"print pose_stamped \n{my_cp}")
