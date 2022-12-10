from dataclasses import dataclass
from typing import List
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import numpy as np


@dataclass
class MyJointState:
    ts: float = None
    name: List[str] = None
    position: List[float] = None
    velocity: List[float] = None
    effort: List[float] = None

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
        joint_state.position = msg.position
        joint_state.velocity = msg.velocity
        joint_state.effort = msg.effort

        return joint_state


class MyPoseStamped:
    name: str = None
    ts: float = None
    position: List[float] = None
    orientation: List[float] = None

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


if __name__ == "__main__":

    js_msg = JointState()
    cp_msg = PoseStamped()

    my_js = MyJointState.from_ros_msg(js_msg)
    my_cp = MyPoseStamped.from_ros_msg("test_object", cp_msg)

    print(f"print joint_state \n{my_js}")
    print(f"print pose_stamped \n{my_cp}")
