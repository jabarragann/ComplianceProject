from dataclasses import dataclass
from typing import List
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped


@dataclass
class MyJointState:
    ts: float
    name: List[str]
    position: List[float]
    velocity: List[float]
    effort: List[float]

    def __post_init__():
        pass

    @classmethod
    def from_ros_msg(cls, msg: JointState):
        joint_state = MyJointState()
        joint_state.ts = msg.header.stamp.secs + msg.header.stamp.nsecs / 10e9
        joint_state.position = msg.position
        joint_state.velocity = msg.velocity
        joint_state.effort = msg.effort


class MyPoseStamped:
    name: str
    ts: float
    position: List[float]
    orientation: List[float]

    def __post_init__():
        pass

    @classmethod
    def from_ros_msg(cls, name, msg: PoseStamped):
        pose_stamped = MyPoseStamped()
        pose_stamped.name = name
        pose_stamped.position = msg.pose.position
        pose_stamped.orientation = msg.pose.orientation
