from PyKDL import Frame
from tf_conversions import posemath as pm

from Msgs import MyJointState, MyPoseStamped


class RosConversion:
    @staticmethod
    def pykdl_to_myposestamped(name, frame: Frame):
        msg = pm.toMsg(frame)
        return MyPoseStamped.from_ros_msg(name, msg)
