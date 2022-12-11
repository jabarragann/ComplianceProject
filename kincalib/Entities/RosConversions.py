from PyKDL import Frame
from tf_conversions import posemath as pm
from geometry_msgs.msg import PoseStamped
from kincalib.Entities.Msgs import MyJointState, MyPoseStamped


class RosConversion:
    @staticmethod
    def pykdl_to_myposestamped(name, frame: Frame):
        pose = pm.toMsg(frame)
        msg = PoseStamped()
        msg.pose = pose
        return MyPoseStamped.from_ros_msg(name, msg)
