import numpy as np
import PyKDL
from PyKDL import Frame
from tf_conversions import posemath as pm
from geometry_msgs.msg import PoseStamped
from kincalib.Entities.Msgs import MyJointState, MyPoseStamped
from kincalib.Transforms.Rotation import Rotation3D


class RosConversion:
    @staticmethod
    def pykdl_to_myposestamped(name, frame: PyKDL.Frame):
        pose = pm.toMsg(frame)
        msg = PoseStamped()
        msg.pose = pose
        return MyPoseStamped.from_ros_msg(name, msg)

    @staticmethod
    def quaternions_to_Rotation3d(qx, qy, qz, qw):
        pose = PoseStamped()
        pose.pose.position.x = 0
        pose.pose.position.y = 0
        pose.pose.position.z = 0

        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        frame: PyKDL.Frame = pm.fromMsg(pose.pose)
        frame: np.ndarray = pm.toMatrix(frame)

        return Rotation3D(frame[:3, :3])

    @staticmethod
    def frame_to_pykdl_frame(frame: Frame):
        return pm.fromMatrix(np.array(frame))
