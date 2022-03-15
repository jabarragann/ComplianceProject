"""
DVRK forward kinematic service example
To find service name look with the command

rosservice list

Juan Antonio Barragan
jbarrag3@jh.edu
"""

from sympy import O
import rospy
from cisst_msgs.srv import QueryForwardKinematics
from sensor_msgs.msg import JointState
import tf_conversions.posemath as pm
import numpy as np
import argparse
from kincalib.utils.Logger import Logger

log = Logger(__name__).log
np.set_printoptions(precision=4, suppress=True, sign=" ")
rospy.init_node("service_client")

parser = argparse.ArgumentParser(description="DVRK forward kinematic service example")
parser.add_argument(
    "--service_name",
    type=str,
    help="rosservice name",
    default="/PSM1/local/query_cp",
    required=False,
)

args = parser.parse_args()

########################################
## Wait for this sevice to be running
service_name = args.service_name
rospy.wait_for_service(service_name, timeout=1)
print("service {:} is available ...".format(service_name))

########################################
## Create the connection to the service.
kinematics_service = rospy.ServiceProxy(service_name, QueryForwardKinematics)
print("connected to {:} ...".format(service_name))

########################################
## Create Joint msg.
joints = JointState()
joints.name = [
    "outer_yaw",
    "outer_pitch",
    "outer_insertion",
    "outer_roll",
    "outer_wrist_pitch",
    "outer_wrist_yaw",
]

########################################
## Request kinematic frames
j1 = [np.pi / 4, 0.0, 0.12, np.pi / 4, np.pi / 4, 0.0, 0.0]

for i in range(7):
    j = j1[:i]

    joints.position = j
    msg = kinematics_service(joints)
    msg = msg.cp.pose
    end_effector_frame = pm.toMatrix(pm.fromMsg(msg))

    log.info(f"Frame {i}\n Joint values f:{j}\n{end_effector_frame}")


log.debug(msg)
