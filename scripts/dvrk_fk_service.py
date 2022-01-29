import rospy
from cisst_msgs.srv import QueryForwardKinematics
from sensor_msgs.msg import JointState
import tf_conversions.posemath as pm
import numpy as np
import argparse

np.set_printoptions(precision=3, suppress=True)
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
joints.position = [0.0, 0.0, 0.12, 0.0, 0.0, 0.0]
msg = kinematics_service(joints)
msg = msg.cp.pose
end_effector_frame = pm.toMatrix(pm.fromMsg(msg))

joints.position = [0.0, 0.0, 0.12]
msg = kinematics_service(joints)
msg = msg.cp.pose
third_frame = pm.toMatrix(pm.fromMsg(msg))

print("Results from the queries")
print(f"end effector frame\n{end_effector_frame}")
print(f"Third frame\n{third_frame}")
