import rospy
from cisst_msgs.srv import QueryForwardKinematics
from sensor_msgs.msg import JointState

print("init node")
rospy.init_node("service_client")

# wait for this sevice to be running
srv_name = "/PSM1/local/query_cp"
rospy.wait_for_service(srv_name)
print("service {:} is available ...".format(srv_name))

## Create the connection to the service.
kinematics_service = rospy.ServiceProxy(srv_name, QueryForwardKinematics)
print("connected to {:} ...".format(srv_name))

joints = JointState()
joints.name = [
    "outer_yaw",
    "outer_pitch",
    "outer_insertion",
    "outer_roll",
    "outer_wrist_pitch",
    "outer_wrist_yaw",
]
joints.position = [0.0, 0.0, 0.12, 0.0, 0.0, 0.0]

# joints.position = [0.0]
# joints.position = [0.0,0.0,0.0,0.0,0.0,0.0]

result = kinematics_service(joints)

print(result)


"""
To be continued...
(1) Service example script to share with Anton.
(2) Script showing all the dvrk frames to see if I understand correctly the kinematic model of the robot.

"""
