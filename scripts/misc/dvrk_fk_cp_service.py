# python
import sys
import time
import numpy as np
from numpy import pi
import argparse

from sympy import exp
from kincalib.Motion.DvrkKin import DvrkPsmKin

# ros
import rospy
from cisst_msgs.srv import QueryForwardKinematics
import tf_conversions.posemath as pm
from sensor_msgs.msg import JointState

# custom
from kincalib.Motion.ReplayDevice import ReplayDevice
from kincalib.utils.Logger import Logger

log = Logger(__name__).log
np.set_printoptions(precision=4, suppress=True, sign=" ")
rospy.init_node("service_client")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DVRK forward kinematic service example")
    parser.add_argument(
        "--service_name",
        type=str,
        help="rosservice name",
        default="/PSM2/local/query_cp",
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
    # j1 = [np.pi / 4, 0.0, 0.12, np.pi / 4, np.pi / 4, 0.0]
    # j1 = [0.0, 0.0, 0.12, 0.0, 0.0, 0.0, 0.0]

    psm = DvrkPsmKin()
    psm_crtk = ReplayDevice("PSM2", expected_interval=0.01)
    time.sleep(0.2)
    # make sure the arm is powered
    print("-- Enabling arm")
    if not psm_crtk.enable(10):
        sys.exit("-- Failed to enable within 10 seconds")
    print("-- Homing arm")
    if not psm_crtk.home(10):
        sys.exit("-- Failed to home within 10 seconds")

    j = psm_crtk.measured_jp()
    log.info(j)
    # CRTK frame
    log.info(f"Frame crtk\n{pm.toMatrix(psm_crtk.measured_cp())}")
    # Service frame
    joints.position = j
    msg = kinematics_service(joints)
    msg = msg.cp.pose
    frame_cisst = pm.toMatrix(pm.fromMsg(msg))
    log.info(f"Frame cisst service\n{frame_cisst}")
    # Custom fk model
    frame_custom = psm.fkine_chain(j)
    log.info(f"Frame custom\n{frame_custom}")
    frame_custom2 = psm.fkine(j)
    log.info(f"Frame custom2\n{frame_custom2.data[0]}")
    # log.info(f"Difference between them \n {frame_cisst-frame_custom}")
