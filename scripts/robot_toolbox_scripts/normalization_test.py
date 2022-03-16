from roboticstoolbox.robot.DHLink import PrismaticMDH, RevoluteMDH
from roboticstoolbox.robot import DHRobot
from numpy import pi
import numpy as np
from spatialmath import SE3
from kincalib.utils.Frame import Frame
from spatialmath.base import trnorm
from kincalib.utils.Frame import Frame

# OPEN an issue asking why the I need to renormalize my matrix twice in a row? This only happens when I use
# my frame class.
# Also suggest that the error message needs to be improved.

# fmt:off
robot2tracker_T= [[ -0.9131, -0.4064,  0.0317,  0.0358],
                  [ -0.0455,  0.0243, -0.9986, -0.0933],
                  [  0.4051, -0.9133, -0.0407,  0.9338],
                  [  0.0000,  0.0000,  0.0000,  1.0000]]
robot2tracker_T = np.array(robot2tracker_T)
print(type(robot2tracker_T))
robot2tracker_T = Frame.init_from_matrix(robot2tracker_T)
robot2tracker_T = np.array(robot2tracker_T) 
print(type(robot2tracker_T))

# robot2tracker_T = trnorm(robot2tracker_T) 
# robot2tracker_T = trnorm(trnorm(robot2tracker_T))
robot2tracker_T = SE3(trnorm(robot2tracker_T))
# robot2tracker_T = SE3(trnorm(trnorm(robot2tracker_T)))

# robot2tracker_T = SE3(robot2tracker_T)

print(robot2tracker_T)
