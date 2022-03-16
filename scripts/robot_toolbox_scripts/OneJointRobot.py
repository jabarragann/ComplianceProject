import numpy as np
from numpy import pi
from spatialmath import SE3
from spatialmath.base import trnorm
from roboticstoolbox import DHRobot, RevoluteMDH

# fmt:off
robot2tracker_T= [[ -0.9131, -0.4064,  0.0317,  0.0358],
                  [ -0.0455,  0.0243, -0.9986, -0.0933],
                  [  0.4051, -0.9133, -0.0407,  0.9338],
                  [  0.0000,  0.0000,  0.0000,  1.0000]]
# fmt:on

robot2tracker_T = np.array(robot2tracker_T)
robot2tracker_T = SE3(trnorm(robot2tracker_T))

ltool = 0.4162
deg2rad = lambda x: x * pi / 180
rad2deg = lambda x: x * 180 / pi

joint_3_4 = DHRobot([RevoluteMDH(a=0.0, alpha=0.0, d=ltool, offset=0)], base=robot2tracker_T)

print(joint_3_4)

target_T = joint_3_4.fkine([deg2rad(45)])
print("Target frame")
print(target_T)

sol = joint_3_4.ikine_LM(target_T)
print(sol)
print(f"Solution: {rad2deg(sol.q)}")
print(f"Residual: {sol.residual:0.4e}")
