from roboticstoolbox import ETS as ET
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
from spatialmath import SE3
import numpy as np

pitch2yaw = 0.0092
# end_effector = SE3(0.1, 0.0, 0.1)
end_effector = SE3(0.0, 0.0, 0.0)

E = ET.rz() * ET.tx(pitch2yaw) * ET.rx(90, "deg") * ET.rz() * ET.tx(0.01) * ET.tz(0.01)
robot = rtb.ERobot(E, name="test", tool=end_effector)
rtb.ERobot2

print(E)
print(robot.fkine([30, 30], "deg"))
print(robot)
# robot.plot(30, block=True)
robot.teach([30, 30], block=True)
