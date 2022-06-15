# Robotics toolbox
from roboticstoolbox.robot import DHRobot
from roboticstoolbox.robot.DHLink import RevoluteDH, RevoluteMDH, PrismaticMDH
from spatialmath.base import trnorm
from spatialmath import SE3

# Python
from numpy import pi
import numpy as np

# Custom
from kincalib.utils.Logger import Logger
from kincalib.utils.Frame import Frame

log = Logger(__name__).log
np.set_printoptions(precision=4, suppress=True, sign=" ")


class DvrkPsmKin(DHRobot):
    lrcc = 0.4318
    ltool = 0.4162
    lpitch2yaw = 0.0091
    links = [
        RevoluteMDH(a=0.0, alpha=pi / 2, d=0, offset=pi / 2),
        RevoluteMDH(a=0.0, alpha=-pi / 2, d=0, offset=-pi / 2),
        PrismaticMDH(a=0.0, alpha=pi / 2, theta=0, offset=-lrcc),
        RevoluteMDH(a=0.0, alpha=0.0, d=ltool, offset=0),
        RevoluteMDH(a=0.0, alpha=-pi / 2, d=0, offset=-pi / 2),
        RevoluteMDH(a=lpitch2yaw, alpha=-pi / 2, d=0, offset=-pi / 2),
    ]

    # fmt:off
    # Base transforms based on DVRK console configuration file
    tool_offset = np.array([[ 0.0, -1.0,  0.0,  0.0  ],
                            [ 0.0,  0.0,  1.0,  0.019],
                            [-1.0,  0.0,  0.0,  0.0  ],
                            [ 0.0,  0.0,  0.0,  1.0  ]])
    # base_transform =np.array([[  1.0,  0.0,          0.0,          0.20],
    #                           [  0.0, -0.866025404,  0.5,          0.0 ],
    #                           [  0.0, -0.5,         -0.866025404,  0.0 ],
    #                           [  0.0,  0.0,          0.0,          1.0 ]])
    # fmt:on

    def __init__(self, tool_offset=None, base_transform=None):
        if tool_offset is None:
            self.tool_offset = SE3(trnorm(DvrkPsmKin.tool_offset))
        else:
            self.tool_offset = SE3(trnorm(tool_offset))
        if base_transform is None:
            self.base_transform = SE3(trnorm(np.identity(4)))
        else:
            self.base_transform = SE3(trnorm(base_transform))

        super(DvrkPsmKin, self).__init__(
            DvrkPsmKin.links, tool=self.tool_offset, base=self.base_transform, name="DVRK PSM"
        )

    def fkine(self, q, **kwargs) -> SE3:
        return super().fkine(q, **kwargs)

    def fkine_chain(self, q, ignore_base=False, **kwargs) -> Frame:
        """Calculate the forward kinematics for an intermediate frame of the kinematic chain.

        Args:
            q (_type_): _description_

        Returns:
            _type_: _description_
        """
        T = SE3.Empty()

        if isinstance(q, list):
            q = np.array(q)
        if len(q.shape) > 1:
            q = q.squeeze()
        if len(q.shape) > 1:
            raise Exception("q should be a one dimensional array containing the joints")
        if q.shape[0] > len(self.links):
            raise Exception("q should not exceed the number of joints")

        # Calculate the forward kinematics
        Tr = SE3()
        for qi, L in zip(q, self.links):
            Tr *= L.A(qi)

        if self.base_transform is not None and not ignore_base:
            Tr = self.base_transform * Tr
        # Only add the base transformation if all the joint values are given.
        if q.shape[0] == len(self.links):
            Tr = Tr * self._tool

        T.append(Tr)
        return T.data[0]


if __name__ == "__main__":
    psm = DvrkPsmKin()
    print(psm)
    j = [pi / 4, 0.0, 0.12, pi / 4, pi / 4, 0]
    # log.info("3rd kinematic chain of the DVRK")
    # log.info(psm.fkine_chain([0.0, 0.0, 0.12]))
    log.info("4th kinematic chain of the DVRK")
    log.info(psm.fkine_chain(j[:4]))
    log.info("dvrk fkine func")
    log.info(psm.fkine(j).data[0])  # .SE3.data[0] returns a ndarray
    log.info("dvrk fkine_chain func")
    log.info(psm.fkine_chain(j))
