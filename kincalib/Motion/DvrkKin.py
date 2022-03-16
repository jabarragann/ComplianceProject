from re import I
from roboticstoolbox.robot import DHRobot
from roboticstoolbox.robot.DHLink import RevoluteDH, RevoluteMDH, PrismaticMDH
from numpy import pi
import numpy as np
from spatialmath import SE3
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
    tool_offset = np.array([ [ 0.0, -1.0,  0.0,  0.0],
                             [ 0.0,  0.0,  1.0,  0.0],
                             [-1.0,  0.0,  0.0,  0.0],
                             [ 0.0,  0.0,  0.0,  1.0]])
    # fmt:on

    def __init__(self):
        super(DvrkPsmKin, self).__init__(DvrkPsmKin.links, tool=DvrkPsmKin.tool_offset, name="DVRK PSM")

    def fkine(self, q, **kwargs) -> SE3:
        return super().fkine(q, **kwargs)

    def fkine_chain(self, q, **kwargs) -> Frame:
        """Calculate the forward kinematics for an intermediate frame of the kinematic chain.

        Args:
            q (_type_): _description_

        Returns:
            _type_: _description_
        """
        T = SE3.Empty()

        if isinstance(q, list):
            q = np.array(q)

        q = q.squeeze()
        if len(q.shape) > 1:
            raise ("q should be a one dimensional array containing the joints")

        # Calculate the forward kinematics
        Tr = SE3()
        for q, L in zip(q, self.links):
            Tr *= L.A(q)

        if self._base is not None:
            Tr = self._base * Tr

        T.append(Tr)
        return Frame.init_from_matrix(T.data[0])


if __name__ == "__main__":
    psm = DvrkPsmKin()
    print(psm)
    # log.info("3rd kinematic chain of the DVRK")
    # log.info(psm.fkine_chain([0.0, 0.0, 0.12]))
    log.info("4th kinematic chain of the DVRK")
    log.info(psm.fkine_chain([pi / 4, 0.0, 0.12, pi / 4]))
    log.info("DVRK forward kinematics")
    log.info(psm.fkine([pi / 4, 0.0, 0.12, pi / 4, pi / 4, 0]).data[0])
