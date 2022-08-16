from kincalib.Geometry.geometry import Plane3D
from kincalib.Geometry.Plotter import Plotter3D

import numpy as np
import matplotlib.pyplot as plt

from kincalib.utils.Logger import Logger

log = Logger("utils_log").log


def main():

    plane1 = Plane3D(np.array([0.5, 0.5, 0.2]), 0.0)
    plane2 = Plane3D(np.array([0.5, 0.0, 0.5]), 60.0)

    pts1 = plane1.generate_pts(500)
    pts2 = plane2.generate_pts(500)

    p1_est = Plane3D.from_data(pts1)
    p2_est = Plane3D.from_data(pts2)

    plotter = Plotter3D("Plane example")
    plotter.scatter_3d(pts1.T, color="black", label="plane1")
    plotter.scatter_3d(pts2.T, color="red", label="plane2")

    log.info(f"Plane1 \n{str(plane1)}")
    log.info(f"Plane1 estimate \n{str(p1_est)}")
    log.info(f"Plane2 \n{str(plane2)}")
    log.info(f"Plane2 estimate \n{str(p2_est)}")
    # plt.show()


if __name__ == "__main__":
    main()
