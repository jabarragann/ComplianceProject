from kincalib.Geometry.geometry import Plane3D
from kincalib.Geometry.Plotter import Plotter3D

import numpy as np
import matplotlib.pyplot as plt
from kincalib.utils.CmnUtils import mean_std_str

from kincalib.utils.Logger import Logger

log = Logger("utils_log").log


def main():

    plane1 = Plane3D(np.array([0.5, 0.5, 0.2]), 0.0)
    plane2 = Plane3D(np.array([0.5, 0.0, 0.5]), -40.0)
    plane3 = Plane3D(np.array([0.0, 0.0, 1.0]), -80.0)

    pts1 = plane1.generate_pts(500)
    pts2 = plane2.generate_pts(500, noise_std=3)
    pts3 = plane3.generate_pts(500, noise_std=9)

    dist_vect_p1 = plane1.point_cloud_dist(pts1)
    dist_vect_p2 = plane2.point_cloud_dist(pts2)
    dist_vect_p3 = plane3.point_cloud_dist(pts3)

    p1_est = Plane3D.from_data(pts1)
    p2_est = Plane3D.from_data(pts2)
    p3_est = Plane3D.from_data(pts3)

    plotter = Plotter3D("Plane example")
    plotter.scatter_3d(pts1.T, color="black", label="plane1")
    plotter.scatter_3d(pts2.T, color="red", label="plane2")
    plotter.scatter_3d(pts3.T, color="green", label="plane3")

    log.info(f"Plane1 \n{str(plane1)}")
    log.info(f"Plane1 estimate \n{str(p1_est)}")
    log.info(f"Distance to plane1: {mean_std_str(dist_vect_p1.mean(),dist_vect_p1.std())} ")

    log.info(f"Plane2 \n{str(plane2)}")
    log.info(f"Plane2 estimate \n{str(p2_est)}")
    log.info(f"Distance to plane1: {mean_std_str(dist_vect_p2.mean(),dist_vect_p2.std())} ")

    log.info(f"Plane3 \n{str(plane3)}")
    log.info(f"Plane3 estimate \n{str(p3_est)}")
    log.info(f"Distance to plane3: {mean_std_str(dist_vect_p3.mean(),dist_vect_p3.std())} ")

    plt.show()


def main2():

    plane3 = Plane3D(np.array([0.0, 0.0, 1.0]), -80.0)

    pts3 = plane3.generate_pts(500, noise_std=3)

    p3_est = Plane3D.from_data(pts3)

    plotter = Plotter3D("Plane example")
    plotter.ax.set_zlim(0, 100)
    plotter.scatter_3d(pts3.T, color="green", label="plane3")

    log.info(f"Plane2 \n{str(plane3)}")
    log.info(f"Plane2 estimate \n{str(p3_est)}")
    plt.show()


if __name__ == "__main__":
    main()
