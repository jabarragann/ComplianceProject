# Python imports
from pathlib import Path
import time
import argparse
import sys
import black
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Set
from rich.logging import RichHandler
from rich.progress import track

# ROS and DVRK imports
import dvrk
import rospy

# kincalib module imports
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.utils.RosbagUtils import RosbagUtils
from kincalib.Geometry.Plotter import Plotter3D
from kincalib.Geometry.geometry import Line3D

np.set_printoptions(precision=4, suppress=True, sign=" ")


def main():
    log = Logger("skew_test").log
    # ------------------------------------------------------------
    # Intersection tests
    # ------------------------------------------------------------

    log.info(f"{'*'*30}\nIntersection test 1\n{'*'*30}")
    l1 = Line3D(np.array([0, 1, 1]), np.array([0, 0, -1]))
    l2 = Line3D(np.array([1, 1, 0]), np.array([1, 0, 0]))
    interesection_params = []
    log.info(f"l1 --> {l1}")
    log.info(f"l2 --> {l2}")
    intersect = l1.intersect(l2, intersect_params=interesection_params)
    log.info(f"Do they intersect? {intersect}")
    if intersect:
        log.info(f"Intersection point {l1(interesection_params[0][0])}")

    log.info(f"{'*'*30}\nIntersection test 2\n{'*'*30}")
    l1 = Line3D(np.array([-1, 1, 2]), np.array([1, 0, -1]))
    l2 = Line3D(np.array([1, 1, 0]), np.array([1, 0, 0]))
    interesection_params = []
    log.info(f"l1 --> {l1}")
    log.info(f"l2 --> {l2}")
    intersect = l1.intersect(l2, intersect_params=interesection_params)
    log.info(f"Do they intersect? {intersect}")
    if intersect:
        log.info(f"Intersection point {l1(interesection_params[0][0])}")

    # ------------------------------------------------------------
    # Skew functions test
    # ------------------------------------------------------------
    log.info(f"{'*'*30}\nSkew test 1\n{'*'*30}")
    l1 = Line3D(np.array([0, 2, 1]), np.array([0, 1, 0]))
    l2 = Line3D(np.array([1, 1, 0]), np.array([1, 0, 0]))

    log.info(f"l1 --> {l1}")
    log.info(f"l2 --> {l2}")
    log.info(f"Do they intersect? {l1.intersect(l2)}")
    skew_res = Line3D.is_skew(l1, l2)
    log.info(f"Are they lines skew? {skew_res}")
    if skew_res:
        inter_params = []
        l3 = Line3D.perpendicular_to_skew(l1, l2, intersect_params=inter_params)
        midpoint = l3(inter_params[0][2] / 2)
        log.info(f"perp to l1 and l2 --> {l3}")
        log.info(f"Intersect params: {inter_params[0].squeeze()}")
        log.info(f"Mid point {midpoint}")

    log.info(f"{'*'*30}\nSkew test 2\n{'*'*30}")
    l1 = Line3D(np.array([0, 0, 0]), np.array([1, 1, 0]))
    l2 = Line3D(np.array([1, 1, 1]), np.array([-1, 1, 0]))

    log.info(f"l1 --> {l1}")
    log.info(f"l2 --> {l2}")
    log.info(f"Do they intersect? {l1.intersect(l2)}")
    skew_res = Line3D.is_skew(l1, l2)
    log.info(f"Are they lines skew? {skew_res}")
    if skew_res:
        inter_params = []
        l3 = Line3D.perpendicular_to_skew(l1, l2, intersect_params=inter_params)
        midpoint = l3(inter_params[0][2] / 2)
        log.info(f"perp to l1 and l2 --> {l3}")
        log.info(f"Intersect params: {inter_params[0].squeeze()}")
        log.info(f"Mid point {midpoint}")

    # ------------------------------------------------------------
    # Generate points and plot
    # ------------------------------------------------------------
    plotter = Plotter3D()
    plotter.scatter_3d(l1.generate_pts(20, tmin=-2, tmax=2).T, marker="^", color="black")
    plotter.scatter_3d(l2.generate_pts(20, tmin=-2, tmax=2).T, marker="^", color="black")
    plotter.scatter_3d(l3.generate_pts(20, tmin=-2, tmax=2).T, marker="^", color="red")
    plotter.scatter_3d(midpoint.reshape((3, 1)), marker="o", color="blue", marker_size=40)
    plt.show()


if __name__ == "__main__":

    main()
