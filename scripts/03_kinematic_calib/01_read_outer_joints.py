from __future__ import annotations
from doctest import OutputChecker
import os
from pathlib import Path
from re import I
from unittest import result
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from kincalib.Geometry.Plotter import Plotter3D
from kincalib.Geometry.geometry import Circle3D, Line3D
from kincalib.utils.Logger import Logger
from kincalib.Geometry.Ransac import Ransac, RansacCircle3D
from dataclasses import dataclass

log = Logger("outer").log
np.set_printoptions(precision=4, suppress=True, sign=" ")


@dataclass
class DhParamEstimator:
    axis1: AxisEstimator
    axis2: AxisEstimator

    def __post_init__(self):
        params = []
        perp_line = Line3D.perpendicular_to_skew(self.axis1.axis, self.axis2.axis, params)
        params = params[0].squeeze()
        assert perp_line.intersect(self.axis1.axis), "no intersection error"

        self.angle = np.arccos(self.axis1.circle.normal.dot(self.axis2.circle.normal)) * 180 / np.pi
        self.perp_dist = (
            np.linalg.norm(self.axis1.axis(params[0]) - self.axis2.axis(params[1])) * 1000
        )


@dataclass
class AxisEstimator:
    """Axis estimator

    Estimate the rotation axis of joint

    Parameters
    ----------
    pt : np.ndarray
        Points used in ransac fitting
    jp : np.ndarray
        Joint angles

    """

    pt: np.ndarray
    jp: np.ndarray

    def __post_init__(self):
        self.estimate_axis()

    def estimate_axis(self):

        self.circle: Circle3D = None
        self.axis: Line3D = None

        _, self.inliers_idx = Ransac.ransac(
            self.pt, model=RansacCircle3D(), n=3, k=500 * 2, t=0.5 / 1000, d=8, debug=False
        )
        self.circle = Circle3D.from_lstsq_fit(
            self.pt[self.inliers_idx]
        )  # fit again with ordered points

        self.axis = self.circle.get_ray()
        self.residual_error = self.circle.dist_pt2circle(self.pt.T)

    def plot_circle(self, plotter: Plotter3D, label: str):
        plotter.scatter_3d_inliers_outliers(self.pt.T, self.inliers_idx, label=label)
        plotter.plot_circle(self.circle)

    def plot_error(self, ax: plt.Axes, label: str):
        ax.plot(self.jp, self.residual_error * 1000, "-bo")
        ax.set_xlabel(label)
        ax.set_ylabel("dist (mm)")
        return ax


def outer_axis_analysis(path: Path, plot=False):
    outer_pitch_df = pd.read_csv(path / "outer_pitch.txt")
    outer_yaw_df = pd.read_csv(path / "outer_yaw.txt")

    # Get  marker origin position and commanded joint
    outer_yaw_df = outer_yaw_df.loc[outer_yaw_df["m_t"] == "m"][["q1", "px", "py", "pz"]]
    outer_pitch_df = outer_pitch_df.loc[outer_pitch_df["m_t"] == "m"][["q2", "px", "py", "pz"]]

    c_q1_yaw = outer_yaw_df.to_numpy()[:, 0]
    pt_yaw = outer_yaw_df.to_numpy()[:, 1:]
    c_q2_pitch = outer_pitch_df.to_numpy()[:, 0]
    pt_pitch = outer_pitch_df.to_numpy()[:, 1:]

    # Estimate parameters
    pitch_axis = AxisEstimator(pt_pitch, c_q2_pitch)
    yaw_axis = AxisEstimator(pt_yaw, c_q1_yaw)
    parameter_est = DhParamEstimator(pitch_axis, yaw_axis)

    log.info(f"Angle between normals {parameter_est.angle:0.3f} (deg)")
    log.info(f"Perpendicular distance between axis {parameter_est.perp_dist:0.3f} mm")

    # Plot data
    if plot:
        plotter = Plotter3D()
        pitch_axis.plot_circle(plotter, "outer_pitch")
        yaw_axis.plot_circle(plotter, "outer_yaw")
        plotter.plot()
        # Plot error vs command joint position
        fig, ax = plt.subplots(2)
        pitch_axis.plot_error(ax[0], "outer_pitch (deg)")
        yaw_axis.plot_error(ax[1], "outer_yaw(deg)")

        plt.show()

    return (
        parameter_est.angle,
        parameter_est.perp_dist,
        pitch_axis.residual_error,
        yaw_axis.residual_error,
    )


def main1():
    path = Path("./data/03_replay_trajectory/")

    results_dict = {"angle": [], "dist": []}
    pitch_error_dict = {"mean": [], "std": [], "max": [], "min": []}
    yaw_error_dict = {"mean": [], "std": [], "max": [], "min": []}
    for p in path.glob("*/outer_mov"):
        log.info(p)
        angle, perp_dist, pitch_error, yaw_error = outer_axis_analysis(p)
        pitch_error *= 1000
        yaw_error *= 1000
        results_dict["angle"].append(angle)
        results_dict["dist"].append(perp_dist)

        pitch_error_dict["mean"].append(pitch_error.mean())
        pitch_error_dict["std"].append(pitch_error.std())
        pitch_error_dict["max"].append(pitch_error.max())
        pitch_error_dict["min"].append(pitch_error.min())

        yaw_error_dict["mean"].append(yaw_error.mean())
        yaw_error_dict["std"].append(yaw_error.std())
        yaw_error_dict["max"].append(yaw_error.max())
        yaw_error_dict["min"].append(yaw_error.min())

    results_df = pd.DataFrame(results_dict)
    pitch_error_df = pd.DataFrame(pitch_error_dict)
    yaw_error_df = pd.DataFrame(yaw_error_dict)

    # results_df = results_df.melt(value_vars=["angle", "dist"], var_name="type", value_name="value")
    # print(results_df)
    fig, ax = plt.subplots(1, 2)
    sns.boxplot(data=results_df, y="angle", ax=ax[0])
    sns.swarmplot(data=results_df, y="angle", ax=ax[0], color="black")
    sns.boxplot(data=results_df, y="dist", ax=ax[1])
    sns.swarmplot(data=results_df, y="dist", ax=ax[1], color="black")
    plt.show()


def main2():
    # Perfect case to debug ransac!
    # path = Path("./data/03_replay_trajectory/d04-rec-11-traj01/outer_mov")
    path = Path("./data/03_replay_trajectory/d04-rec-08-traj02/outer_mov")
    outer_axis_analysis(path, plot=True)


def main3():
    # path = Path("./data/03_replay_trajectory/d04-rec-08-traj02/outer_mov")
    path = Path("./data/03_replay_trajectory/d04-rec-11-traj01/outer_mov")
    angle_list = []
    dist_list = []
    for i in range(10):
        print(i)
        angle, dist, _, _ = outer_axis_analysis(path, plot=False)
        angle_list.append(angle)
        dist_list.append(dist)

    # print(angle_list)
    # print(dist_list)

    fig, ax = plt.subplots(1)
    sns.histplot(x=angle_list, ax=ax, bins=50)
    ax.set_title("Angle distributions")
    ax.set_xlabel("degree")

    fig, ax = plt.subplots(1)
    sns.histplot(x=dist_list, ax=ax, bins=50)
    ax.set_title("distance distributions")
    ax.set_xlabel("mm")
    plt.show()


if __name__ == "__main__":

    main1()
