from doctest import OutputChecker
import os
from pathlib import Path
from unittest import result
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from kincalib.Geometry.Plotter import Plotter3D
from kincalib.Geometry.geometry import Circle3D, Line3D
from kincalib.utils.Logger import Logger
from kincalib.Geometry.Ransac import Ransac, RansacCircle3D

log = Logger("outer").log
np.set_printoptions(precision=4, suppress=True, sign=" ")


def outer_axis_analysis(path: Path, plot=False):
    # ------------------------------------------------------------
    # plot outer joint motions
    # ------------------------------------------------------------

    # path = Path("data/03_replay_trajectory/d04-rec-23-trajrand/outer_mov/")
    # path = Path("data/03_replay_trajectory/d04-rec-20-trajsoft/outer_mov/")
    outer_pitch_df = pd.read_csv(path / "outer_pitch.txt")
    outer_yaw_df = pd.read_csv(path / "outer_yaw.txt")

    # Get only the marker origin positions
    outer_pitch_df = outer_pitch_df.loc[outer_pitch_df["m_t"] == "m"][["px", "py", "pz"]]
    outer_yaw_df = outer_yaw_df.loc[outer_yaw_df["m_t"] == "m"][["px", "py", "pz"]]

    pt_pitch = outer_pitch_df.to_numpy()
    pt_yaw = outer_yaw_df.to_numpy()

    # outer_pitch_df.to_csv("./temp/data_ransac1.csv", index=False)
    # outer_yaw_df.to_csv("./temp/data_ransac2.csv", index=False)

    # Fit circle to data
    pitch_circle: Circle3D = None
    yaw_circle: Circle3D = None

    pitch_circle, inliers_idx = Ransac.ransac(
        pt_pitch, model=RansacCircle3D(), n=3, k=500 * 2, t=0.5 / 1000, d=8, debug=True
    )
    # pitch_circle = Circle3D.from_lstsq_fit(pt_pitch)
    pitch_axis = pitch_circle.get_ray()
    pitch_residual_error = pitch_circle.dist_pt2circle(pt_pitch.T)

    yaw_circle, inliers_idx = Ransac.ransac(
        pt_yaw, model=RansacCircle3D(), n=3, k=500 * 2, t=0.5 / 1000, d=8, debug=True
    )
    # yaw_circle = Circle3D.from_lstsq_fit(pt_yaw)
    yaw_axis = yaw_circle.get_ray()
    yaw_residual_error = yaw_circle.dist_pt2circle(pt_yaw.T)

    params = []
    perp_line = Line3D.perpendicular_to_skew(pitch_axis, yaw_axis, params)
    params = params[0].squeeze()
    assert perp_line.intersect(pitch_axis), "no intersection"

    # log.info(pitch_circle)
    # log.info(yaw_circle)
    angle = np.arccos(pitch_circle.normal.dot(yaw_circle.normal)) * 180 / np.pi
    perp_dist = np.linalg.norm(pitch_axis(params[0]) - yaw_axis(params[1])) * 1000
    log.info(f"Angle between normals {angle:0.3f} (deg)")
    log.info(f"Perpendicular distance between axis {perp_dist:0.3f} mm")

    # Plot data
    if plot:
        plotter = Plotter3D()
        plotter.scatter_3d(pt_pitch.T, label="outer_pitch")
        plotter.scatter_3d(pt_yaw.T, label="outer_yaw")
        plotter.plot_circle(pitch_circle)
        plotter.plot_circle(yaw_circle)
        plotter.plot()

    # print(pt_pitch.shape)
    return angle, perp_dist, pitch_residual_error, yaw_residual_error


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
    path = Path("./data/03_replay_trajectory/d04-rec-11-traj01/outer_mov")
    # path = Path("./data/03_replay_trajectory/d04-rec-08-traj02/outer_mov")
    outer_axis_analysis(path, plot=True)
    plt.show()


if __name__ == "__main__":

    main1()
