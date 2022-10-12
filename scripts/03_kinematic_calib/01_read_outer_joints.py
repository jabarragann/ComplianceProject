from doctest import OutputChecker
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from kincalib.Geometry.Plotter import Plotter3D
from kincalib.Geometry.geometry import Circle3D, Line3D
from kincalib.utils.Logger import Logger


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

    # Fit circle to data
    pitch_circle = Circle3D.from_lstsq_fit(pt_pitch)
    pitch_axis = pitch_circle.get_ray()
    yaw_circle = Circle3D.from_lstsq_fit(pt_yaw)
    yaw_axis = yaw_circle.get_ray()
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
        plotter.plot()

    # print(pt_pitch.shape)


def main():
    path = Path("./data/03_replay_trajectory/")

    for p in path.glob("*/outer_mov"):
        log.info(p)
        outer_axis_analysis(p)


# def main():
#     # Perfect case to debug ransac!
#     path = Path("./data/03_replay_trajectory/d04-rec-11-traj01/outer_mov")
#     outer_axis_analysis(path, plot=True)


if __name__ == "__main__":

    main()
