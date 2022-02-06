import pandas as pd
import PyKDL
from tf_conversions import posemath as pm
import numpy as np
import matplotlib.pyplot as plt
import dvrk
from pathlib import Path
from kincalib.utils.Logger import Logger
from kincalib.utils.SavingUtilities import save_without_overwritting
from kincalib.Atracsys.ftk_utils import markerfile2triangles, identify_marker
from kincalib.Atracsys.ftk_500_api import ftk_500
from kincalib.geometry import Circle3D, Plotter3D
import argparse
import sys
from pathlib import Path
import time


def main():
    """Analyze pitch exp 02 data.
       Calculate distance between pitch axis and shaft marker's origin
    """

    # ------------------------------------------------------------
    # Setup 
    # ------------------------------------------------------------
    parser = argparse.ArgumentParser()
    #fmt:off
    parser.add_argument( "-f", "--file", type=str, default="pitch_exp01.txt", 
                         help="file to analyze") 
    parser.add_argument( "-l", "--log", type=str, default="INFO", 
                         help="log level") #fmt:on

    args = parser.parse_args()
    # Important paths
    log_level = args.log
    log = Logger("pitch_exp_analize", log_level=log_level).log
    root = Path("./data/02_pitch_experiment/")
    marker_file = Path("./share/custom_marker_id_112.json")

    filename = root / args.file 
    if not filename.exists():
        log.error("file not found")
        sys.exit(0)
    if not marker_file.exists():
        log.error("marker file not found")
        sys.exit(0)

    # ------------------------------------------------------------
    # Calculate distance between pitch axis and shaft marker's origin
    # ------------------------------------------------------------
    # Read data and marker file
    df = pd.read_csv(filename)
    triangles_list = markerfile2triangles(marker_file)
    # split fiducials and markers
    df_f = df.loc[df["m_t"] == "f"]
    # get number of steps
    n_step = df["step"].to_numpy().max()
    # Get wrist's fiducials
    wrist_fiducials = []
    for n in range(n_step + 1):
        step_n_d = df_f.loc[df_f["step"] == n]
        dd, closest_t = identify_marker(
            step_n_d.loc[:, ["px", "py", "pz"]].to_numpy(), triangles_list[0]
        )
        if len(dd["other"]) > 0:
            wrist_fiducials.append(step_n_d.iloc[dd["other"]][["px", "py", "pz"]].to_numpy())
        log.debug(step_n_d)
        log.debug(dd)
    # Obtain wrist axis position via lsqt
    wrist_fiducials = np.array(wrist_fiducials).squeeze().T
    est_circle = Circle3D.from_sphere_lstsq(wrist_fiducials)

    # Obtain mean marker frame - Only the wrist was moving, which means there
    # should be no variation in the frame position.
    df_m = df.loc[df["m_t"] == "m"]
    pose_arr = []
    for i in range(df_m.shape[0]):
        m = df_m.iloc[i][["px", "py", "pz"]].to_numpy()
        R = df_m.iloc[i][["qx", "qy", "qz", "qw"]].to_numpy()
        pose_arr.append(PyKDL.Frame(PyKDL.Rotation.Quaternion(*R), PyKDL.Vector(*m)))
    mean_frame, pos_std, r_std = ftk_500.average_marker_pose(pose_arr)

    # ------------------------------------------------------------
    # Show results
    # ------------------------------------------------------------
    log.info(f"Analysing file: {args.file}")
    log.debug(f"CIRCLE ESTIMATION TO FIND PITCH AXIS")
    log.debug(f"estimated radius: {est_circle.radius:.04f}")
    log.debug(f"estimated center: {est_circle.center.squeeze()}")
    log.debug(f"estimated normal: {est_circle.normal}")
    log.debug(f"MEAN MARKER POSITION")
    log.debug(f"mean frame: \n {pm.toMatrix(mean_frame)}")
    log.debug(f"position std:    {pos_std}")
    log.debug(f"orientation std: {r_std}")
    log.info(f"DISTANCE BETWEEN MARKER ORIGIN AND PITCH AXIS")
    dist = np.array(list(mean_frame.p)) - est_circle.center.squeeze()
    log.info(f"direction (marker-pitch_ax): {dist}")
    log.info(f"mag: {np.linalg.norm(dist):0.4f}")
    if log_level == "DEBUG":
        Plotter3D.scatter_3d(wrist_fiducials)


if __name__ == "__main__":

    main()
