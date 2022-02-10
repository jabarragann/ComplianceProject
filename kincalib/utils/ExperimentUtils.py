import pandas as pd
import PyKDL
from tf_conversions import posemath as pm
import numpy as np
import matplotlib.pyplot as plt
from kincalib.Atracsys.ftk_utils import markerfile2triangles, identify_marker

# ------------------------------------------------------------
# EXPERIMENT 02 UTILS
# ------------------------------------------------------------
def separate_markerandfiducial(filename, marker_file, df: pd.DataFrame = None):

    # Read df and marker files
    if df is None:
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
        # log.debug(step_n_d)
        # log.debug(dd)
    # Obtain wrist axis position via lsqt
    wrist_fiducials = np.array(wrist_fiducials).squeeze().T

    # Get marker measurements
    df_m = df.loc[df["m_t"] == "m"]
    pose_arr = []
    for i in range(df_m.shape[0]):
        m = df_m.iloc[i][["px", "py", "pz"]].to_numpy()
        R = df_m.iloc[i][["qx", "qy", "qz", "qw"]].to_numpy()
        pose_arr.append(PyKDL.Frame(PyKDL.Rotation.Quaternion(*R), PyKDL.Vector(*m)))

    return pose_arr, wrist_fiducials
