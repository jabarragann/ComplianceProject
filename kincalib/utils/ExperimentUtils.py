import pandas as pd
import PyKDL
from tf_conversions import posemath as pm
import numpy as np
import matplotlib.pyplot as plt
from kincalib.Atracsys.ftk_utils import markerfile2triangles, identify_marker
from kincalib.geometry import Line3D, Circle3D, Triangle3D
from pathlib import Path
from typing import Tuple

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


def calculate_midpoints(roll_df: pd.DataFrame, pitch_df: pd.DataFrame) -> Tuple[np.ndarray]:
    """Calculate pitch axis origin candidates from a roll movement and pitch movement.

    Args:
        roll_df (pd.DataFrame): [description]
        pitch_df (pd.DataFrame): [description]

    Returns:
        Tuple[np.ndarray]: [description]
    """
    marker_file = Path("./share/custom_marker_id_112.json")

    # Calculate roll axis (Shaft axis)
    roll = roll_df.q4.unique()
    marker_orig_arr = []
    for r in roll:
        df_temp = roll_df.loc[roll_df["q4"] == r]
        pose_arr, wrist_fiducials = separate_markerandfiducial(None, marker_file, df=df_temp)
        if len(pose_arr) > 0:
            marker_orig_arr.append(list(pose_arr[0].p))
    marker_orig_arr = np.array(marker_orig_arr)
    roll_circle = Circle3D.from_lstsq_fit(marker_orig_arr)
    roll_axis = Line3D(ref_point=roll_circle.center, direction=roll_circle.normal)

    # calculate pitch axis
    roll = pitch_df.q4.unique()
    fid_arr = []
    for r in roll:
        df_temp = pitch_df.loc[pitch_df["q4"] == r]
        pose_arr, wrist_fiducials = separate_markerandfiducial(None, marker_file, df=df_temp)
        fid_arr.append(wrist_fiducials)

    pitch_circle1 = Circle3D.from_lstsq_fit(fid_arr[0].T)
    pitch_circle2 = Circle3D.from_lstsq_fit(fid_arr[1].T)

    # ------------------------------------------------------------
    # Calculate mid point between rotation axis
    # ------------------------------------------------------------
    l1 = Line3D(ref_point=pitch_circle1.center, direction=pitch_circle1.normal)
    l2 = Line3D(ref_point=pitch_circle2.center, direction=pitch_circle2.normal)

    # Calculate perpendicular
    inter_params = []
    l3 = Line3D.perpendicular_to_skew(l1, l2, intersect_params=inter_params)
    midpoint1 = l3(inter_params[0][2] / 2)
    inter_params = []
    roll_axis1 = Line3D.perpendicular_to_skew(roll_axis, l1, intersect_params=inter_params)
    midpoint2 = roll_axis1(inter_params[0][2] / 2)
    inter_params = []
    roll_axis2 = Line3D.perpendicular_to_skew(roll_axis, l2, intersect_params=inter_params)
    midpoint3 = roll_axis2(inter_params[0][2] / 2)

    return midpoint1, midpoint2, midpoint3
