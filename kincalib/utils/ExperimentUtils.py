import time
import pandas as pd
import PyKDL
from tf_conversions import posemath as pm
import numpy as np
import matplotlib.pyplot as plt
from kincalib.Sensors.ftk_utils import markerfile2triangles, identify_marker
from kincalib.Geometry.geometry import Line3D, Circle3D, Triangle3D
from kincalib.utils.CmnUtils import calculate_mean_frame
from pathlib import Path
from typing import Tuple, List
from collections import defaultdict
import re
from kincalib.utils.Logger import Logger

log = Logger(__name__).log
# ------------------------------------------------------------
# EXPERIMENT 02 UTILS
# ------------------------------------------------------------


def separate_markerandfiducial(
    filename, marker_file, df: pd.DataFrame = None
) -> Tuple[List[PyKDL.Frame], np.ndarray]:
    """Read atracsys data and separates the wrist fiducials from the fiducials on the shaft marker.

    Args:
        filename ([type]): [description]
        marker_file ([type]): [description]
        df (pd.DataFrame, optional): [description]. Defaults to None.

    Returns:
        pose_arr: List[PyKDL.Frame]
            list PyKDL.Frames corresponding to the all the available poses
            of the marker in the shaft

        wrist_fiducials: np.ndarray
            array of the wrist marker fiducial.

    """

    # Read df and marker files
    if df is None:
        df = pd.read_csv(filename)

    triangles_list = markerfile2triangles(marker_file)

    # split fiducials and markers
    df_f = df.loc[df["m_t"] == "f"]
    # get number of steps
    n_step = df["step"].unique()
    # Get wrist's fiducials
    # log.info(f" ")
    wrist_fiducials = []
    for n in n_step:
        step_n_d = df_f.loc[df_f["step"] == n]
        s_time = time.time()
        dd, closest_t = identify_marker(
            step_n_d.loc[:, ["px", "py", "pz"]].to_numpy(), triangles_list[0]
        )
        e_time = time.time()
        # log.info(f"identify_marker time {e_time-s_time:0.04f}")
        # log.info(f"{n} - step_n_d shape {step_n_d.shape}")
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


def calculate_midpoints(
    roll_df: pd.DataFrame, pitch_df: pd.DataFrame, other_vals_dict: dict = {}
) -> Tuple[np.ndarray]:
    """Calculate pitch axis origin candidates from a roll movement and pitch movement.

    Args:
        roll_df (pd.DataFrame): [description]
        pitch_df (pd.DataFrame): [description]
        other_vals_dict (dict): Save the mean marker frame for each pitch circle data, and the pitch1, pitch2 and
        roll vectors. To get this values pass a empty dict and the function will populate it. If None, this information is not
        calculated.

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

    # calculate pitch and yaw axis
    marker_frame_dict = []
    roll = pitch_df.q4.unique()
    fid_arr = []
    for idx, r in enumerate(roll):
        # Get pitch data
        df_temp = pitch_df.loc[(pitch_df["q4"] == r) & (pitch_df["q6"] == 0.0)]
        pose_arr, wrist_fiducials = separate_markerandfiducial(None, marker_file, df=df_temp)
        if len(pose_arr) > 0:
            mean_pose, position_std, orientation_std = calculate_mean_frame(pose_arr)
        else:
            mean_pose, position_std, orientation_std = None, None, None
        marker_frame_dict.append([mean_pose, position_std, orientation_std])
        fid_arr.append(wrist_fiducials)
        # Get yaw data

    pitch_circle1 = Circle3D.from_lstsq_fit(fid_arr[0].T)
    pitch_circle2 = Circle3D.from_lstsq_fit(fid_arr[1].T)
    yaw_circle1 = None
    yaw_circle2 = None

    # ------------------------------------------------------------
    # Populate intermediate values
    # ------------------------------------------------------------
    # Marker frame from pitch1 circle
    other_vals_dict["marker_frame_pitch1"] = marker_frame_dict[0][0]
    other_vals_dict["marker_frame1_pos_std"] = marker_frame_dict[0][1]
    other_vals_dict["marker_frame1_ori_std"] = marker_frame_dict[0][2]
    # Marker frame from pitch2 circle
    other_vals_dict["marker_frame_pitch2"] = marker_frame_dict[1][0]
    other_vals_dict["marker_frame2_pos_std"] = marker_frame_dict[1][1]
    other_vals_dict["marker_frame2_ori_std"] = marker_frame_dict[1][2]
    other_vals_dict["pitch_axis1"] = pitch_circle1.normal
    other_vals_dict["pitch_axis2"] = pitch_circle2.normal
    other_vals_dict["roll_axis"] = roll_circle.normal
    ## Todo add yaw axis to other_val_dict

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
